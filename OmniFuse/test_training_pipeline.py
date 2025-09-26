#!/usr/bin/env python3
"""
Test script to validate the complete training pipeline with dummy data.
This tests that all components work together without errors.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
import tempfile
import numpy as np
from PIL import Image
from torchvision import transforms

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_comprehensive_dummy_data():
    """Create a complete dummy dataset for testing."""
    temp_base = tempfile.mkdtemp()
    
    # Define paths
    image_root = os.path.join(temp_base, "image")
    sound_root = os.path.join(temp_base, "sound")
    current_root = os.path.join(temp_base, "current")
    
    classes = ["normal", "defect"]
    samples_per_class = 10
    
    # Create directories and files
    for cls in classes:
        os.makedirs(os.path.join(image_root, cls), exist_ok=True)
        os.makedirs(os.path.join(sound_root, cls), exist_ok=True)
        os.makedirs(os.path.join(current_root, cls), exist_ok=True)
        
        for i in range(samples_per_class):
            base_name = f"{cls}_{i:03d}"
            
            # Create dummy image
            img = Image.new('RGB', (224, 224), color=(128, 128, 128))
            img.save(os.path.join(image_root, cls, f"{base_name}.jpg"))
            
            # Create dummy sound file (just create the path, dataset handles missing gracefully)
            # We could create actual audio files but for testing it's not necessary
            
            # Create dummy current data with realistic length variation
            current_len = np.random.randint(2000, 5000)
            current_data = np.random.randn(current_len).astype(np.float32) * 10 + 50  # Realistic current values
            np.save(os.path.join(current_root, cls, f"{base_name}.npy"), current_data)
    
    return temp_base, image_root, sound_root, current_root, classes

def test_complete_pipeline():
    """Test the complete training pipeline end-to-end."""
    try:
        logger.info("Creating dummy dataset...")
        temp_base, image_root, sound_root, current_root, classes = create_comprehensive_dummy_data()
        
        # Import components
        from dataset_multimodal import MultiModalWeldDataset
        from encoders import ImageEncoder, SoundEncoder, CurrentEncoder
        from cra import CRA
        from dwfuse import DWFuse
        from utils import detect_current_len
        
        # Build file list with proper label mapping
        file_list = []
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        for cls in classes:
            cls_dir = os.path.join(image_root, cls)
            for fname in os.listdir(cls_dir):
                if fname.endswith(".jpg"):
                    base = os.path.splitext(fname)[0]
                    file_list.append((cls, base))
        
        logger.info(f"Created {len(file_list)} dummy samples")
        
        # Detect current length
        current_len = detect_current_len(current_root, classes, sample_num=20)
        logger.info(f"Detected current length: {current_len}")
        
        # Create dataset with transforms
        img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = MultiModalWeldDataset(
            image_root, sound_root, current_root,
            file_list[:10],  # Use subset for testing
            current_len=current_len,
            enable_augmentation=True,
            transform_img=img_transform,
            classes=classes  # Pass the classes we're using
        )
        
        # Create data loader
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        logger.info("Testing data loading...")
        for batch_idx, (imgs, mels, curs, labels) in enumerate(loader):
            logger.info(f"Batch {batch_idx}: imgs={imgs.shape}, mels={mels.shape}, curs={curs.shape}, labels={labels.shape}")
            if batch_idx >= 2:  # Test a few batches
                break
        
        # Test model components
        logger.info("Testing model components...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create models
        img_enc = ImageEncoder(512).to(device)
        snd_enc = SoundEncoder(256).to(device)
        cur_enc = CurrentEncoder(in_len=current_len, out_dim=128).to(device)
        cra = CRA(dim=(512+256+128), K=5).to(device)
        dw = DWFuse([512,256,128], len(classes), beta=0.5).to(device)
        
        # Test forward pass
        logger.info("Testing forward pass...")
        imgs, mels, curs, labels = next(iter(loader))
        imgs, mels, curs, labels = imgs.to(device), mels.to(device), curs.to(device), labels.to(device)
        
        # Forward pass through encoders
        v_img = img_enc(imgs)
        v_snd = snd_enc(mels)
        v_cur = cur_enc(curs)
        v_concat = torch.cat([v_img, v_snd, v_cur], dim=1)
        
        logger.info(f"Encoder outputs: img={v_img.shape}, snd={v_snd.shape}, cur={v_cur.shape}")
        
        # CRA processing
        v_prime = cra.forward_impute(v_concat)
        v_double = cra.backward_impute(v_prime)
        
        logger.info(f"CRA outputs: prime={v_prime.shape}, double={v_double.shape}")
        
        # DWFuse
        v_img_p = v_prime[:, :512]
        v_snd_p = v_prime[:, 512:768]
        v_cur_p = v_prime[:, 768:]
        
        logits_joint, per_logits, omegas = dw([v_img_p, v_snd_p, v_cur_p], labels=labels)
        
        logger.info(f"DWFuse outputs: joint={logits_joint.shape}, per_logits={[l.shape for l in per_logits]}, omegas={omegas.shape}")
        
        # Test loss computation
        criterion = nn.CrossEntropyLoss()
        
        L_forward = ((v_prime - v_concat)**2).mean()
        L_backward = ((v_double - v_concat)**2).mean()
        per_losses = [criterion(l, labels)*omegas[:,i].mean() for i,l in enumerate(per_logits)]
        L_enc = sum(per_losses)
        L_ra = criterion(logits_joint, labels)
        L_DWFuse = L_enc + 0.1 * L_ra
        
        total_loss = L_forward + 1.0*L_backward + 10.0*L_DWFuse
        
        logger.info(f"Loss computation successful: total_loss={total_loss.item():.4f}")
        
        # Test backward pass
        logger.info("Testing backward pass...")
        total_loss.backward()
        logger.info("✅ Backward pass successful")
        
        # Test optimizer step
        params = list(img_enc.parameters()) + list(snd_enc.parameters()) + \
                 list(cur_enc.parameters()) + list(cra.parameters()) + list(dw.parameters())
        optimizer = optim.Adam(params, lr=1e-3)
        optimizer.step()
        logger.info("✅ Optimizer step successful")
        
        # Clean up
        import shutil
        shutil.rmtree(temp_base)
        
        logger.info("🎉 Complete pipeline test PASSED! All components working correctly.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_augmentation_effectiveness():
    """Test that augmentation actually changes the data appropriately."""
    logger.info("Testing augmentation effectiveness...")
    
    try:
        # Create test data
        original_data = torch.randn(1000) * 10 + 50  # Simulated current data
        
        from dataset_multimodal import MultiModalWeldDataset
        dataset = MultiModalWeldDataset(
            "/tmp", "/tmp", "/tmp", [("test", "test")],
            current_len=1000, enable_augmentation=True, augmentation_prob=1.0  # Always augment
        )
        
        # Test multiple augmentations
        augmented_samples = []
        for _ in range(10):
            augmented = dataset._apply_safe_augmentation(original_data.clone())
            augmented_samples.append(augmented)
        
        # Check that augmentations are different but reasonable
        diffs = [torch.abs(aug - original_data).mean().item() for aug in augmented_samples]
        
        logger.info(f"Augmentation differences: min={min(diffs):.4f}, max={max(diffs):.4f}, mean={np.mean(diffs):.4f}")
        
        # Ensure augmentations are not identical but not too different
        assert max(diffs) > 0.01, "Augmentation too weak"
        assert max(diffs) < original_data.std().item() * 0.1, "Augmentation too strong"
        
        logger.info("✅ Augmentation effectiveness test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Augmentation test FAILED: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting complete training pipeline validation...")
    
    tests = [
        ("Complete Pipeline", test_complete_pipeline),
        ("Augmentation Effectiveness", test_augmentation_effectiveness),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info('='*60)
        
        if test_func():
            passed += 1
            logger.info(f"✅ {test_name} PASSED")
        else:
            failed += 1
            logger.error(f"❌ {test_name} FAILED")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL RESULTS")
    logger.info('='*60)
    logger.info(f"Passed: {passed}/{len(tests)}")
    logger.info(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        logger.info("🎉 ALL TESTS PASSED! Training pipeline ready for use.")
    else:
        logger.error(f"❌ {failed} test(s) failed. Review the issues above.")