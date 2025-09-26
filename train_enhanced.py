# train_enhanced.py
"""
Enhanced training script with current modality boosting for balanced three-stage welding fusion
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import sys
import os

# Add paths for imports
sys.path.append('.')
sys.path.append('./OmniFuse')
sys.path.append('./models')
sys.path.append('./config')
sys.path.append('./training')

from config.enhanced_config import *
from OmniFuse.dataset_multimodal import MultiModalWeldDataset
from OmniFuse.encoders import ImageEncoder, SoundEncoder
from models.enhanced_bottleneck_fusion import EnhancedBottleneckFusion
from training.balanced_three_stage_trainer import BalancedThreeStageTrainer
from OmniFuse.utils import build_file_list, detect_current_len


def create_data_loaders(config):
    """Create train and validation data loaders"""
    print("Building file list...")
    file_list = build_file_list()
    print(f"Found {len(file_list)} samples")
    
    # Auto-detect current signal length if needed
    if hasattr(config, 'AUTO_DETECT_CURRENT_LEN') and config.AUTO_DETECT_CURRENT_LEN:
        current_len = detect_current_len(CURRENT_ROOT, CLASSES, sample_num=50)
        print(f"Auto-detected current signal length: {current_len}")
        config.CURRENT_LEN = current_len
    
    # Create dataset
    dataset = MultiModalWeldDataset(
        IMAGE_ROOT, SOUND_ROOT, CURRENT_ROOT, 
        file_list=file_list,
        current_len=config.CURRENT_LEN
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def create_model(config, device):
    """Create the enhanced model with current boosting"""
    print("Creating enhanced model...")
    
    # Create individual encoders
    image_encoder = ImageEncoder(config.IMAGE_DIM, pretrained=True).to(device)
    sound_encoder = SoundEncoder(config.SOUND_DIM).to(device)
    
    # Create enhanced fusion model
    model = EnhancedBottleneckFusion(
        image_dim=config.IMAGE_DIM,
        sound_dim=config.SOUND_DIM,
        current_len=config.CURRENT_LEN,
        current_dim=config.CURRENT_DIM,
        num_classes=config.NUM_CLASSES,
        cra_k=config.K,
        beta=config.BETA,
        current_boost=config.CURRENT_BOOST
    ).to(device)
    
    print(f"Model created with parameters:")
    print(f"  - Image dim: {config.IMAGE_DIM}")
    print(f"  - Sound dim: {config.SOUND_DIM}")
    print(f"  - Current len: {config.CURRENT_LEN}")
    print(f"  - Current dim: {config.CURRENT_DIM}")
    print(f"  - Current boost: {config.CURRENT_BOOST}")
    print(f"  - Beta: {config.BETA}")
    
    return model, image_encoder, sound_encoder


def test_missing_modality(model, image_encoder, sound_encoder, val_loader, device, config):
    """Test model performance with missing modalities"""
    print("\n=== Missing Modality Analysis ===")
    
    model.eval()
    image_encoder.eval()
    sound_encoder.eval()
    
    scenarios = {
        'complete': {'image': True, 'sound': True, 'current': True},
        'missing_image': {'image': False, 'sound': True, 'current': True},
        'missing_sound': {'image': True, 'sound': False, 'current': True},
        'missing_current': {'image': True, 'sound': True, 'current': False}
    }
    
    results = {}
    
    with torch.no_grad():
        for scenario_name, mask in scenarios.items():
            correct = 0
            total = 0
            
            for images, sounds, currents, labels in val_loader:
                images = images.to(device)
                sounds = sounds.to(device)
                currents = currents.to(device)
                labels = labels.to(device)
                
                # Get modality features
                img_features = image_encoder(images) if mask['image'] else torch.zeros(images.size(0), config.IMAGE_DIM).to(device)
                snd_features = sound_encoder(sounds) if mask['sound'] else torch.zeros(sounds.size(0), config.SOUND_DIM).to(device)
                cur_signal = currents if mask['current'] else torch.zeros_like(currents)
                
                # Forward pass
                outputs = model(img_features, snd_features, cur_signal, labels)
                
                _, predicted = outputs['logits_joint'].max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            accuracy = 100. * correct / total
            results[scenario_name] = accuracy
            print(f"{scenario_name:15}: {accuracy:.2f}%")
    
    # Calculate performance drops
    complete_acc = results['complete']
    print(f"\nPerformance drops from complete data:")
    print(f"Missing image:   {complete_acc - results['missing_image']:.2f} percentage points")
    print(f"Missing sound:   {complete_acc - results['missing_sound']:.2f} percentage points")
    print(f"Missing current: {complete_acc - results['missing_current']:.2f} percentage points")
    
    return results


def main():
    """Main training function"""
    print("=== Enhanced OmniFuse Training with Current Boosting ===")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration - create a config object
    class Config:
        pass
    
    config = Config()
    
    # Copy all config variables
    for key in globals():
        if key.isupper() and not key.startswith('_'):
            setattr(config, key, globals()[key])
    
    # Create data loaders
    try:
        train_loader, val_loader = create_data_loaders(config)
    except Exception as e:
        print(f"Warning: Could not create data loaders: {e}")
        print("Creating dummy data loaders for testing...")
        
        # Create dummy dataset for testing
        class DummyDataset:
            def __init__(self, size=100):
                self.size = size
                
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                return (
                    torch.randn(3, 224, 224),  # image
                    torch.randn(1, 128, 87),   # sound mel-spectrogram
                    torch.randn(100),          # current signal
                    torch.randint(0, 6, (1,)).item()  # label
                )
        
        train_dataset = DummyDataset(800)
        val_dataset = DummyDataset(200)
        
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        print("Using dummy data for demonstration")
    
    # Create model
    model, image_encoder, sound_encoder = create_model(config, device)
    
    # Create trainer
    trainer = BalancedThreeStageTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Train model
    print("\nStarting three-stage training...")
    history = trainer.train_three_stage()
    
    # Test missing modality performance
    results = test_missing_modality(
        model, image_encoder, sound_encoder, val_loader, device, config
    )
    
    # Print summary
    print("\n=== Training Summary ===")
    print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
    print("\nCurrent modality importance analysis:")
    if 'current_weight' in history and history['current_weight']:
        avg_current_weight = sum(history['current_weight']) / len(history['current_weight'])
        print(f"Average current modality weight: {avg_current_weight:.3f}")
    
    print("\nExpected outcome:")
    if 'missing_current' in results:
        current_drop = results['complete'] - results['missing_current']
        print(f"Current performance drop: {current_drop:.2f} percentage points")
        if current_drop >= 5.0:
            print("✅ SUCCESS: Current modality is now properly utilized!")
        else:
            print("⚠️  Current modality may need further enhancement")
    
    return model, history, results


if __name__ == "__main__":
    model, history, results = main()