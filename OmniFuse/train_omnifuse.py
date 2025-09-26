# train_omnifuse.py
import torch, torch.nn as nn, torch.optim as optim
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import *
from dataset_multimodal import MultiModalWeldDataset
from encoders import ImageEncoder, SoundEncoder, CurrentEncoder
from cra import CRA
from dwfuse import DWFuse
from tla import TLAMiner
from utils import build_file_list, detect_current_len

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    logger.info(f"Using device: {device}")
    
    # Build file list and detect optimal current length
    file_list = build_file_list()
    logger.info(f"Total samples: {len(file_list)}")
    
    # Detect current sequence length from all data before creating dataset
    current_len = detect_current_len(CURRENT_ROOT, CLASSES, sample_num=100)
    logger.info(f"Detected optimal current sequence length: {current_len}")
    
    # Split data into train/val/test before creating datasets to ensure consistent processing
    train_files, temp_files = train_test_split(file_list, test_size=0.3, random_state=42, stratify=[cls for cls, _ in file_list])
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42, stratify=[cls for cls, _ in temp_files])
    
    logger.info(f"Train samples: {len(train_files)}, Val samples: {len(val_files)}, Test samples: {len(test_files)}")
    
    # Create datasets with consistent current_len and data augmentation settings
    train_dataset = MultiModalWeldDataset(
        IMAGE_ROOT, SOUND_ROOT, CURRENT_ROOT, 
        train_files, current_len=current_len, enable_augmentation=True
    )
    val_dataset = MultiModalWeldDataset(
        IMAGE_ROOT, SOUND_ROOT, CURRENT_ROOT, 
        val_files, current_len=current_len, enable_augmentation=False
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Models - Fix current encoder input dimension
    img_enc = ImageEncoder(512).to(device)
    snd_enc = SoundEncoder(256).to(device)
    cur_enc = CurrentEncoder(in_len=current_len, out_dim=128).to(device)  # Use detected length
    cra = CRA(dim=(512+256+128), K=K).to(device)
    dw = DWFuse([512,256,128], NUM_CLASSES, beta=BETA).to(device)

    # Optimizer and criterion
    params = list(img_enc.parameters()) + list(snd_enc.parameters()) + \
             list(cur_enc.parameters()) + list(cra.parameters()) + list(dw.parameters())
    opt = optim.Adam(params, lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    tla_miner = TLAMiner(M=3)

    logger.info("Starting training...")
    
    # Training loop with proper validation and error handling
    for epoch in range(EPOCHS):
        # Training phase
        img_enc.train()
        snd_enc.train()
        cur_enc.train()
        cra.train()
        dw.train()
        
        train_loss = 0.0
        train_samples = 0
        
        try:
            for batch_idx, (imgs, mels, cur, labels) in enumerate(train_loader):
                # Validate data shapes
                if cur.shape[1] != current_len:
                    logger.warning(f"Current sequence length mismatch: expected {current_len}, got {cur.shape[1]}")
                
                imgs, mels, cur, labels = imgs.to(device), mels.to(device), cur.to(device), labels.to(device)

                # Forward pass
                v_img = img_enc(imgs)
                v_snd = snd_enc(mels)
                v_cur = cur_enc(cur)
                v_concat = torch.cat([v_img, v_snd, v_cur], dim=1)

                # CRA processing
                v_prime = cra.forward_impute(v_concat)
                v_double = cra.backward_impute(v_prime)

                # Extract modality features
                v_img_p = v_prime[:, :512]
                v_snd_p = v_prime[:, 512:768]
                v_cur_p = v_prime[:, 768:]

                # DWFuse
                logits_joint, per_logits, omegas = dw([v_img_p, v_snd_p, v_cur_p], labels=labels)

                # Loss computation
                L_forward = ((v_prime - v_concat)**2).mean()
                L_backward = ((v_double - v_concat)**2).mean()
                per_losses = [criterion(l, labels)*omegas[:,i].mean() for i,l in enumerate(per_logits)]
                L_enc = sum(per_losses)
                L_ra = criterion(logits_joint, labels)
                L_DWFuse = L_enc + ALPHA * L_ra
                L_TLA = 0.0  # Placeholder for TLA loss

                loss = L_forward + LAMBDA1*L_backward + LAMBDA2*L_DWFuse + LAMBDA3*L_TLA

                # Optimization step
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                train_loss += loss.item()
                train_samples += len(imgs)
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        except Exception as e:
            logger.error(f"Training error at epoch {epoch+1}: {e}")
            raise

        # Validation phase
        if val_loader and len(val_loader) > 0:
            img_enc.eval()
            snd_enc.eval()
            cur_enc.eval()
            cra.eval()
            dw.eval()
            
            val_loss = 0.0
            val_samples = 0
            
            with torch.no_grad():
                for imgs, mels, cur, labels in val_loader:
                    imgs, mels, cur, labels = imgs.to(device), mels.to(device), cur.to(device), labels.to(device)
                    
                    v_img = img_enc(imgs)
                    v_snd = snd_enc(mels)
                    v_cur = cur_enc(cur)
                    v_concat = torch.cat([v_img, v_snd, v_cur], dim=1)

                    v_prime = cra.forward_impute(v_concat)
                    v_double = cra.backward_impute(v_prime)

                    v_img_p = v_prime[:, :512]
                    v_snd_p = v_prime[:, 512:768]
                    v_cur_p = v_prime[:, 768:]

                    logits_joint, per_logits, omegas = dw([v_img_p, v_snd_p, v_cur_p], labels=labels)

                    L_forward = ((v_prime - v_concat)**2).mean()
                    L_backward = ((v_double - v_concat)**2).mean()
                    per_losses = [criterion(l, labels)*omegas[:,i].mean() for i,l in enumerate(per_logits)]
                    L_enc = sum(per_losses)
                    L_ra = criterion(logits_joint, labels)
                    L_DWFuse = L_enc + ALPHA * L_ra
                    L_TLA = 0.0

                    loss = L_forward + LAMBDA1*L_backward + LAMBDA2*L_DWFuse + LAMBDA3*L_TLA
                    val_loss += loss.item()
                    val_samples += len(imgs)
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            logger.info(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        else:
            avg_train_loss = train_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}")

    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
