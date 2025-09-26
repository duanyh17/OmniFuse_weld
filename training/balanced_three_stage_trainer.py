# balanced_three_stage_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import os
import logging
from datetime import datetime


class CurrentAwareLoss(nn.Module):
    """Current-aware loss function that emphasizes current modality"""
    def __init__(self, num_classes, current_weight=3.0, pattern_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.current_weight = current_weight
        self.pattern_weight = pattern_weight
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, joint_logits, per_logits, omegas, labels):
        # Standard classification loss
        joint_loss = self.ce_loss(joint_logits, labels)
        
        # Individual modality losses with current emphasis
        modality_losses = []
        for i, logits in enumerate(per_logits):
            loss = self.ce_loss(logits, labels)
            # Boost current modality loss (assuming index 2)
            if i == 2:
                loss = loss * self.current_weight
            modality_losses.append(loss * omegas[:, i].mean())
            
        modality_loss = sum(modality_losses)
        
        # Current pattern consistency loss
        current_probs = F.softmax(per_logits[2], dim=1)
        joint_probs = F.softmax(joint_logits, dim=1)
        pattern_loss = F.kl_div(
            F.log_softmax(per_logits[2], dim=1), 
            joint_probs, 
            reduction='batchmean'
        ) * self.pattern_weight
        
        return joint_loss + modality_loss + pattern_loss


class BalancedThreeStageTrainer:
    """Enhanced three-stage trainer with current modality emphasis"""
    
    def __init__(self, model, train_loader, val_loader, config, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Setup logging
        self.setup_logging()
        
        # Loss functions
        self.criterion = nn.CrossEntropyLoss()
        self.current_aware_loss = CurrentAwareLoss(
            config.NUM_CLASSES, 
            current_weight=config.LAMBDA_CURRENT
        )
        
        # Optimizers for different stages
        self.setup_optimizers()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'current_weight': []
        }
        
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_filename = os.path.join(
            self.config.LOG_DIR, 
            f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_optimizers(self):
        """Setup different optimizers for different training stages"""
        # Standard parameters
        standard_params = []
        # Current-specific parameters with higher learning rate
        current_params = []
        
        for name, param in self.model.named_parameters():
            if 'current' in name.lower():
                current_params.append(param)
            else:
                standard_params.append(param)
                
        # Stage 1: Individual modality training
        self.optimizer_stage1 = optim.Adam([
            {'params': standard_params, 'lr': self.config.LR},
            {'params': current_params, 'lr': self.config.LR * self.config.CURRENT_LR_MULTIPLIER}
        ], weight_decay=self.config.WEIGHT_DECAY)
        
        # Stage 2: Fusion training
        self.optimizer_stage2 = optim.Adam([
            {'params': standard_params, 'lr': self.config.LR * 0.8},
            {'params': current_params, 'lr': self.config.LR * self.config.CURRENT_LR_MULTIPLIER * 0.8}
        ], weight_decay=self.config.WEIGHT_DECAY)
        
        # Stage 3: Fine-tuning
        self.optimizer_stage3 = optim.Adam([
            {'params': standard_params, 'lr': self.config.LR * 0.5},
            {'params': current_params, 'lr': self.config.LR * self.config.CURRENT_LR_MULTIPLIER * 0.5}
        ], weight_decay=self.config.WEIGHT_DECAY)
        
        # Learning rate schedulers
        self.scheduler_stage1 = optim.lr_scheduler.StepLR(self.optimizer_stage1, step_size=5, gamma=0.8)
        self.scheduler_stage2 = optim.lr_scheduler.StepLR(self.optimizer_stage2, step_size=7, gamma=0.8)
        self.scheduler_stage3 = optim.lr_scheduler.StepLR(self.optimizer_stage3, step_size=10, gamma=0.9)
    
    def train_epoch(self, optimizer, stage='stage2'):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        for batch_idx, (images, sounds, currents, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            sounds = sounds.to(self.device) 
            currents = currents.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Get image and sound features (assuming we have encoders)
            # For now, we'll use placeholder - in real implementation, these would come from encoders
            batch_size = images.size(0)
            image_features = torch.randn(batch_size, self.config.IMAGE_DIM).to(self.device)
            sound_features = torch.randn(batch_size, self.config.SOUND_DIM).to(self.device)
            
            # Forward pass
            outputs = self.model(image_features, sound_features, currents, labels)
            
            # Calculate losses based on stage
            if stage == 'stage1':
                # Focus on individual modality training
                loss = self.calculate_stage1_loss(outputs, labels)
            elif stage == 'stage2':
                # Focus on fusion with current emphasis  
                loss = self.calculate_stage2_loss(outputs, labels)
            else:  # stage3
                # Balanced fine-tuning
                loss = self.calculate_stage3_loss(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_samples += batch_size
            
            if batch_idx % self.config.LOG_INTERVAL == 0:
                self.logger.info(
                    f'Batch [{batch_idx}/{len(self.train_loader)}] '
                    f'Loss: {loss.item():.4f}'
                )
        
        return total_loss / len(self.train_loader)
    
    def calculate_stage1_loss(self, outputs, labels):
        """Stage 1: Individual modality training with current emphasis"""
        joint_logits = outputs['logits_joint']
        per_logits = outputs['per_logits']
        omegas = outputs['omegas']
        
        # Standard losses
        L_joint = self.criterion(joint_logits, labels)
        
        # Enhanced current modality loss
        L_current = self.criterion(per_logits[2], labels) * self.config.LAMBDA_CURRENT
        L_image = self.criterion(per_logits[0], labels)
        L_sound = self.criterion(per_logits[1], labels)
        
        return L_joint + L_current + L_image + L_sound
    
    def calculate_stage2_loss(self, outputs, labels):
        """Stage 2: Fusion training with current awareness"""
        joint_logits = outputs['logits_joint']
        per_logits = outputs['per_logits']
        omegas = outputs['omegas']
        v_prime = outputs['v_prime']
        v_double = outputs['v_double']
        v_concat = outputs['v_concat']
        
        # CRA losses
        L_forward = ((v_prime - v_concat) ** 2).mean()
        L_backward = ((v_double - v_concat) ** 2).mean()
        
        # Current-aware fusion loss
        L_fusion = self.current_aware_loss(joint_logits, per_logits, omegas, labels)
        
        # Current pattern regularization
        current_features = v_prime[:, -self.config.CURRENT_DIM:]
        L_current_reg = torch.norm(current_features, p=2) * self.config.CURRENT_L2_REG
        
        total_loss = (L_forward + 
                     self.config.LAMBDA1 * L_backward + 
                     self.config.LAMBDA2 * L_fusion + 
                     L_current_reg)
        
        return total_loss
    
    def calculate_stage3_loss(self, outputs, labels):
        """Stage 3: Balanced fine-tuning"""
        joint_logits = outputs['logits_joint']
        per_logits = outputs['per_logits']
        omegas = outputs['omegas']
        v_prime = outputs['v_prime']
        v_double = outputs['v_double']
        v_concat = outputs['v_concat']
        
        # All loss components with balanced weights
        L_forward = ((v_prime - v_concat) ** 2).mean()
        L_backward = ((v_double - v_concat) ** 2).mean()
        
        # Modality-balanced loss
        L_joint = self.criterion(joint_logits, labels)
        per_losses = [self.criterion(logits, labels) * omegas[:, i].mean() 
                     for i, logits in enumerate(per_logits)]
        L_modalities = sum(per_losses)
        
        # Current consistency loss
        L_consistency = F.mse_loss(
            F.softmax(per_logits[2], dim=1),
            F.softmax(joint_logits, dim=1)
        )
        
        total_loss = (L_forward + 
                     self.config.LAMBDA1 * L_backward + 
                     L_joint + 
                     L_modalities + 
                     0.5 * L_consistency)
        
        return total_loss
    
    def validate(self):
        """Validation loop"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, sounds, currents, labels in self.val_loader:
                images = images.to(self.device)
                sounds = sounds.to(self.device)
                currents = currents.to(self.device)
                labels = labels.to(self.device)
                
                # Placeholder features
                batch_size = images.size(0)
                image_features = torch.randn(batch_size, self.config.IMAGE_DIM).to(self.device)
                sound_features = torch.randn(batch_size, self.config.SOUND_DIM).to(self.device)
                
                outputs = self.model(image_features, sound_features, currents, labels)
                
                loss = self.criterion(outputs['logits_joint'], labels)
                total_loss += loss.item()
                
                _, predicted = outputs['logits_joint'].max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = total_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def save_checkpoint(self, epoch, stage, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'stage': stage,
            'model_state_dict': self.model.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        filename = f"checkpoint_stage{stage}_epoch{epoch}.pth"
        if is_best:
            filename = f"best_model_stage{stage}.pth"
            
        filepath = os.path.join(self.config.CHECKPOINT_DIR, filename)
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved: {filepath}")
    
    def train_three_stage(self):
        """Main three-stage training loop"""
        self.logger.info("Starting enhanced three-stage training with current emphasis")
        
        # Stage 1: Individual modality training
        self.logger.info("=== STAGE 1: Individual Modality Training ===")
        for epoch in range(self.config.STAGE1_EPOCHS):
            train_loss = self.train_epoch(self.optimizer_stage1, 'stage1')
            val_loss, val_acc = self.validate()
            
            self.scheduler_stage1.step()
            
            self.logger.info(
                f"Stage 1 Epoch {epoch+1}/{self.config.STAGE1_EPOCHS}: "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_acc:.2f}%"
            )
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, 1, is_best=True)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        
        # Stage 2: Fusion training with current emphasis
        self.logger.info("=== STAGE 2: Fusion Training with Current Emphasis ===")
        for epoch in range(self.config.STAGE2_EPOCHS):
            train_loss = self.train_epoch(self.optimizer_stage2, 'stage2')
            val_loss, val_acc = self.validate()
            
            self.scheduler_stage2.step()
            
            # Track current modality weight
            with torch.no_grad():
                sample_batch = next(iter(self.val_loader))
                images, sounds, currents, labels = [x.to(self.device) for x in sample_batch]
                batch_size = images.size(0)
                image_features = torch.randn(batch_size, self.config.IMAGE_DIM).to(self.device)
                sound_features = torch.randn(batch_size, self.config.SOUND_DIM).to(self.device)
                outputs = self.model(image_features, sound_features, currents, labels)
                current_weight = outputs['omegas'][:, 2].mean().item()
                self.history['current_weight'].append(current_weight)
            
            self.logger.info(
                f"Stage 2 Epoch {epoch+1}/{self.config.STAGE2_EPOCHS}: "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_acc:.2f}%, Current Weight: {current_weight:.3f}"
            )
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, 2, is_best=True)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        
        # Stage 3: Balanced fine-tuning
        self.logger.info("=== STAGE 3: Balanced Fine-tuning ===")
        for epoch in range(self.config.STAGE3_EPOCHS):
            train_loss = self.train_epoch(self.optimizer_stage3, 'stage3')
            val_loss, val_acc = self.validate()
            
            self.scheduler_stage3.step()
            
            self.logger.info(
                f"Stage 3 Epoch {epoch+1}/{self.config.STAGE3_EPOCHS}: "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_acc:.2f}%"
            )
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, 3, is_best=True)
        
        self.logger.info(f"Training completed! Best validation accuracy: {self.best_val_acc:.2f}%")
        return self.history