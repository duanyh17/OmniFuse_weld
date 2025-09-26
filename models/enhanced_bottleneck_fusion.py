# enhanced_bottleneck_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for current signals"""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        # x: [batch_size, seq_len] 
        # Reshape to [batch_size, seq_len, 1] for attention
        x_reshaped = x.unsqueeze(-1)
        
        # Compute attention weights
        attention_weights = self.attention(x_reshaped)  # [batch_size, seq_len, 1]
        attention_weights = F.softmax(attention_weights.squeeze(-1), dim=1)  # [batch_size, seq_len]
        
        # Apply attention
        weighted_x = x * attention_weights
        return weighted_x, attention_weights


class EnhancedCurrentEncoder(nn.Module):
    """Enhanced current encoder with deeper processing and temporal attention"""
    def __init__(self, in_len=100, out_dim=128):
        super().__init__()
        self.in_len = in_len
        self.out_dim = out_dim
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(1, hidden_dim=32)
        
        # 1D Convolutional layers for pattern detection
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)  # Fixed output length
        )
        
        # Frequency domain features
        self.freq_fc = nn.Linear(in_len // 2, 32)  # For FFT features
        
        # Calculate the combined feature dimension
        self.combined_dim = 64 * 8 + 32  # conv features + freq features
        
        # Input projection to standardize dimension
        self.input_proj = nn.Linear(self.combined_dim, 256)
        
        # Residual connections
        self.residual_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 256)
            ) for _ in range(2)
        ])
        
        # Final projection
        self.final_proj = nn.Linear(256, out_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Apply temporal attention
        x_attended, attention_weights = self.temporal_attention(x)
        
        # Convolutional features
        x_conv = x_attended.unsqueeze(1)  # Add channel dimension
        conv_features = self.conv_layers(x_conv)
        conv_features = conv_features.view(batch_size, -1)
        
        # Frequency domain features (simple FFT)
        x_freq = torch.fft.rfft(x_attended, dim=1)
        x_freq_mag = torch.abs(x_freq)[:, :self.in_len//2]  # Take magnitude
        freq_features = self.freq_fc(x_freq_mag)
        
        # Combine features
        combined = torch.cat([conv_features, freq_features], dim=1)
        
        # Project to standard dimension
        h = self.input_proj(combined)
        
        # Residual layers
        for layer in self.residual_layers:
            residual = layer(h)
            h = h + residual  # Residual connection
            
        # Final projection
        output = self.final_proj(h)
        
        return output


class CurrentAwareFusion(nn.Module):
    """Enhanced fusion with current modality boosting"""
    def __init__(self, modal_dims, num_classes, beta=0.5, current_boost=2.0):
        super().__init__()
        self.M = len(modal_dims)
        self.current_boost = current_boost
        
        # Individual modality classifiers
        self.g_mod = nn.ModuleList([
            self._build_modality_classifier(d, num_classes, i) 
            for i, d in enumerate(modal_dims)
        ])
        
        self.beta = beta
        total_dim = sum(modal_dims)
        
        # Enhanced joint classifier with current emphasis
        self.joint_clf = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        # Current-specific attention
        self.current_attention = nn.Sequential(
            nn.Linear(modal_dims[2], 64),  # Assuming current is 3rd modality
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def _build_modality_classifier(self, in_dim, num_classes, modality_idx):
        """Build classifier with enhanced capacity for current modality"""
        if modality_idx == 2:  # Current modality (assuming it's 3rd)
            return nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )
        else:
            return nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )
    
    def forward(self, vs, labels=None):
        # Individual modality predictions
        per_logits = [g(v) for g, v in zip(self.g_mod, vs)]
        per_probs = [F.softmax(l, dim=1) for l in per_logits]
        
        # Get confidence scores
        if labels is not None:
            p_true = [p.gather(1, labels.view(-1,1)).squeeze(1) for p in per_probs]
        else:
            p_true = [p.max(dim=1).values for p in per_probs]

        # Enhanced omega calculation with current boosting
        p_stack = torch.stack(p_true, dim=1)  # [B, M]
        omegas = []
        
        for m in range(self.M):
            others = 1.0 - torch.cat([p_stack[:, :m], p_stack[:, m+1:]], dim=1)
            prod = torch.prod(others, dim=1)
            omega = prod ** (self.beta / (self.M - 1))
            
            # Boost current modality (assuming it's index 2)
            if m == 2:
                current_attention = self.current_attention(vs[2])
                omega = omega * (1 + self.current_boost * current_attention.squeeze())
                
            omegas.append(omega)
            
        omegas = torch.stack(omegas, dim=1)
        
        # Normalize omegas
        omegas = omegas / (omegas.sum(dim=1, keepdim=True) + 1e-8)
        
        # Joint prediction
        joint = torch.cat(vs, dim=1)
        logits_joint = self.joint_clf(joint)
        
        return logits_joint, per_logits, omegas


class EnhancedBottleneckFusion(nn.Module):
    """Complete enhanced model with current boosting"""
    def __init__(self, image_dim=512, sound_dim=256, current_len=100, current_dim=128, 
                 num_classes=6, cra_k=5, beta=0.5, current_boost=2.0):
        super().__init__()
        
        # Enhanced current encoder
        self.current_encoder = EnhancedCurrentEncoder(current_len, current_dim)
        
        # Keep original encoders for image and sound (can be passed as parameters)
        self.modal_dims = [image_dim, sound_dim, current_dim]
        
        # Enhanced CRA (from original)
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'OmniFuse'))
        from cra import CRA
        self.cra = CRA(dim=sum(self.modal_dims), K=cra_k)
        
        # Current-aware fusion
        self.fusion = CurrentAwareFusion(
            self.modal_dims, num_classes, beta=beta, current_boost=current_boost
        )
        
    def forward(self, image_features, sound_features, current_signal, labels=None):
        # Enhanced current encoding
        current_features = self.current_encoder(current_signal)
        
        # Concatenate all modalities
        v_concat = torch.cat([image_features, sound_features, current_features], dim=1)
        
        # CRA processing
        v_prime = self.cra.forward_impute(v_concat)
        v_double = self.cra.backward_impute(v_prime)
        
        # Split enhanced features
        v_img_p = v_prime[:, :self.modal_dims[0]]
        v_snd_p = v_prime[:, self.modal_dims[0]:self.modal_dims[0]+self.modal_dims[1]]
        v_cur_p = v_prime[:, self.modal_dims[0]+self.modal_dims[1]:]
        
        # Current-aware fusion
        logits_joint, per_logits, omegas = self.fusion([v_img_p, v_snd_p, v_cur_p], labels)
        
        return {
            'logits_joint': logits_joint,
            'per_logits': per_logits,
            'omegas': omegas,
            'v_prime': v_prime,
            'v_double': v_double,
            'v_concat': v_concat
        }