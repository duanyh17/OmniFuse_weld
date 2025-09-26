# dataset_multimodal.py (更新版)

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import librosa
import logging
from config import CLASSES

logger = logging.getLogger(__name__)

class MultiModalWeldDataset(Dataset):
    def __init__(self, image_root, sound_root, current_root, file_list,
                 transform_img=None, sr=22050, n_mels=128, sound_duration=2.0,
                 current_len=3333, enable_augmentation=False, augmentation_prob=0.3,
                 classes=None):
        """
        Enhanced multimodal welding dataset with proper data augmentation control.
        
        Args:
            image_root: Path to image directory
            sound_root: Path to sound directory  
            current_root: Path to current data directory
            file_list: List of (class, basename) tuples
            transform_img: Image transformations (optional)
            sr: Sample rate for audio
            n_mels: Number of mel frequency bins
            sound_duration: Duration of sound clips in seconds
            current_len: Fixed length for current sequences
            enable_augmentation: Whether to enable data augmentation
            augmentation_prob: Probability of applying augmentation
            classes: List of class names (if None, uses default CLASSES)
        """
        self.image_root = image_root
        self.sound_root = sound_root
        self.current_root = current_root
        self.file_list = file_list
        self.transform_img = transform_img
        self.sr = sr
        self.n_mels = n_mels
        self.sound_duration = sound_duration
        self.current_len = current_len  # 固定长度
        self.enable_augmentation = enable_augmentation
        self.augmentation_prob = augmentation_prob
        
        # Handle class mapping
        if classes is not None:
            self.classes = classes
        else:
            self.classes = CLASSES
            
        # Create class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        logger.info(f"Dataset initialized with {len(file_list)} samples, "
                   f"current_len={current_len}, augmentation={'enabled' if enable_augmentation else 'disabled'}")
        logger.info(f"Classes: {self.classes}")

    def __len__(self):
        return len(self.file_list)

    def load_mel(self, wav_path):
        """Load and process mel spectrogram from audio file."""
        try:
            y, _ = librosa.load(wav_path, sr=self.sr, duration=self.sound_duration)
            mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
            return mel_db.astype(np.float32)
        except Exception as e:
            logger.error(f"Error loading audio file {wav_path}: {e}")
            # Return zero spectrogram as fallback
            return np.zeros((self.n_mels, int(self.sr * self.sound_duration / 512)), dtype=np.float32)

    def _pad_or_truncate(self, arr):
        """把 1D 电流信号 pad/截断 到固定长度"""
        if len(arr) > self.current_len:
            return arr[:self.current_len]
        elif len(arr) < self.current_len:
            pad_width = self.current_len - len(arr)
            return np.pad(arr, (0, pad_width), mode="constant")
        return arr
    
    def _apply_safe_augmentation(self, cur_data):
        """
        Apply safe data augmentation that preserves multimodal correspondence.
        Uses additive noise instead of scaling to maintain relative relationships.
        """
        if not self.enable_augmentation or torch.rand(1).item() > self.augmentation_prob:
            return cur_data
        
        # Add small amount of gaussian noise instead of scaling
        noise_std = 0.02 * cur_data.std()  # 2% of signal standard deviation
        noise = torch.randn_like(cur_data) * noise_std
        augmented = cur_data + noise
        
        return augmented

    def __getitem__(self, idx):
        """Get a single multimodal sample with proper error handling and augmentation."""
        try:
            cls, base = self.file_list[idx]
            # Use class mapping if available, otherwise use default CLASSES
            if cls in self.class_to_idx:
                label = self.class_to_idx[cls]
            else:
                logger.warning(f"Class {cls} not found in class mapping, using index 0")
                label = 0

            # 图像
            img_path = os.path.join(self.image_root, cls, base + ".jpg")
            if not os.path.exists(img_path):
                logger.warning(f"Image file not found: {img_path}")
                # Create a dummy image as fallback
                img = Image.new('RGB', (224, 224), color='black')
            else:
                img = Image.open(img_path).convert("RGB")
            
            if self.transform_img:
                img = self.transform_img(img)

            # 声音
            wav_path = os.path.join(self.sound_root, cls, base + ".wav")
            mel = self.load_mel(wav_path)
            mel = torch.from_numpy(mel).unsqueeze(0)  # [1, n_mels, T]

            # 电流 - with proper error handling and validation
            cur_path = os.path.join(self.current_root, cls, base + ".npy")
            if not os.path.exists(cur_path):
                logger.warning(f"Current file not found: {cur_path}")
                # Create dummy current data as fallback
                cur = np.zeros(self.current_len, dtype=np.float32)
            else:
                cur = np.load(cur_path).astype(np.float32)
                
            # Ensure consistent length
            cur = self._pad_or_truncate(cur)
            cur = torch.from_numpy(cur)
            
            # Apply safe augmentation if enabled
            cur = self._apply_safe_augmentation(cur)
            
            # Validate data shapes
            if len(cur) != self.current_len:
                logger.error(f"Current data length mismatch: expected {self.current_len}, got {len(cur)}")
                # Fix the length issue
                cur = torch.zeros(self.current_len, dtype=torch.float32)

            return img, mel, cur, label
            
        except Exception as e:
            logger.error(f"Error loading sample {idx} ({cls}/{base}): {e}")
            # Return dummy data to prevent training interruption
            dummy_img = torch.zeros(3, 224, 224) if self.transform_img else Image.new('RGB', (224, 224))
            dummy_mel = torch.zeros(1, self.n_mels, int(self.sr * self.sound_duration / 512))
            dummy_cur = torch.zeros(self.current_len)
            dummy_label = 0
            return dummy_img, dummy_mel, dummy_cur, dummy_label
