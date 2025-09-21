# dataset_multimodal.py (更新版)

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import librosa
from config import CLASSES

class MultiModalWeldDataset(Dataset):
    def __init__(self, image_root, sound_root, current_root, file_list,
                 transform_img=None, sr=22050, n_mels=128, sound_duration=2.0,
                 current_len=3333):
        self.image_root = image_root
        self.sound_root = sound_root
        self.current_root = current_root
        self.file_list = file_list
        self.transform_img = transform_img
        self.sr = sr
        self.n_mels = n_mels
        self.sound_duration = sound_duration
        self.current_len = current_len  # 固定长度

    def __len__(self):
        return len(self.file_list)

    def load_mel(self, wav_path):
        y, _ = librosa.load(wav_path, sr=self.sr, duration=self.sound_duration)
        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
        return mel_db.astype(np.float32)

    def _pad_or_truncate(self, arr):
        """把 1D 电流信号 pad/截断 到固定长度"""
        if len(arr) > self.current_len:
            return arr[:self.current_len]
        elif len(arr) < self.current_len:
            pad_width = self.current_len - len(arr)
            return np.pad(arr, (0, pad_width), mode="constant")
        return arr

    def __getitem__(self, idx):
        cls, base = self.file_list[idx]
        label = CLASSES.index(cls)

        # 图像
        img_path = os.path.join(self.image_root, cls, base + ".jpg")
        img = Image.open(img_path).convert("RGB")
        if self.transform_img:
            img = self.transform_img(img)

        # 声音
        wav_path = os.path.join(self.sound_root, cls, base + ".wav")
        mel = self.load_mel(wav_path)
        mel = torch.from_numpy(mel).unsqueeze(0)  # [1, n_mels, T]

        # 电流
        cur_path = os.path.join(self.current_root, cls, base + ".npy")
        cur = np.load(cur_path).astype(np.float32)
        cur = self._pad_or_truncate(cur)         # 保证长度一致
        cur = torch.from_numpy(cur)

        return img, mel, cur, label
