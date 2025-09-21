# utils.py
import os
import numpy as np
from config import IMAGE_ROOT, CLASSES

def build_file_list(image_root=IMAGE_ROOT, classes=CLASSES):
    """
    遍历 image_root 下所有类别文件夹，生成 (cls, basename) 列表
    """
    file_list = []
    for cls in classes:
        cls_dir = os.path.join(image_root, cls)
        for fname in os.listdir(cls_dir):
            if fname.endswith(".jpg"):
                base = os.path.splitext(fname)[0]
                file_list.append((cls, base))
    return file_list

def detect_current_len(current_root, classes, sample_num=50):
    """
    自动检测电流信号的平均长度
    Args:
        current_root: 电流数据路径
        classes: 类别列表
        sample_num: 每类采样数（避免加载全量）
    Returns:
        int, 平均长度 (四舍五入)
    """
    lengths = []
    for cls in classes:
        cls_dir = os.path.join(current_root, cls)
        for i, fname in enumerate(os.listdir(cls_dir)):
            if fname.endswith(".npy"):
                arr = np.load(os.path.join(cls_dir, fname))
                lengths.append(len(arr))
                if len(lengths) >= sample_num:
                    break
    return int(round(sum(lengths) / len(lengths)))
