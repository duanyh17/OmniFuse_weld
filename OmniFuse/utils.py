# utils.py
import os
import numpy as np
import logging
from config import IMAGE_ROOT, CLASSES

logger = logging.getLogger(__name__)

def build_file_list(image_root=IMAGE_ROOT, classes=CLASSES):
    """
    遍历 image_root 下所有类别文件夹，生成 (cls, basename) 列表
    """
    file_list = []
    for cls in classes:
        cls_dir = os.path.join(image_root, cls)
        if not os.path.exists(cls_dir):
            logger.warning(f"Class directory not found: {cls_dir}")
            continue
            
        for fname in os.listdir(cls_dir):
            if fname.endswith(".jpg"):
                base = os.path.splitext(fname)[0]
                file_list.append((cls, base))
    return file_list

def detect_current_len(current_root, classes, sample_num=100):
    """
    自动检测电流信号的最佳长度，考虑全局数据分布
    Args:
        current_root: 电流数据路径
        classes: 类别列表
        sample_num: 每类采样数（避免加载全量，但增加采样数以提高准确性）
    Returns:
        int, 推荐长度 (基于统计分析)
    """
    lengths = []
    total_files = 0
    
    for cls in classes:
        cls_dir = os.path.join(current_root, cls)
        if not os.path.exists(cls_dir):
            logger.warning(f"Current data directory not found: {cls_dir}")
            continue
            
        class_lengths = []
        for i, fname in enumerate(os.listdir(cls_dir)):
            if fname.endswith(".npy"):
                try:
                    arr = np.load(os.path.join(cls_dir, fname))
                    class_lengths.append(len(arr))
                    total_files += 1
                    if len(class_lengths) >= sample_num:
                        break
                except Exception as e:
                    logger.warning(f"Error loading {fname}: {e}")
                    continue
        
        lengths.extend(class_lengths)
        logger.info(f"Class {cls}: {len(class_lengths)} files, avg length: {np.mean(class_lengths):.1f}")
    
    if not lengths:
        logger.error("No valid current data files found!")
        return 3333  # Default fallback value
    
    lengths = np.array(lengths)
    
    # Statistical analysis for optimal length selection
    mean_len = np.mean(lengths)
    median_len = np.median(lengths)
    std_len = np.std(lengths)
    percentile_95 = np.percentile(lengths, 95)
    
    logger.info(f"Current length statistics:")
    logger.info(f"  Total files analyzed: {total_files}")
    logger.info(f"  Mean: {mean_len:.1f}")
    logger.info(f"  Median: {median_len:.1f}") 
    logger.info(f"  Std: {std_len:.1f}")
    logger.info(f"  95th percentile: {percentile_95:.1f}")
    logger.info(f"  Min: {np.min(lengths)}, Max: {np.max(lengths)}")
    
    # Choose length that captures 95% of data while being computationally reasonable
    # Use 95th percentile but cap at reasonable maximum to avoid outliers
    recommended_len = min(int(percentile_95), int(mean_len + 2 * std_len))
    
    logger.info(f"Recommended current sequence length: {recommended_len}")
    return recommended_len

def validate_data_consistency(image_root, sound_root, current_root, file_list):
    """
    验证多模态数据的一致性
    Args:
        image_root, sound_root, current_root: 数据路径
        file_list: 文件列表
    Returns:
        dict: 包含验证结果的字典
    """
    results = {
        'total_samples': len(file_list),
        'missing_images': 0,
        'missing_sounds': 0, 
        'missing_currents': 0,
        'valid_samples': 0
    }
    
    for cls, base in file_list:
        img_path = os.path.join(image_root, cls, base + ".jpg")
        wav_path = os.path.join(sound_root, cls, base + ".wav")
        cur_path = os.path.join(current_root, cls, base + ".npy")
        
        if not os.path.exists(img_path):
            results['missing_images'] += 1
        if not os.path.exists(wav_path):
            results['missing_sounds'] += 1
        if not os.path.exists(cur_path):
            results['missing_currents'] += 1
            
        if all(os.path.exists(p) for p in [img_path, wav_path, cur_path]):
            results['valid_samples'] += 1
    
    logger.info(f"Data consistency validation:")
    logger.info(f"  Total samples: {results['total_samples']}")
    logger.info(f"  Valid samples: {results['valid_samples']}")
    logger.info(f"  Missing images: {results['missing_images']}")
    logger.info(f"  Missing sounds: {results['missing_sounds']}")
    logger.info(f"  Missing currents: {results['missing_currents']}")
    
    return results
