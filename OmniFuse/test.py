# test_dataset.py
import torch
from dataset_multimodal import MultiModalWeldDataset
from utils import build_file_list, detect_current_len
from config import IMAGE_ROOT, SOUND_ROOT, CURRENT_ROOT, CLASSES

if __name__ == "__main__":
    # 自动检测电流信号长度
    cur_len = detect_current_len(CURRENT_ROOT, CLASSES, sample_num=50)
    print("检测到的电流信号长度:", cur_len)

    # 构建样本列表
    file_list = build_file_list()
    print("样本数:", len(file_list))
    print("前 5 个样本:", file_list[:5])

    # 构建 Dataset
    dataset = MultiModalWeldDataset(
        IMAGE_ROOT, SOUND_ROOT, CURRENT_ROOT,
        file_list=file_list,
        current_len=cur_len
    )

    # 随机取一个样本
    img, mel, cur, label = dataset[0]
    print("图像 shape:", img.shape)   # [3, 224, 224] (如果你加了 transform)
    print("声音 shape:", mel.shape)  # [1, n_mels, T]
    print("电流 shape:", cur.shape)  # [3333]
    print("类别 id:", label, "类别名:", CLASSES[label])
