# config.py
# 存放路径、类别、超参数配置

DATASET_ROOT = r"E:\11_weld_data\dataset"

# 三模态路径
IMAGE_ROOT   = DATASET_ROOT + r"\image"
SOUND_ROOT   = DATASET_ROOT + r"\sound"
CURRENT_ROOT = DATASET_ROOT + r"\current"

# 六个类别
CLASSES = [
    "burn_through",
    "lack_of_penetration",
    "misalignment",
    "normal",
    "over_penetration",
    "stomata"
]

# 超参数 (参考 OmniFuse 论文 Table 7)
NUM_CLASSES = len(CLASSES)
K = 5          # CRA residual adapters
ALPHA = 0.1    # DWFuse 辅助项系数
BETA = 0.5     # DWFuse 权重参数
LAMBDA1 = 1.0  # backward loss
LAMBDA2 = 10.0 # DWFuse loss
LAMBDA3 = 0.5  # TLA loss

# 训练参数
BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 30
