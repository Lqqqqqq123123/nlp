from pathlib import Path
import torch
# Root_Dir
ROOT_DIR = Path(__file__).parents[1]

# DATA_DIR
DATA_DIR = ROOT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESS_DATA_DIR = DATA_DIR / 'process'

# LOG_DIR
LOGS_DIR = ROOT_DIR / 'logs'

# MODEL_DIR
MODEL_DIR = ROOT_DIR / 'models'

# 文件名
RAW_DATA_FILE = 'online_shopping_10_cats.csv'

# 模型
BASE_MODEL = 'google-bert/bert-base-chinese'
BEST_MODEL = 'best_model.pt'


# 训练超参数
EPOCHS = 2
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
MAX_LENGTH = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

