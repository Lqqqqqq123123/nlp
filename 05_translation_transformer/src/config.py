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
MODEL_DIR = ROOT_DIR / 'model'

# 文件名
RAW_DATA_FILE = 'cmn.txt'
TRAIN_DATA_FILE = 'train.txt'
TEST_DATA_FILE = 'test.txt'


# 模型
BASE_MODEL = 'best_model.pt'

# 词表
ZH_VOCAB_FILE = 'zh_vocab.txt'
EN_VOCAB_FILE = 'en_vocab.txt'

# 特殊TOKEN
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'

# 模型参数
D_MODEL = 128
NHEAD = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2


# 训练超参数
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.1
MAX_LENGTH = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

