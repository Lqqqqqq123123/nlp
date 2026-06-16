# process.py

import pandas as pd
from sklearn.model_selection import train_test_split
from tokenizer import ChineseTokenizer, EnglishTokenizer
import config

def process():
    """
    数据预处理主函数。
    负责读取原始数据、清洗、划分数据集、构建词表，并将文本转换为索引序列后保存。
    """
    print('开始处理数据')

    # 1. 读取原始数据
    # 假设原始文件是 Tab 分隔的文本文件（如常见的 cmn.txt 中英对齐语料）
    df = pd.read_csv(
        config.RAW_DATA_DIR / 'cmn.txt',   # 原始数据路径
        sep='\t',                          # 使用 Tab 键作为分隔符
        header=None,                       # 原始文件没有表头
        usecols=[0, 1],                    # 只读取前两列（英文和中文）
        names=['en', 'zh']                 # 为这两列命名
    )

    # 2. 数据清洗
    df = df.dropna()  # 丢弃包含空值（NaN）的行
    # 去除首尾空格后，过滤掉英文或中文为空白字符串的无效数据
    df = df[df['en'].str.strip().ne('') & df['zh'].str.strip().ne('')]

    # 3. 划分训练集和测试集
    # 将 20% 的数据划分为测试集，80% 为训练集
    # random_state=42 保证每次运行代码时划分的结果一致，方便复现
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 4. 构建词表（仅使用训练集构建，防止测试集信息泄露）
    EnglishTokenizer.build_vocab(train_df['en'].tolist(), config.PROCESS_DATA_DIR / 'en_vocab.txt')
    ChineseTokenizer.build_vocab(train_df['zh'].tolist(), config.PROCESS_DATA_DIR / 'zh_vocab.txt')

    # 5. 加载刚刚构建好的词表，实例化分词器对象
    en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESS_DATA_DIR / 'en_vocab.txt')
    zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESS_DATA_DIR / 'zh_vocab.txt')

    # 6. 将训练集文本转换为索引序列
    # 英文作为源语言（Source），通常需要添加 <sos> 和 <eos> 标记
    train_df['en'] = train_df['en'].apply(
        lambda x: en_tokenizer.encode(x, seq_len=config.MAX_LENGTH, add_sos_eos=True)
    )
    # 中文作为目标语言（Target），在计算损失时通常不需要 <sos> 和 <eos>
    train_df['zh'] = train_df['zh'].apply(
        lambda x: zh_tokenizer.encode(x, seq_len=config.MAX_LENGTH, add_sos_eos=False)
    )
    # 将处理好的训练集保存为 JSONL 格式（每行一个 JSON 对象，适合大文件读取）
    train_df.to_json(config.PROCESS_DATA_DIR / 'indexed_train.jsonl', orient='records', lines=True)

    # 7. 将测试集文本转换为索引序列（逻辑同训练集）
    test_df['en'] = test_df['en'].apply(
        lambda x: en_tokenizer.encode(x, seq_len=config.MAX_LENGTH, add_sos_eos=True)
    )
    test_df['zh'] = test_df['zh'].apply(
        lambda x: zh_tokenizer.encode(x, seq_len=config.MAX_LENGTH, add_sos_eos=False)
    )
    # 保存处理好的测试集
    test_df.to_json(config.PROCESS_DATA_DIR / 'indexed_test.jsonl', orient='records', lines=True)

    print('数据处理完成')

if __name__ == '__main__':
    # 只有当直接运行此脚本时才执行 process()，被其他文件 import 时不会自动执行
    process()