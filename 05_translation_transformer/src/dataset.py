# dataset.py

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import config


class TranslationDataset(Dataset):
    """
    自定义翻译数据集。
    继承自 PyTorch 的 Dataset 基类，负责将已编码的中英文索引序列加载到内存，
    并提供标准的 __len__ 和 __getitem__ 接口供 DataLoader 调用。
    """

    def __init__(self, data_path):
        """
        初始化数据集。
        :param data_path: 数据文件路径（JSONL 格式，由 process.py 生成）。
        """
        # 读取 JSONL 文件，并转换为字典列表（List of Dicts）
        # 例如: [{'en': [1, 2, 3], 'zh': [4, 5]}, {'en': [6, 7], 'zh': [8, 9]}]
        self.data = pd.read_json(data_path, lines=True).to_dict(orient='records')

    def __len__(self):
        """
        返回数据集的总样本数。
        这是 Dataset 必须实现的魔法方法，供 DataLoader 获取数据集大小。
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        根据索引获取指定样本。
        这是 Dataset 必须实现的魔法方法，DataLoader 在迭代时会调用此方法获取单条数据。
        :param index: 样本的整数索引。
        :return: 包含输入张量和目标张量的元组 (input_tensor, target_tensor)。
        """
        # 获取中文索引序列，并转换为 PyTorch 的长整型张量（LongTensor）
        # 注意：这里将中文作为模型的输入（Source）
        input_tensor = torch.tensor(self.data[index]['zh'], dtype=torch.long)

        # 获取英文索引序列，同样转换为长整型张量
        # 英文作为模型预测的目标（Target）
        target_tensor = torch.tensor(self.data[index]['en'], dtype=torch.long)

        return input_tensor, target_tensor


def get_dataloader(train=True):
    """
    构建数据加载器（DataLoader）。
    :param train: 布尔值，是否加载训练集（True 为训练集，False 为测试集）。
    :return: 配置好批大小和洗牌策略的 DataLoader 实例。
    """
    # 根据 train 参数动态选择加载训练集或测试集的 JSONL 文件路径
    data_path = config.PROCESS_DATA_DIR / ('indexed_train.jsonl' if train else 'indexed_test.jsonl')

    # 实例化自定义数据集
    dataset = TranslationDataset(data_path)

    # 构建 DataLoader
    # batch_size: 每次迭代提取的样本数量（从 config 中读取）
    # shuffle=True: 每个 epoch 开始时打乱数据顺序，防止模型记住数据顺序，提升泛化能力
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)


if __name__ == '__main__':
    # 获取训练集的 DataLoader
    train_loader = get_dataloader(train=True)

    # 测试 DataLoader 是否能正常迭代
    for inputs, targets in train_loader:
        # 打印当前批次（Batch）的输入和目标张量的形状
        # 预期输出类似: torch.Size([32, 50]) torch.Size([32, 50])
        # 32 为 BATCH_SIZE，50 为 SEQ_LEN
        print(inputs.shape, targets.shape)
        break  # 仅测试第一个批次后退出循环