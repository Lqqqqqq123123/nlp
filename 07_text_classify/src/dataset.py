import torch
from torch.utils.data import DataLoader
from config import *
from datasets import load_from_disk

def get_dataloader(train=True):
    """
    构建数据加载器（DataLoader）。
    :param train: 布尔值，是否加载训练集（True 为训练集，False 为测试集）。

    :return: 返回数据加载器。
    """
    # 加载数据集
    data_path = PROCESS_DATA_DIR / ('train' if train else 'test')


    ds = load_from_disk(data_path)
    ds.set_format('torch', ['input_ids', 'attention_mask', 'labels'])

    print(ds.info)

    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=train)


if __name__ == '__main__':
    train_loader = get_dataloader(train=True)


