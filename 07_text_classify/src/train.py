import torch
import torch.nn as nn
from accelerate.utils import tqdm
from torch.nn import BCEWithLogitsLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from model import MyModel
from dataset import get_dataloader
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, ROOT_DIR
import matplotlib.pyplot as plt
def train_one_epoch(model, dataloader:DataLoader, loss_fn:BCEWithLogitsLoss, optimizer:Optimizer, device) -> float:
    """
    训练模型，外层循环 epochs，每个 epoch 完成的遍历一遍 dataloader
    :return:
    """
    # 1. 环境初始化
    model.train()
    total_loss = 0
    idx = 0
    # 2. 遍历
    train_loader = get_dataloader(train=True)
    for batch in tqdm(train_loader, desc='训练'):

        input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

        # 2.1 前向传播
        outputs = model(**inputs)
        # 2.2 计算损失
        # outputs [batch_size, ], labels [batch_size, ]
        loss = loss_fn(outputs,  labels.float())
        total_loss += loss.item()

        # 2.3 反向传播
        loss.backward()

        # 2.4 更新参数
        optimizer.step()
        optimizer.zero_grad()


    return total_loss / len(train_loader)


def train():
    # 1. 初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = get_dataloader(train=True)
    model = MyModel().to(device)
    loss_fn = BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_history = []

    # 2. 训练
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        print(f'========== Epoch {epoch} ==========')
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        loss_history.append(train_loss)

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), ROOT_DIR / 'models' / 'best_model.pt')


    # 3. 画图
    plt.plot(loss_history)
    plt.show()



if __name__ == '__main__':
    train()



