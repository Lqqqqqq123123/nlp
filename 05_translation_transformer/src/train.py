# train.py

import time
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloader
from tokenizer import ChineseTokenizer, EnglishTokenizer
import config
from model import TranslationModel


def train_one_epoch(dataloader, model, loss_function, optimizer, device):
    """
    执行一个 Epoch 的训练。
    :param dataloader: 数据加载器。
    :param model: 翻译模型。
    :param loss_function: 损失函数。
    :param optimizer: 优化器。
    :param device: 训练设备（CPU 或 GPU）。
    :return: 当前 Epoch 的平均损失值。
    """
    # 将模型切换为训练模式（这会激活 Dropout 等训练专属行为）
    model.train()
    total_loss = 0

    # 使用 tqdm 显示训练进度条
    for src, tgt in tqdm(dataloader, desc='训练'):
        # 将输入数据和目标数据移动到指定设备（如 GPU）
        src = src.to(device)
        tgt = tgt.to(device)

        # 1. 生成填充掩码（Padding Mask）
        # 找出源语言和目标语言中 <pad> 的位置，生成布尔掩码，告诉模型忽略这些填充位置
        src_pad_mask = (src == model.src_embedding.padding_idx)
        tgt_pad_mask = (tgt == model.tgt_embedding.padding_idx)

        # 2. 目标序列的移位操作（Teacher Forcing 机制）
        # 在训练时，解码器的输入是目标序列去掉最后一个词（tgt_input）
        tgt_input = tgt[:, :-1]
        # 解码器需要预测的目标是目标序列去掉第一个词（tgt_output）
        tgt_output = tgt[:, 1:]

        # 3. 生成因果掩码（Causal Mask）
        # 防止解码器在预测当前词时“偷看”到未来的词（自回归特性）
        tgt_mask = model.transformer.generate_square_subsequent_mask(tgt_input.shape[1]).to(device)

        # 4. 前向传播与反向传播
        optimizer.zero_grad()  # 清空上一步的梯度，防止梯度累积
        # 将数据送入模型，注意目标端的填充掩码也要去掉最后一个词，以匹配 tgt_input 的长度
        output = model(src, tgt_input, src_pad_mask, tgt_pad_mask[:, :-1], tgt_mask)

        # 5. 计算损失
        # 将输出和目标张量展平为一维，以适配交叉熵损失函数的输入要求
        loss = loss_function(
            output.reshape(-1, output.shape[-1]),  # 形状变为 (batch_size * seq_len, vocab_size)
            tgt_output.reshape(-1)  # 形状变为 (batch_size * seq_len,)
        )

        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型参数

        total_loss += loss.item()  # 累加当前批次的损失值

    # 返回当前 Epoch 的平均损失
    return total_loss / len(dataloader)


def train():
    """
    训练主函数。负责初始化环境、模型，并执行多轮训练循环。
    """
    # 1. 环境初始化
    # 自动检测是否有可用的 GPU，有则使用 GPU，否则使用 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = get_dataloader()  # 获取训练集的 DataLoader

    # 2. 加载词表并实例化分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESS_DATA_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESS_DATA_DIR / 'en_vocab.txt')

    # 3. 实例化翻译模型并移动到指定设备
    model = TranslationModel(
        zh_tokenizer.vocab_size,  # 中文词表大小
        en_tokenizer.vocab_size,  # 英文词表大小
        zh_tokenizer.pad_token_index,  # 中文 <pad> 的索引
        en_tokenizer.pad_token_index  # 英文 <pad> 的索引
    ).to(device)

    # 4. 定义损失函数、优化器和 TensorBoard 日志记录器
    # ignore_index: 在计算损失时忽略目标序列中的 <pad>，避免填充符影响梯度
    loss_function = CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_index)
    # 使用 Adam 优化器，学习率从 config 中读取
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # 初始化 TensorBoard，按当前时间命名日志文件夹，方便对比不同次实验
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))

    # 5. 训练循环
    best_loss = float('inf')  # 记录历史最佳损失值，初始化为无穷大

    for epoch in range(1, config.EPOCHS + 1):
        print(f'========== Epoch {epoch} ==========')
        # 执行单轮训练，获取平均损失
        avg_loss = train_one_epoch(dataloader, model, loss_function, optimizer, device)
        print(f'平均损失: {avg_loss:.4f}')

        # 将当前 Epoch 的损失写入 TensorBoard，方便可视化训练曲线
        writer.add_scalar('Loss', avg_loss, epoch)

        # 6. 模型保存策略：仅当当前损失优于历史最佳损失时，才保存模型权重
        if avg_loss < best_loss:
            best_loss = avg_loss
            # 保存模型的 state_dict（仅保存参数，不保存模型结构，节省空间且更灵活）
            torch.save(model.state_dict(), config.MODEL_DIR / 'model.pt')
            print('模型已保存')
        else:
            print('未保存模型')


if __name__ == '__main__':
    # 只有直接运行此脚本时才启动训练
    train()