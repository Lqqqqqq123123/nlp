# model.py

import torch
from torch import nn
import config

class PositionEncoding(nn.Module):
    """
    位置编码（Positional Encoding）。
    因为 Transformer 没有像 RNN 那样的循环结构，无法感知序列中单词的先后顺序。
    位置编码通过正弦和余弦函数生成固定的位置向量，并将其加到词向量上，从而为模型注入位置信息。
    """
    def __init__(self, d_model, max_len=500):
        """
        初始化位置编码。
        :param d_model: 模型的隐藏层维度（即词向量的维度）。
        :param max_len: 支持的最大序列长度，默认 500。
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # 生成位置索引矩阵，形状为 (max_len, 1)
        pos = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)  
        # 生成用于计算分母的偶数序列，形状为 (d_model/2,)
        _2i = torch.arange(0, self.d_model, step=2, dtype=torch.float)  
        # 计算分母项：10000^(2i/d_model)
        div_term = torch.pow(10000, _2i / self.d_model)

        # 分别计算奇数维度的 sin 值和偶数维度的 cos 值
        sins = torch.sin(pos / div_term)  # sins.shape: (max_len, d_model/2)
        coss = torch.cos(pos / div_term)  # coss.shape: (max_len, d_model/2)

        # 初始化全零的位置编码矩阵
        pe = torch.zeros(self.max_len, self.d_model, dtype=torch.float)  # pe.shape: (max_len, d_model)
        # 将 sin 值填入偶数维度（0, 2, 4...），将 cos 值填入奇数维度（1, 3, 5...）
        pe[:, 0::2] = sins
        pe[:, 1::2] = coss

        # 使用 register_buffer 将 pe 注册为缓冲区。
        # 它不会被当作模型参数参与梯度更新，但会随模型自动保存、加载，并自动跟随模型切换设备（CPU/GPU）。
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播：将位置编码加到输入张量上。
        :param x: 输入张量，形状为 (batch_size, seq_len, d_model)。
        :return: 融合了位置信息的张量，形状不变。
        """
        seq_len = x.size(1)
        # 根据当前序列的实际长度截取对应的位置编码，并与输入相加
        return x + self.pe[:seq_len]


class TranslationModel(nn.Module):
    """
    基于 Transformer 的机器翻译模型。
    包含源语言和目标语言的词嵌入、位置编码、Transformer 核心网络以及最终的线性映射层。
    """
    def __init__(self, zh_vocab_size, en_vocab_size, zh_padding_index, en_padding_index):
        """
        初始化翻译模型。
        :param zh_vocab_size: 中文词表大小。
        :param en_vocab_size: 英文词表大小。
        :param zh_padding_index: 中文填充符（<pad>）在词表中的索引。
        :param en_padding_index: 英文填充符（<pad>）在词表中的索引。
        """
        super().__init__()
        # 中文（源语言）词嵌入层，padding_idx 用于确保填充符的向量全为 0，不参与计算
        self.src_embedding = nn.Embedding(num_embeddings=zh_vocab_size, embedding_dim=config.D_MODEL,
                                          padding_idx=zh_padding_index)
        # 英文（目标语言）词嵌入层
        self.tgt_embedding = nn.Embedding(num_embeddings=en_vocab_size, embedding_dim=config.D_MODEL,
                                          padding_idx=en_padding_index)
        # 实例化位置编码模块
        self.position_encoding = PositionEncoding(d_model=config.D_MODEL)

        # 实例化 PyTorch 内置的 Transformer 模块
        self.transformer = nn.Transformer(
            d_model=config.D_MODEL,           # 模型特征维度
            nhead=config.NHEAD,             # 多头注意力机制的头数
            num_encoder_layers=config.NUM_ENCODER_LAYERS,  # 编码器层数
            num_decoder_layers=config.NUM_DECODER_LAYERS,  # 解码器层数
            batch_first=True                    # 指定输入张量的第一维为 batch_size
        )

        # 线性映射层，将 Transformer 输出的隐藏状态映射回英文词表大小，用于计算词汇概率分布
        self.linear = nn.Linear(config.D_MODEL, en_vocab_size)

    def encode(self, src, src_pad_mask):
        """
        编码器前向传播。
        :param src: 源语言（中文）输入张量，形状 (batch_size, src_seq_len)。
        :param src_pad_mask: 源语言的填充掩码，用于屏蔽 <pad> 位置的注意力。
        :return: 编码器的输出记忆（memory），形状 (batch_size, src_seq_len, d_model)。
        """
        src_embed = self.src_embedding(src)          # 词嵌入
        src_embed = self.position_encoding(src_embed) # 添加位置编码
        # 传入 Transformer 编码器
        memory = self.transformer.encoder(src=src_embed, src_key_padding_mask=src_pad_mask)
        return memory

    def decode(self, tgt, memory, tgt_mask, tgt_pad_mask, src_pad_mask):
        """
        解码器前向传播。
        :param tgt: 目标语言（英文）输入张量，形状 (batch_size, tgt_seq_len)。
        :param memory: 编码器的输出。
        :param tgt_mask: 目标语言的因果掩码（Causal Mask），防止解码器在训练时看到未来的词。
        :param tgt_pad_mask: 目标语言的填充掩码。
        :param src_pad_mask: 源语言的填充掩码（在交叉注意力中屏蔽源语言的 <pad>）。
        :return: 解码器的输出，经过线性层映射后形状为 (batch_size, tgt_seq_len, en_vocab_size)。
        """
        tgt_embed = self.tgt_embedding(tgt)          # 词嵌入
        tgt_embed = self.position_encoding(tgt_embed) # 添加位置编码
        # 传入 Transformer 解码器
        output = self.transformer.decoder(
            tgt=tgt_embed, 
            memory=memory, 
            tgt_mask=tgt_mask,                # 因果掩码
            tgt_key_padding_mask=tgt_pad_mask, # 目标端填充掩码
            memory_key_padding_mask=src_pad_mask # 源端填充掩码
        )
        # 将解码器输出通过线性层，得到词表大小的 logits
        return self.linear(output)

    def forward(self, src, tgt, src_pad_mask, tgt_pad_mask, tgt_mask):
        """
        模型整体的前向传播。
        串联编码器和解码器，返回最终的预测结果。
        """
        memory = self.encode(src, src_pad_mask)
        output = self.decode(tgt, memory, tgt_mask, tgt_pad_mask, src_pad_mask)
        return output