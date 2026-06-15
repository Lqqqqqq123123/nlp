import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 每个头的维度

        # 1. 定义 Q, K, V 投影矩阵 (这里我们直接投影到整个 dim，稍后在 forward 中拆分)
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)

        # 2. 定义最终的输出投影矩阵
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        """
        Args:
            x: 输入张量 shape: (batch_size, seq_len, dim)
        Returns:
            output: 多头注意力输出 shape: (batch_size, seq_len, dim)
            attn_weights: 注意力权重 shape: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, dim = x.shape

        # 1. 线性投影得到 Q, K, V
        q = self.W_q(x)  # (batch, seq_len, dim)
        k = self.W_k(x)  # (batch, seq_len, dim)
        v = self.W_v(x)  # (batch, seq_len, dim)

        # 2. 拆分多头 (Reshape 并转置)
        # 将 (batch, seq_len, dim) 变为 (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 计算缩放点积注意力分数
        # (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # 4. Softmax 得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # (batch, num_heads, seq_len, seq_len)

        # 5. 加权求和
        # (batch, num_heads, seq_len, head_dim)
        context = torch.matmul(attn_weights, v)

        # 6. 拼接多头 (Concat)
        # 转置回 (batch, seq_len, num_heads, head_dim)，然后合并最后两维
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)

        # 7. 最终线性投影
        output = self.out_proj(context)

        return output, attn_weights