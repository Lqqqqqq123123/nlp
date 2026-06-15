import torch
import torch.nn as nn

class selfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # 定义权重矩阵
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)


    def forward(self, x):

        batch_size, seq_len, dim = x.shape # 批次在前
        # 计算 q,k,v
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # 计算注意力分数
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / torch.sqrt(torch.tensor(self.dim))

        # 获取注意力权重
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # 获取注意力输出
        attn_output = torch.bmm(attn_weights, v)

        return attn_output, attn_weights

if __name__ == '__main__':
    x = torch.randn(1, 10, 512)
    self_attn = selfAttention(512)
    attn_output, attn_weights = self_attn(x)
    print(attn_output.shape)
    print(attn_weights.shape)

