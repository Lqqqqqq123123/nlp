import gensim
import torch
import torch.nn as nn
# 1. 创建一个 Embedding 层
# embed = nn.Embedding(num_embeddings=1000, embedding_dim=100, padding_idx=0)
# print(embed.weight.shape)
# print(embed.weight[0])


# 2. 加载训练好的 word2vec 模型
model = gensim.models.keyedvectors.load_word2vec_format('../model/mycustom_w2v.kv')
nums_embeddings = len(model.vectors)
embedding_dim = model.vectors.shape[1]

embed = nn.Embedding.from_pretrained(
    embeddings=torch.FloatTensor(
        model.vectors
    ),
    freeze=True
)

print(embed.weight.shape)

# 测试
wid = model.key_to_index['电脑']
t = torch.tensor([wid]).reshape(-1, 1)

out = embed(t)

print(out.shape, out, out.grad_fn)