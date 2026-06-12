import pandas as pd
import gensim
import jieba
"""
基于真实语料，去通过 gensim 训练一个 word2Vector 模型
"""

# 1. 处理数据集
df = pd.read_csv('./data/online_shopping_10_cats.csv').dropna()
df = df['review']
# print(df.head(10))
data = list(df.values)

# 2. 分词
sentences = [[word for word in list(jieba.cut(text)) if word.strip() != ''] for text in data]

# 3. 训练模型
model = gensim.models.Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    sg=1,
    min_count=3,
    workers=4,
    epochs=10,
)

print(model.wv.similarity('电脑', '手机'))
# 4. 保存模型
model.wv.save_word2vec_format('../model/mycustom_w2v.kv')




