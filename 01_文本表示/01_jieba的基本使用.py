import jieba

text = "小明硕士毕业于中国科学院计算所，后在日本京都大学深造"
# 1. 分词的三种模式
# 1.1 默认模式
# 返回的是一个生成器
words = jieba.cut(text, cut_all=False)
print(list(words))
# 1.2 全模式
words_all = jieba.cut(text, cut_all=True)
print(list(words_all))
# 1.3 搜索引擎模式
words_for_search = jieba.cut_for_search(text)
print(list(words_for_search))

