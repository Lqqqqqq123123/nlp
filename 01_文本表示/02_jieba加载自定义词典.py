import jieba
# 加载自定义词典
# jieba.load_userdict(r'./data/mydict.txt')

text = "我爱吃大香蕉"
# 对于大香蕉，如果不加载自定义词典，则分词结果为大，香蕉

print(list(jieba.cut(text, cut_all=False)))