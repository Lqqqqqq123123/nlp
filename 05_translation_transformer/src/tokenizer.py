# tokenizer.py

from abc import abstractmethod
from nltk import word_tokenize, TreebankWordDetokenizer
from tqdm import tqdm
import config

class BaseTokenizer:
    """
    分词器基类。
    定义了文本分词、词表构建、文本与索引序列互转的通用接口和基础功能。
    """
    # 定义四个常用的特殊Token，在构建词表时会默认放在最前面
    unk_token = config.UNK_TOKEN  # 未知词 (Unknown)，用于替换词表中不存在的词
    pad_token = config.PAD_TOKEN  # 填充词 (Padding)，用于将不同长度的序列对齐到相同长度
    sos_token = config.SOS_TOKEN  # 序列开始标记 (Start of Sentence)
    eos_token = config.EOS_TOKEN  # 序列结束标记 (End of Sentence)

    @staticmethod
    @abstractmethod
    def tokenize(sentence):
        """
        【抽象方法】对输入的句子进行分词。
        子类必须实现此方法以提供具体的分词逻辑。
        :param sentence: 待分词的原始字符串。
        :return: 分词后的单词/字符列表。
        """
        pass

    @abstractmethod
    def decode(self, indexes):
        """
        【抽象方法】将索引序列还原为文本。
        子类必须实现此方法以提供具体的文本拼接逻辑。
        :param indexes: 整数索引列表。
        :return: 还原后的字符串。
        """
        pass

    @classmethod
    def build_vocab(cls, sentences, vocab_file):
        """
        构建词表并保存到本地文件。
        :param sentences: 句子列表（用于遍历提取词汇）。
        :param vocab_file: 词表保存的文件路径。
        """
        unique_words = set()  # 使用集合(set)来自动去重
        # 使用tqdm显示分词进度条
        for sentence in tqdm(sentences, desc='分词'):
            for word in cls.tokenize(sentence):
                unique_words.add(word)

        # 将特殊Token放在词表最前面，保证它们的索引值固定
        vocab_list = [cls.pad_token, cls.unk_token, cls.sos_token, cls.eos_token] + list(unique_words)

        # 将词表逐行写入文件
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for word in vocab_list:
                f.write(word + '\n')

    def __init__(self, vocab_list):
        """
        初始化分词器。
        :param vocab_list: 词汇列表（通常从文件中读取）。
        """
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)  # 词表大小

        # 构建双向映射字典，方便在词和索引之间快速查找
        self.word2index = {word: i for i, word in enumerate(vocab_list)}
        self.index2word = {i: word for i, word in enumerate(vocab_list)}

        # 提前获取特殊Token的索引，避免在encode时频繁查字典
        self.unk_token_index = self.word2index[self.unk_token]
        self.pad_token_index = self.word2index[self.pad_token]
        self.sos_token_index = self.word2index[self.sos_token]
        self.eos_token_index = self.word2index[self.eos_token]

    @classmethod
    def from_vocab(cls, vocab_file):
        """
        从本地文件加载词表并实例化分词器。
        :param vocab_file: 词表文件路径。
        :return: 实例化后的分词器对象。
        """
        with open(vocab_file, 'r', encoding='utf-8') as f:
            # 读取文件并去除每行末尾的换行符
            vocab_list = [line.strip() for line in f.readlines()]
        return cls(vocab_list)

    def encode(self, sentence, seq_len, add_sos_eos=False):
        """
        将文本句子编码为固定长度的索引序列。
        :param sentence: 待编码的原始句子。
        :param seq_len: 目标序列长度（用于截断或填充）。
        :param add_sos_eos: 是否在序列首尾添加 <sos> 和 <eos> 标记。
        :return: 固定长度的整数索引列表。
        """
        # 1. 分词
        tokens = self.tokenize(sentence)

        # 2. 将词转换为索引，如果词不在词表中，则使用 <unk> 的索引
        indexes = [self.word2index.get(token, self.unk_token_index) for token in tokens]

        # 3. 处理序列长度（截断或添加特殊标记）
        if add_sos_eos:
            # 如果需要添加首尾标记，则内容最多只能占 seq_len - 2 的长度
            indexes = indexes[:seq_len - 2]
            indexes = [self.sos_token_index] + indexes + [self.eos_token_index]
        else:
            # 否则直接截断到 seq_len
            indexes = indexes[:seq_len]

        # 4. 填充 (Padding)：如果当前长度不足 seq_len，则在末尾补充 <pad>
        if len(indexes) < seq_len:
            indexes += [self.pad_token_index] * (seq_len - len(indexes))

        return indexes


class ChineseTokenizer(BaseTokenizer):
    """
    中文分词器。
    采用最基础的字符级（Character-level）分词，将句子直接拆分为单个汉字/字符。
    """

    @staticmethod
    def tokenize(sentence):
        """
        将中文字符串转换为字符列表。
        例如: "你好世界" -> ["你", "好", "世", "界"]
        """
        return list(sentence)

    def decode(self, indexes):
        """
        将索引列表还原为中文字符串。
        中文单词之间不需要空格，直接拼接即可。
        """
        return ''.join([self.index2word[i] for i in indexes])


class EnglishTokenizer(BaseTokenizer):
    """
    英文分词器。
    使用 NLTK 库进行词级别（Word-level）分词，能正确处理英文标点和缩写。
    """

    @staticmethod
    def tokenize(sentence):
        """
        使用 nltk.word_tokenize 对英文句子进行分词。
        例如: "Hello, world!" -> ["Hello", ",", "world", "!"]
        """
        return word_tokenize(sentence)

    def decode(self, indexes):
        """
        将索引列表还原为英文句子。
        使用 TreebankWordDetokenizer 智能处理单词与标点之间的空格。
        例如: ["Hello", ",", "world", "!"] -> "Hello, world!"
        """
        tokens = [self.index2word[i] for i in indexes]
        return TreebankWordDetokenizer().detokenize(tokens)