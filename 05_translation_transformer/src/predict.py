# predict.py

import torch
from tokenizer import ChineseTokenizer, EnglishTokenizer
from model import TranslationModel
import config


def predict_batch(input_tensor, model, en_tokenizer, device):
    """
    批量生成翻译结果（自回归生成过程）。
    :param input_tensor: 中文输入张量，形状为 (batch_size, seq_len)。
    :param model: 翻译模型。
    :param en_tokenizer: 英文分词器（用于获取 <sos>, <eos>, <pad> 的索引）。
    :param device: 运行设备（CPU/GPU）。
    :return: 生成的英文索引列表的列表，形状为 (batch_size, generated_seq_len)。
    """
    # 1. 切换模型为评估模式（禁用 Dropout 等训练专属行为，确保推理结果稳定）
    model.eval()

    # 2. 禁用梯度计算。推理阶段不需要反向传播，这能显著降低显存占用并加速计算
    with torch.no_grad():
        # 生成源语言的填充掩码（假设 padding 的索引为 0）
        src_pad_mask = (input_tensor == 0)

        # 执行编码器前向传播，获取源语言的上下文记忆（memory）
        # 注意：在自回归生成时，编码器只需要执行一次，后续解码器会反复使用这个 memory
        memory = model.encode(src=input_tensor, src_pad_mask=src_pad_mask)

        batch_size = input_tensor.shape[0]

        # 初始化解码器的输入：每个样本都以 <sos>（起始符）开头
        decoder_input = torch.full(
            size=(batch_size, 1),
            fill_value=en_tokenizer.sos_token_index,
            device=device
        )

        # 记录每个样本生成的单词序列
        generated = [[] for _ in range(batch_size)]
        # 记录每个样本是否已经生成了 <eos>（结束符），用于提前终止生成
        finished = [False for _ in range(batch_size)]

        # 开始自回归循环：每次生成一个词，直到达到最大长度或所有样本都生成了 <eos>
        for step in range(1, config.MAX_LENGTH):
            # 生成当前步的因果掩码（防止看到未来）和填充掩码
            tgt_mask = model.transformer.generate_square_subsequent_mask(decoder_input.shape[1]).to(device)
            tgt_pad_mask = (decoder_input == en_tokenizer.pad_token_index)

            # 执行解码器前向传播
            decoder_output = model.decode(decoder_input, memory, tgt_mask, tgt_pad_mask, src_pad_mask)

            # 提取当前步（即序列最后一个位置）的预测结果，并取概率最大（argmax）的词索引
            # decoder_output 形状: (batch_size, current_seq_len, vocab_size)
            # predict_indexes 形状: (batch_size,)
            predict_indexes = decoder_output[:, -1, :].argmax(dim=-1)

            # 遍历批次中的每个样本，处理生成的词
            for i in range(batch_size):
                # 如果该样本已经结束，跳过
                if finished[i]:
                    continue

                # 如果当前预测的词是 <eos>，标记该样本为已完成，不将 <eos> 加入结果
                if predict_indexes[i].item() == en_tokenizer.eos_token_index:
                    finished[i] = True
                    continue

                # 将预测到的词索引追加到该样本的结果列表中
                generated[i].append(predict_indexes[i].item())

            # 如果批次中所有样本都已生成 <eos>，提前退出循环
            if all(finished):
                break

            # 将当前步预测的词拼接到解码器输入中，作为下一步的输入（自回归的核心）
            # unsqueeze(1) 将形状从 (batch_size,) 变为 (batch_size, 1) 以便拼接
            decoder_input = torch.cat([decoder_input, predict_indexes.unsqueeze(1)], dim=1)

        return generated


def predict(zh_sentence, model, zh_tokenizer, en_tokenizer, device):
    """
    翻译单句中文（对外的单条推理接口）。
    :param zh_sentence: 待翻译的中文字符串。
    :param model: 翻译模型。
    :param zh_tokenizer: 中文分词器。
    :param en_tokenizer: 英文分词器。
    :param device: 运行设备。
    :return: 翻译后的英文字符串。
    """
    # 1. 将中文句子编码为索引列表，并进行 Padding/截断处理
    input_ids = zh_tokenizer.encode(zh_sentence, seq_len=config.MAX_LENGTH, add_sos_eos=False)

    # 2. 转换为张量，并增加 batch 维度（因为 predict_batch 期望输入是二维的）
    input_tensor = torch.tensor([input_ids], device=device)

    # 3. 调用批量预测函数
    generated = predict_batch(input_tensor, model, en_tokenizer, device)

    # 4. 取出第一个（也是唯一一个）样本的生成结果
    en_indexes = generated[0]

    # 5. 将英文索引序列解码回字符串
    en_sentence = en_tokenizer.decode(en_indexes)

    return en_sentence


def run_predict():
    """
    启动交互式翻译系统。
    负责加载模型、分词器，并提供命令行交互循环。
    """
    # 1. 自动选择运行设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 加载中英文分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESS_DATA_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESS_DATA_DIR / 'en_vocab.txt')

    # 3. 实例化模型结构，并加载训练好的权重
    model = TranslationModel(
        zh_vocab_size=zh_tokenizer.vocab_size,
        en_vocab_size=en_tokenizer.vocab_size,
        zh_padding_index=zh_tokenizer.pad_token_index,
        en_padding_index=en_tokenizer.pad_token_index
    ).to(device)
    # 从指定路径加载模型参数（state_dict）
    model.load_state_dict(torch.load(config.MODEL_DIR / 'model.pt'))

    # 4. 交互式命令行循环
    print('欢迎使用翻译系统，请输入中文句子：（输入 q 或 quit 退出）')
    while True:
        user_input = input('中文：')

        # 退出条件
        if user_input in ['q', 'quit']:
            print('谢谢使用，再见！')
            break

        # 过滤空白输入
        if not user_input.strip():
            print('请输入内容')
            continue

        # 调用单句翻译函数并打印结果
        result = predict(user_input, model, zh_tokenizer, en_tokenizer, device)
        print(f'英文：{result}')


if __name__ == '__main__':
    run_predict()