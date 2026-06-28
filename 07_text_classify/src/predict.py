import torch
import torch.nn as nn
from torch import return_types
from model import MyModel
from config import BASE_MODEL, BEST_MODEL, MODEL_DIR, MAX_LENGTH
from transformers import AutoTokenizer



tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MyModel().to(device=device)
model.load_state_dict(torch.load(MODEL_DIR / BEST_MODEL))




def predict_batch(model, inputs, device):
    """
    传入一批数据，返回这批数据的预测结果
    :param model:
    :param inputs:
    :param device:
    :return:
    """
    model.eval()
    # 前向传播
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

    # 将预测输出转换为概率
    outputs = torch.sigmoid(outputs)

    return outputs.tolist()


def predict(text, model, tokenizer, device):
    """
    预测文本
    :param text:
    :param model:
    :param tokenizer:
    :param device:
    :return:
    """
    # 1. 预处理输出
    inputs = tokenizer(
        text,
        padding='max_length',
        max_length=MAX_LENGTH,
        truncation=True,
        return_tensors='pt'
    )

    # 2. 预测 (batch, 1),得到概率
    out = predict_batch(model, inputs, device)

    return out



def run_app():

    print('欢迎来到文本分类系统，输入 exit 退出系统')

    while True:
        text = input('请输入要预测的文本：')
        if text == 'exit':
            print('谢谢使用，再见！')
            break
        out = predict(text, model, tokenizer, device)
        print(out)




if __name__ == '__main__':
    run_app()

