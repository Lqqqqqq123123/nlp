import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from config import BASE_MODEL
from transformers import AutoModel, BertModel


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # BERT 层
        self.bert:BertModel = AutoModel.from_pretrained(BASE_MODEL, add_pooling_layer=True)
        # 分类层
        self.cls = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids = None, labels = None):
        # 先走 bert 层，传入的参数就是 ids, attention_mask, token_type_ids, labels
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        # 然后获取最后一层的 pooler
        pooler_output = outputs.pooler_output # [batch_size, 768]

        # 获取分类结果
        out = self.cls(pooler_output) # [batch_size, 1]

        # 返回(batch_size, )
        return out.squeeze(-1)




if __name__ == '__main__':
    # bert:BertModel = AutoModel.from_pretrained(BASE_MODEL)
    # print(bert.config.hidden_size)
    model = MyModel()
    print(model)




