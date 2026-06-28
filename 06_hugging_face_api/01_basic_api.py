from transformers import AutoModel
from huggingface_hub import login, constants



print("当前实际生效的缓存路径是:", constants.HF_HUB_CACHE)
model_name = 'google-bert/bert-base-chinese'
model = AutoModel.from_pretrained(model_name, dtype='auto', device_map='auto')

print(model)

print(model.embeddings.word_embeddings.weight.shape)

