from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer

from config import RAW_DATA_FILE, RAW_DATA_DIR, BASE_MODEL, MAX_LENGTH, PROCESS_DATA_DIR
# 数据集处理


def process_data():
    """
    预处理数据集，数据集路径：RAW_DATA_DIR / RAW_DATA_FILE
    :return:
    """

    # 1. 加载数据集
    ds = load_dataset('csv', data_files=str(RAW_DATA_DIR / RAW_DATA_FILE))['train']
    print(ds.column_names)

    # 2. 预处理
    ds = ds.remove_columns(['cat'])
    print(ds.shape)

    # 3. 过滤空行
    ds = ds.filter(lambda x: x['review'] is not None and x['label'] is not None)

    print(ds.shape)

    # 4. 划分数据集
    ds = ds.cast_column('label', ClassLabel(names=['neg', 'pos']))
    ds = ds.train_test_split(test_size=0.2, stratify_by_column='label')

    # print(ds['train'].num_rows, ds['test'].num_rows)

    # 5. 创建分词器
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # 6. 处理数据集
    def encode(x):
        inputs = tokenizer(
            x['review'],
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )

        inputs['labels'] = x['label']
        inputs['']
        return inputs

    ds = ds.map(encode, batched=True, batch_size=32, remove_columns=['review', 'label'])

    # 保存数据集
    ds.save_to_disk(PROCESS_DATA_DIR)





if __name__ == '__main__':
    process_data()