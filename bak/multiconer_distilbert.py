import os
import sys
import logging

from pathlib import Path
import re
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, AdamW
from transformers import DistilBertForTokenClassification
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def get_docs(file_path):
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ''.join(sys.argv))

    file_path = Path(file_path)
    raw_text = file_path.read_text(encoding='utf-8').strip()
    raw_text = re.sub(r'# id[^#\n]*\n', "", raw_text)
    raw_docs = re.split(r'\n\n', raw_text)
    token_docs = []  # 元素为一个文档的所有token
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            token, tag = line.split(r' _ _ ')  # 一行中包含token和tag（用制表符分隔）
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs

def encode_tags(tags, encodings):
    '''分词器将一个标记拆分为多个子标记，那么最终会出现标记和标签之间的不匹配'''
    labels = [[tag2id[tag] for tag in doc]
              for doc in tags]  # 把tags转换映射id（未分词前）
    encoded_labels = []
    # offset_mapping指示子标记相对于从其拆分的原始标记的开始位置和结束位置。
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(
            len(doc_offset), dtype=int) * -100  # 长度以子标记数量为准
        arr_offset = np.array(doc_offset)

        # 将那些偏移首位为0，第二位不为零的子token，说明是原来token的第一个子token
        # 筛选下来数量就是分词器的token数，标记为对应的id，其他还是-100
        doc_enc_labels[(arr_offset[:, 0] == 0) & (
            arr_offset[:, 1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels

class NEDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}  # items就是键、值数组
        if not self.labels == None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':
    texts, tags = get_docs('train.conll')
    train_texts, val_texts, train_tags, val_tags = train_test_split(
        texts, tags, test_size=.2)
    test_texts, _ = get_docs('test.conll')

    unique_tags = set(tag for doc in tags for tag in doc)  # 有多少种tag并按序排号
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}  # 标签用数字表示

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
    train_encodings = tokenizer(train_texts, is_split_into_words=True,
                                return_offsets_mapping=True, padding=True, truncation=True)
    val_encodings = tokenizer(val_texts, is_split_into_words=True,
                              return_offsets_mapping=True, padding=True, truncation=True)
    test_encodings = tokenizer(test_texts, is_split_into_words=True,
                               return_offsets_mapping=True, padding=True, truncation=True)

    train_labels = encode_tags(train_tags, train_encodings)
    val_labels = encode_tags(val_tags, val_encodings)
    # we don't want to pass this to the model, 仅保留input_ids，即分词编码
    _ = train_encodings.pop("offset_mapping")
    _ = val_encodings.pop("offset_mapping")
    test_offset = test_encodings.pop("offset_mapping")

    train_dataset = NEDataset(train_encodings, train_labels)
    val_dataset = NEDataset(val_encodings, val_labels)
    test_dataset = NEDataset(test_encodings)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    model = DistilBertForTokenClassification.from_pretrained(
        'distilbert-base-cased', num_labels=len(unique_tags))
    model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    optim = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):
        for batch in tqdm(train_loader):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()

    model.eval()

    input_ids = test_dataset[:]['input_ids'].to(device)
    attention_mask = test_dataset[:]['attention_mask'].to(device)
    with torch.no_grad():
        y_pred = model(input_ids, attention_mask)
    y_pred = np.argmax(y_pred[0].detach().to(torch.device('cpu')), axis=-1)
    labels = []
    for doc_enc_labels, doc_offset in zip(y_pred.numpy(), test_offset):
        doc_offset = np.array(doc_offset)
        doc_labels = doc_enc_labels[
            (doc_offset[:, 0] == 0) & (doc_offset[:, 1] != 0)]
        labels.append(doc_labels)

    tags = [[id2tag[label] for label in doc]
            for doc in labels]  # 把tags转换映射id（未分词前）
    with open('gdis.txt', 'a') as f:
        for tag in tags:
            for entt in tag:
                print(str(entt), file=f)
            print('\n', file=f)
