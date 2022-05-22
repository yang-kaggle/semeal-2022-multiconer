import re
import torch
import numpy as np
from pathlib import Path


def get_docs(file_path):
    """return token list[num_docs, num_words] and label list[num_doc, num_labels]"""
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


def encode_labels_offset(labels, encodings):
    """分词器将一个标记拆分为多个子标记，那么最终会出现标记和标签之间的不匹配"""
    encoded_labels = []

    # offset_mapping指示子标记相对于从其拆分的原始标记的开始位置和结束位置。
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100  # 长度以子标记数量为准
        arr_offset = np.array(doc_offset)  # 否则无法进行下一行的操作
        # 将那些偏移首位为0，第二位不为零的子token，说明是原来token的第一个子token
        # 筛选下来数量就是分词器的token数，标记为对应的id，其他还是-100
        doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())
    return encoded_labels


def encode_labels_keyword(labels, encodings, tokenizer, keyword):
    """分词器将一个标记拆分为多个子标记，那么最终会出现标记和标签之间的不匹配"""
    encoded_labels = []

    # offset_mapping指示子标记相对于从其拆分的原始标记的开始位置和结束位置。
    for doc_labels, doc_input_ids in zip(labels, encodings.input_ids):
        doc_tokens = tokenizer.convert_ids_to_tokens(doc_input_ids)
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_tokens), dtype=int) * -100  # 长度以子标记数量为准
        i = 0
        for idx, token in enumerate(doc_tokens):
            if token[0] == keyword:
                doc_enc_labels[idx] = doc_labels[i]
                i += 1
        encoded_labels.append(doc_enc_labels.tolist())
    return encoded_labels


def decode_labels_keyword(encoded_labels, input_ids, tokenizer, keyword):
    labels = []
    for doc_enc_labels, doc_input_ids in zip(encoded_labels, input_ids):
        doc_tokens = tokenizer.convert_ids_to_tokens(doc_input_ids)
        doc_labels = []
        for idx, token in enumerate(doc_tokens):
            if token[0] == keyword:
                doc_labels.append(doc_enc_labels[idx])
        labels.append(doc_labels)
    return labels


def decode_labels_offset(encoded_labels, offset_mapping):
    labels = []
    for doc_enc_labels, doc_offset in zip(encoded_labels, offset_mapping):
        doc_offset = np.array(doc_offset)
        doc_labels = doc_enc_labels[(doc_offset[:, 0] == 0) & (doc_offset[:, 1] != 0)]
        labels.append(doc_labels)
    return labels


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}  # items就是键、值数组
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
