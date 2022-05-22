import re
import torch
import numpy as np
from pathlib import Path


def get_docs(file_path):
    """return token list[num_docs, num_words] and label list[num_doc, num_labels]"""
    file_path = Path(file_path)
    raw_text = file_path.read_text(encoding='utf-8').strip()
    raw_text = re.sub(r'# id[^#\n]*\n', "", raw_text)
    raw_docs = re.split(r'\n[\n]+', raw_text)
    token_docs = []  # 元素为一个文档的所有token
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            token, tag = line.split(r' _ _')  # 一行中包含token和tag（用制表符分隔）
            if token == '\u200b\u200b' or token == '\u200c':
                token = '.'
            tag = tag.strip()
            if token[:2] in ['O-', 'B-', 'I-']:
                tokens.append(tag)
                tags.append(token)
            else:
                tokens.append(token)
                tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)
    return token_docs, tag_docs


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.inputs = inputs

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.inputs.items()}
        return item

    def __len__(self):
        # 这里很重要，不能填写self.inputs，对此len的结果是很小的，就是inputs，offset字段的数量
        return len(self.inputs['input_ids'])
