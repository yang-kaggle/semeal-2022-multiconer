import os
import zipfile
from datasets import load_metric
from utils.utils import *

lang2dir = {"bn": "BN-Bangla", "de": "DE-German", "en": "EN-English", "es": "ES-Spanish", "fa": "FA-Farsi",
            "hi": "HI-Hindi", "ko": "KO-Korean", "mix": "MIX_Code_mixed", "multi": "MULTI_Multilingual",
            "nl": "NL-Dutch", "ru": "RU-Russian", "tr": "TR-Turkish", "zh": "ZH-Chinese", "te": "Te-Test"}
postfix = {"train": "_train.conll", "valid": "_dev.conll", "test": "_test.conll"}
tag2id = {'B-GRP': 0, 'B-CW': 1, 'I-PER': 2, 'I-CW': 3, 'B-CORP': 4, 'I-CORP': 5, 'I-LOC': 6,
          'I-PROD': 7, 'B-LOC': 8, 'I-GRP': 9, 'B-PROD': 10, 'O': 11, 'B-PER': 12}
id2tag = {0: 'B-GRP', 1: 'B-CW', 2: 'I-PER', 3: 'I-CW', 4: 'B-CORP', 5: 'I-CORP', 6: 'I-LOC',
          7: 'I-PROD', 8: 'B-LOC', 9: 'I-GRP', 10: 'B-PROD', 11: 'O', 12: 'B-PER'}


def get_data(lang, kind):
    """texts为原始文档 即单词列表（多个构成文档列表），tag为原始标签 即各单词对应实体名称列表（多个构成文档标签列表）"""
    data_dir = './data/public_data/' + lang2dir[lang]
    data_path = os.path.join(data_dir, lang + postfix[kind])
    texts, tags = get_docs(data_path)
    if kind == 'test':
        return texts, None
    else:
        return texts, tags


def tokenize_and_align_labels(texts, tags, tokenizer, label_all_tokens=True):
    tokenized_inputs = tokenizer(texts,
                                 is_split_into_words=True,
                                 # return_offsets_mapping=True,
                                 padding=True, truncation=True)
    if tags is not None:
        raw_labels = [[tag2id[tag] for tag in doc] for doc in tags]
        labels = []
        for i, raw_label in enumerate(raw_labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label.append(raw_label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label.append(raw_label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx
            labels.append(label)
        tokenized_inputs["labels"] = labels
    return tokenized_inputs


def encode(lang, tokenizer, kind, label_all_tokens=True):
    texts, tags = get_data(lang, kind)
    tokenized_inputs = tokenize_and_align_labels(texts, tags, tokenizer, label_all_tokens)
    return tokenized_inputs


def decode(labels, tokenized_inputs):
    raw_labels = []
    # print(tokenized_inputs.word_ids(batch_index=0), len(tokenized_inputs.word_ids(batch_index=0)))
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        raw_label = []
        sub_label = []
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            elif word_idx != previous_word_idx:
                if sub_label:
                    raw_label.append(sub_label[0])
                sub_label = [label[idx]]
            else:
                sub_label.append(label[idx])
            previous_word_idx = word_idx
        raw_label.append(sub_label[0])
        raw_labels.append(raw_label)
    return raw_labels


def decode2(labels, tokenized_inputs):
    raw_labels = []
    # print(tokenized_inputs.word_ids(batch_index=0), len(tokenized_inputs.word_ids(batch_index=0)))
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        raw_label = []
        sub_label = {}
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            elif word_idx != previous_word_idx:
                if sub_label:
                    raw_label.append(sub_label[0])
                sub_label = [label[idx]]
            else:
                sub_label.append(label[idx])
            previous_word_idx = word_idx
        raw_label.append(sub_label[0])
        raw_labels.append(raw_label)
    return raw_labels


def predict(labels):
    tags = [[id2tag[label] for label in doc] for doc in labels]
    return tags


def compute_metric(predictions, references):
    metric = load_metric('seqeval')
    results = metric.compute(predictions=predictions, references=references, scheme='IOB2')
    return results


def save_result_zip(tags, lang, score=0.00, model_name=None):
    """不指定模型名称，则直接在预测后生成目标文件"""
    filename = lang + '.pred' if model_name is None else f'{lang}_{model_name}_{score * 100:.2f}'
    result_dir = './results'
    result_path = os.path.join(result_dir, filename + ".conll")
    with open(result_path, 'w') as f:  # w覆盖写，a追加写
        for doc in tags:
            for tag in doc:
                print(str(tag), file=f)
            print('', file=f)
    if not model_name:
        pack_path = os.path.join(result_dir, 'my_submission.zip')
        zfile = zipfile.ZipFile(pack_path, 'w')
        zfile.write(result_path, filename + ".conll")
        zfile.close()
