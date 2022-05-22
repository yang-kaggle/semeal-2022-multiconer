import os
import zipfile
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForTokenClassification
from utils.utilsbakbak import *

lang = "en"
lang2dir = {"be": "BN-Bangla", "de": "DE-German", "en": "EN-English", "ES": "ES-Spanish", "fa": "FA-Farsi",
            "hi": "HI-Hindi", "kr": "KO-Korean", "mix": "MIX_Code_mixed", "multi": "MULTI_Multilingual",
            "nl": "NL-Dutch", "ru": "RU-Russian", "tr": "TR-Turkish", "zh": "ZH-Chinese"}
postfix = {"train": "_train.conll", "valid": "_dev.conll", "test": "_test.conll"}
keywords = {'roberta-base': 'Ġ', 'albert-base-v2': '▁', 'distilbert-base-cased': None}
tag2id = {'B-GRP': 0, 'B-CW': 1, 'I-PER': 2, 'I-CW': 3, 'B-CORP': 4, 'I-CORP': 5, 'I-LOC': 6,
          'I-PROD': 7, 'B-LOC': 8, 'I-GRP': 9, 'B-PROD': 10, 'O': 11, 'B-PER': 12}
id2tag = {0: 'B-GRP', 1: 'B-CW', 2: 'I-PER', 3: 'I-CW', 4: 'B-CORP', 5: 'I-CORP', 6: 'I-LOC',
          7: 'I-PROD', 8: 'B-LOC', 9: 'I-GRP', 10: 'B-PROD', 11: 'O', 12: 'B-PER'}


def get_rawdata(kind):
    """texts为原始文档 即单词列表（多个构成文档列表），tag为原始标签 即各单词对应实体名称列表（多个构成文档标签列表）"""
    data_dir = './data/training_data/' + lang2dir[lang]
    data_path = os.path.join(data_dir, lang + postfix[kind])
    texts, tags = get_docs(data_path)
    if kind == 'test':
        return texts, None
    else:
        return texts, tags


def encode(tokenizer, kind):
    model_name = tokenizer.name_or_path
    raw_texts, raw_tags = get_rawdata(kind)
    labels = [[tag2id[tag] for tag in doc] for doc in raw_tags]  # 把tags转换映射id（未分词前）
    texts_encodings = tokenizer(raw_texts,
                                is_split_into_words=True,
                                return_offsets_mapping=True,
                                padding=True, truncation=True)

    if kind == "train" or kind == "valid":
        keyword = keywords[model_name]
        if keyword is not None:
            labels_encoding = encode_labels_keyword(labels, texts_encodings, tokenizer, keyword)
        else:
            labels_encoding = encode_labels_offset(labels, texts_encodings)
        return texts_encodings, labels_encoding
    elif kind == "test":
        return texts_encodings, None


def decode(encoded_labels, input_ids, offset_mapping, tokenizer=None):
    if tokenizer is not None:
        model_name = tokenizer.name_or_path
        keyword = keywords[model_name]
        labels = decode_labels_keyword(encoded_labels, input_ids, tokenizer, keyword)
    else:
        labels = decode_labels_offset(encoded_labels, offset_mapping)
    return labels


def predict(labels):
    tags = [[id2tag[label] for label in doc] for doc in labels]
    return tags


def get_true_labels(kind):
    _, tags = get_rawdata(kind)
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    return labels


def compute_metric(predictions, labels):
    metrics = [load_metric(i) for i in ['f1', 'recall', 'precision']]
    preds = []
    trues = []
    for prediction in predictions:
        preds.extend(prediction)
    for label in labels:
        trues.extend(label)

    results = {}
    for result in [metric.compute(predictions=preds, references=trues, average='macro') for metric in metrics]:
        results.update(result)
    return results


def save_result_zip(tags, model_name=lang + '.pred'):
    """不指定模型名称，则直接在预测后生成目标文件"""
    result_dir = './results'
    result_path = os.path.join(result_dir, model_name + ".conll")
    with open(result_path, 'a') as f:
        for doc in tags:
            for tag in doc:
                print(str(tag), file=f)
            print('', file=f)
    if model_name == lang + '.pred':
        pack_path = os.path.join(result_dir, 'my_submission.zip')
        zfile = zipfile.ZipFile(pack_path, 'w')
        zfile.write(result_path, model_name + ".conll")
        zfile.close()

if __name__ == '__main__':
    a = [[1,2,3,2], [5,4,3,2]]
    b = [[2,4,3,2], [5,3,3,1]]
    print(compute_metric(a, b)['f1'])
