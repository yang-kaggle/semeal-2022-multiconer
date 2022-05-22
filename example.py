import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from utils.conllUtils import *

lang = 'en'
is_remote = False
model_name = "xlm-roberta-base"
model_dir = None if is_remote else 'E:/cache/huggingface/transformers'
data_dir = None if is_remote else  'E:/cache/huggingface/datasets'
metric_dir = None if is_remote else 'E:/cache/huggingface/metrics'
tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True, cache_dir=model_dir)


def wnut_basic_info():
    wnut = load_dataset("wnut_17", cache_dir=data_dir)
    # include train, validation, test three subset
    print(wnut['train'])

    # for each subset, each sample, using [index]
    # contain information 'id', 'tokens', 'ner_tags'
    # other dataset may have extra 'pos_tags', 'chunk_tags'
    print(wnut["train"][0]['tokens'], len(wnut["train"][0]['tokens']))

    # obtain all samples by certain [key]
    # actually [key] and [index] can simply swap, results are the same
    print(wnut['train']['tokens'])

    # net_tag are stored with number, to revel the entity name by following
    print(wnut["train"].features[f"ner_tags"].feature.names)


def tokenize():
    wnut = load_dataset("wnut_17", cache_dir=data_dir)
    source = wnut['train'][0]['tokens']
    source = [['我是学生，我爱学习'], ['哈哈哈，耗力耗']]
    print(source)
    tokenized_input = tokenizer(source,
                                is_split_into_words=True,
                                # return_offsets_mapping=True
                                )

    # tokenized_input contain information input_id, attention_mask, offset_mapping(if set)
    print(tokenized_input)

    # obtain above information with method item (return a dict)
    print(tokenized_input.items())

    # first key, then index to obtain one sample
    # tokenized_input.input_ids equal to tokenized_input[input_ids]
    print(tokenized_input["input_ids"][0], len(tokenized_input["input_ids"][0]))  # all set to the same tokenized size

    # convert one sample to its tokenized word with key “input_ids”
    print(tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"][0]))

    # convert one sample's tokenized sub-words' original index in un-tokenized text (batch_index to set with sample)
    print(tokenized_input.word_ids(batch_index=0), len(tokenized_input.word_ids(batch_index=0)))


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def apply_tokenizer():
    wnut = load_dataset("wnut_17", cache_dir=data_dir)
    train_data = wnut["train"]
    tokenized_wnut = train_data.map(tokenize_and_align_labels, batched=True)
    print(tokenized_wnut)


def encode_decode_test():
    print("Tokenizing ...")
    train_data = encode(lang, tokenizer, kind="valid", label_all_tokens=True)
    print(train_data['input_ids'][0], len(train_data['input_ids'][0]))
    print(train_data['labels'][0], len(train_data['labels'][0]))
    tokens = tokenizer.convert_ids_to_tokens(train_data['input_ids'][0])
    print(tokens, len(tokens))

    print("decoding ...")
    labels = decode(train_data['labels'], train_data)
    print(labels[0], len(labels[0]))


def metric_test():
    m = load_metric('seqeval')
    a = [['O', 'B-CW'], ['B-PER']]
    b = [['O', 'B-CW'], ['B-PER']]
    print(m.compute(predictions=a, references=b))


def append_write_test():
    result_path = './a.txt'
    a = [['a', 'b'], ['c', 'd']]
    b = [['e', 'f'], ['g', 'h']]
    for tags in [a, b]:
        with open(result_path, 'a') as f:  # w覆盖写，a追加写
            for doc in tags:
                for tag in doc:
                    print(str(tag), file=f)
                print('', file=f)


if __name__ == '__main__':
    tokenize()
    # wnut_basic_info()
    # metric_test()
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", cachr_dir=model_dir)
    # print(tokenizer("Using a Transformer network is simple"))
