import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import logging
from tqdm import tqdm
from utils import conllUtils, NERDataset

if __name__ == '__main__':
    num_labels = 13
    lang, kind, model_name, is_remote = 'mix', "valid", 'xlm-roberta-large', False
    model_dir = None if is_remote else 'E:/cache/huggingface/transformers'
    logging.set_verbosity_error()

    print("Preprocessing ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True, cache_dir=model_dir)
    data = conllUtils.encode(lang, tokenizer, kind=kind, label_all_tokens=False)
    dataset = NERDataset(data)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels, cache_dir=model_dir).to(device)

    model.eval()
    model.load_state_dict(torch.load(f'./models/mono/{model_name}-{lang}.pth', map_location=device))
    print("Predicting ...")
    labels = []
    valid_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    a = 1
    for batch in tqdm(valid_loader):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            output = model(input_ids, attention_mask).logits
            if a == 1:
                print(output, output.shape)
                print(input_ids, input_ids.shape)
                a = 2
            labels.extend(output.argmax(axis=-1).tolist())
    pred_labels = conllUtils.decode(labels, data)
    pred_tags = conllUtils.predict(pred_labels)
    if kind == 'valid':
        print("Metricating ...")
        true_labels = conllUtils.decode(data['labels'], data)
        true_tags = conllUtils.predict(true_labels)
        result = conllUtils.compute_metric(pred_tags, true_tags)
        print(result['overall_f1'])
        conllUtils.save_result_zip(pred_tags, lang, result['overall_f1'], model_name)
    else:
        conllUtils.save_result_zip(pred_tags, lang)
