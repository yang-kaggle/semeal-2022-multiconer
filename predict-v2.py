import torch
from torch.utils.data import DataLoader
from transformers import logging
from transformers import AutoModelForTokenClassification, AutoTokenizer
from tqdm import tqdm
from model import NERModel
from utils import conllUtils, NERDataset

if __name__ == '__main__':
    lang = 'mix'
    num_labels = 13
    is_remote = True
    model_name = 'xlm-roberta-large'
    model_dir = None if is_remote else 'E:/cache/huggingface/transformers'
    logging.set_verbosity_error()

    print("Preprocessing ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True, cache_dir=model_dir)
    valid_data = conllUtils.encode(lang, tokenizer, kind="valid", label_all_tokens=False)
    valid_dataset = NERDataset(valid_data)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = NERModel(model_name, num_labels=num_labels, cache_dir=model_dir)

    model.eval()
    model.load_state_dict(torch.load(f'./models/{model_name}.pth', map_location='cpu'))

    print("Predicting ...")
    labels = []
    model = model.to(device)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
    for batch in tqdm(valid_loader):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            output, _ = model(input_ids, attention_mask)
            labels.extend(output.argmax(axis=-1).tolist())
    pred_labels = conllUtils.decode(labels, valid_data)
    true_labels = conllUtils.decode(valid_data['labels'], valid_data)

    print("Metricating ...")
    pred_tags = conllUtils.predict(pred_labels)
    true_tags = conllUtils.predict(true_labels)
    result = conllUtils.compute_metric(pred_tags, true_tags)
    print(result['overall_f1'])
    conllUtils.save_result_zip(pred_tags, lang, result['overall_f1'], model_name)
