# 前基础上做小修改，修饰了无关紧要的东西
import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from transformers import AdamW, AutoTokenizer
from transformers import AutoModelForTokenClassification
from tqdm import tqdm
from utils import conllUtilsbak
from utils.utilsbak import NERDataset

if __name__ == '__main__':
    lang = 'en'
    num_labels = 13
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    train_data = conllUtilsbak.encode(lang, tokenizer, kind="train")
    valid_data = conllUtilsbak.encode(lang, tokenizer, kind="valid")
    train_dataset = NERDataset(*train_data)
    valid_dataset = NERDataset(*valid_data)
    # test_dataset = NERDataset(*test_data)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels).to(device)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    model.train()
    for epoch in range(3):
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), './models/roberta.pth')

    # model.eval()
    # model.load_state_dict(torch.load(f'./models/{model_name}.pth', map_location=device))
    # input_ids = valid_dataset[:]['input_ids'].to(device)
    # attention_mask = valid_dataset[:]['attention_mask'].to(device)
    # offset_mapping = valid_dataset[:]['offset_mapping'].to(device)
    # with torch.no_grad():
    #     output = model(input_ids, attention_mask).logits
    # pred_encoded_labels = output.argmax(axis=-1-trash).numpy()
    # pred_labels = conllUtils.decode(pred_encoded_labels, input_ids, offset_mapping, tokenizer)
    # true_encoded_labels = valid_dataset[:]['labels'].to('cpu').numpy()
    # true_labels = conllUtils.decode(true_encoded_labels, input_ids, offset_mapping, tokenizer)
    # result = conllUtils.compute_metric(pred_labels, true_labels)
    # print(result)
    # tags = conllUtils.predict(pred_labels)
    # conllUtils.save_result_zip(tags, model_name)

