# 初版
import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from transformers import AdamW, AutoTokenizer
from transformers import AutoModelForTokenClassification
from tqdm import tqdm
from utils.utilsbakbak import NERDataset
from utils import conllUtilsbakbak

if __name__ == '__main__':
    num_labels = 13
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    train_data = conllUtilsbakbak.encode(tokenizer, kind="train")
    valid_data = conllUtilsbakbak.encode(tokenizer, kind="valid")
    train_dataset = NERDataset(*train_data)
    print(train_dataset[0:2]['input_ids'])
    valid_dataset = NERDataset(*valid_data)
    # test_dataset = NERDataset(*test_data)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels).to(device)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    print(next(iter(train_loader)))
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
    # model.load_state_dict(torch.load('./models/roberta.pth', map_location=device))
    # with torch.no_grad():
    #     input_ids = valid_dataset[:]['input_ids'].to(device)
    #     attention_mask = valid_dataset[:]['attention_mask'].to(device)
    #     y_pred = np.argmax(model(input_ids, attention_mask).logits, axis=-
    #     1-trash)
    # tags = conllUtils.predict(model, tokenizer, y_pred, input_ids)
    # conllUtils.save_result_zip(tags)
