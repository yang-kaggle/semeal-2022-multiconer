import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import logging
from tqdm import tqdm
from model import NERModel
from utils import conllUtils, NERDataset

if __name__ == '__main__':
    lang = 'multi'
    num_labels = 13
    is_remote = True  # 是否运行在远程终端，涉及到cache位置
    is_first = True  # 是否为第一次训练，否则继续训练
    # xlm-roberta-large bert-base-multilingual-cased distilbert-base-multilingual-cased
    model_name = 'xlm-roberta-large'
    model_dir = None if is_remote else 'E:/cache/huggingface/transformers'
    logging.set_verbosity_error()

    print("Tokenizing ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True, cache_dir=model_dir)
    train_data = conllUtils.encode(lang, tokenizer, kind="train", label_all_tokens=False)
    valid_data = conllUtils.encode('mix', tokenizer, kind="valid", label_all_tokens=False)

    num_epochs, batch_size = 5, 16
    lr = 2e-5

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = NERModel(model_name, num_labels=num_labels, cache_dir=model_dir, is_freeze=False)
    if not is_first:
        model.load_state_dict(torch.load(f'./models/{model_name}.pth', map_location=torch.device('cpu')))
    train_dataset = NERDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']  # 下列参数 不进行正则化（权重衰减）
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_parameters, lr=lr)
    num_train_steps = int(len(valid_data['labels']) / batch_size * num_epochs)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

    print("Training ...")
    metric_score = 62.0
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            _, loss = model(input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += loss
            loss.backward()
            optimizer.step()
            # scheduler.step()
        print(f"loss: {total_loss:3f}")

        model.eval()
        with torch.no_grad():
            input_ids = torch.tensor(valid_data['input_ids']).to(device)
            attention_mask = torch.tensor(valid_data['attention_mask']).to(device)
            output, _ = model(input_ids, attention_mask)

        pred_labels = conllUtils.decode(output.argmax(axis=-1).tolist(), valid_data)
        true_labels = conllUtils.decode(valid_data['labels'], valid_data)
        results = conllUtils.compute_metric(conllUtils.predict(pred_labels), conllUtils.predict(true_labels))
        print(results['overall_f1'])
        if results['overall_f1'] > metric_score:
            torch.save(model.state_dict(), f'./models/{model_name}.pth')
            metric_score = results['overall_f1']
