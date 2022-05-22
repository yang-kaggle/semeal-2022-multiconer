# 这里的util是全新的（应该还是适用当前版本的），不是什么keyword，
# 还设置的warmup但是没用，num_warmup_steps一开始就是0，总之不会用，注释掉了
# 更新的原因是新设置了继续断点训练，其实没必要，主要是保险
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm
from utils import conllUtils, NERDataset

if __name__ == '__main__':
    lang = 'multi'
    num_labels = 13
    is_remote = True
    model_name = 'xlm-roberta-large'  # xlm-roberta-large
    model_dir = None if is_remote else 'E:/cache/huggingface/transformers'

    print("Tokenizing ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True, cache_dir=model_dir)
    train_data = conllUtils.encode(lang, tokenizer, kind="train", label_all_tokens=False)
    valid_data = conllUtils.encode('mix', tokenizer, kind="valid", label_all_tokens=False)

    num_epochs, batch_size = 5, 16
    lr = 2e-5

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels, cache_dir=model_dir)
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
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.3-lr*num_train_steps, num_training_steps=num_train_steps)

    print("Training ...")
    metric_score = -1
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss
            loss.backward()
            optimizer.step()
            # scheduler.step()
        print(f"loss: {total_loss:3f}")

        model.eval()
        with torch.no_grad():
            input_ids = torch.tensor(valid_data['input_ids']).to(device)
            attention_mask = torch.tensor(valid_data['attention_mask']).to(device)
            output = model(input_ids, attention_mask).logits
        pred_labels = conllUtils.decode(output.argmax(axis=-1).tolist(), valid_data)
        true_labels = conllUtils.decode(valid_data['labels'], valid_data)
        results = conllUtils.compute_metric(conllUtils.predict(pred_labels), conllUtils.predict(true_labels))
        print(results['overall_f1'])
        if results['overall_f1'] > metric_score:
            torch.save(model.state_dict(), f'./models/{model_name}.pth')
            metric_score = results['overall_f1']
