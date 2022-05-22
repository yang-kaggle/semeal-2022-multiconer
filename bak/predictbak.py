# 这里有什么问题呢
# 你看Multi数据集验证集有多大，一次性全放进模型就内存不够了
# 所以呢，只能构建dataset分批次计算
# 话说，其实Mutli就是所有语言的打包在一起，然后打乱顺序再分成两个集合
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from utils import conllUtils

if __name__ == '__main__':
    lang = 'en'
    num_labels = 13
    is_remote = False
    model_name = 'bert-base-multilingual-cased'
    model_dir = None if is_remote else 'E:/homeishere/huggingface/transformers'

    print("Preprocessing ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True, cache_dir=model_dir)
    valid_data = conllUtils.encode(lang, tokenizer, kind="valid", label_all_tokens=False)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels,
                                                            cache_dir=model_dir).to(device)

    model.eval()
    model.load_state_dict(torch.load(f'./models/{model_name}.pth', map_location=device))

    print("Predicting ...")
    with torch.no_grad():
        input_ids = torch.tensor(valid_data['input_ids']).to(device)
        attention_mask = torch.tensor(valid_data['attention_mask']).to(device)
        output = model(input_ids, attention_mask).logits

    print("Metricating ...")
    pred_labels = conllUtils.decode(output.argmax(axis=-1).tolist(), valid_data)
    true_labels = conllUtils.decode(valid_data['labels'], valid_data)
    pred_tags = conllUtils.predict(pred_labels)
    true_tags = conllUtils.predict(true_labels)
    result = conllUtils.compute_metric(pred_tags, true_tags)
    print(result['overall_f1'])
    # conllUtils.save_result_zip(pred_tags, 'en', model_name+'tt')
