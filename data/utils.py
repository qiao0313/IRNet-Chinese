import json


train_path = 'train_new.json'
new_train_path = 'train_new1.json'
dev_path = 'dev_new.json'
new_dev_path = 'dev_new1.json'

with open(train_path, 'r', encoding='utf8') as file:
    datas = json.load(file)

    new_datas = []
    for d in datas:
        if d['question_arg'] == []:
            continue
        new_datas.append(d)

    with open(new_train_path, 'w', encoding='utf8') as file1:
        json.dump(new_datas, file1, ensure_ascii=False)

with open(dev_path, 'r', encoding='utf8') as file:
    datas = json.load(file)

    new_datas = []
    for d in datas:
        if d['question_arg'] == []:
            continue
        new_datas.append(d)

    with open(new_dev_path, 'w', encoding='utf8') as file1:
        json.dump(new_datas, file1, ensure_ascii=False)
