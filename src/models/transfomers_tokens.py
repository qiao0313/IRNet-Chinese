from transformers import BertTokenizer, BertModel
import torch


model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
#
# # question_toks = [['表', '我'], ['列', '是'], ['谁']]
# # lst_question = ["".join(q_tok) for q_tok in question_toks]
# # question = ["".join(lst_question)]
# # print(question)
# question_toks = ['表', '我', '列', '是', '谁']
#
# input = tokenizer(question_toks, padding=True, truncation=True, max_length=512, return_tensors="pt")
# output = model(input["input_ids"])
# print(input["input_ids"].shape)
# print(input["input_ids"])
# tokens = output[0]
# print(tokens.shape)

question = "320kV直流线路名称输电平均值及其终点厂站最高电压等级"
print(len(question))
# token = tokenizer.tokenize(question)
token = ['[CLS]'] + [x for x in question]
print(token)
token_ids = tokenizer.convert_tokens_to_ids(token)
print(token_ids)
output = model(torch.tensor([token_ids]))[0]
print(output.shape)