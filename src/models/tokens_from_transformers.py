from transformers import BertTokenizer, BertModel
import numpy as np
import torch
import os


def transformers_tokens(one_q, is_list=True):

    model_name = '../../pretrained_models/bert-base-multilingual-cased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    q_val = []

    if is_list:
        lst_question = ["".join(q_tok) for q_tok in one_q]
        question = "".join(lst_question)
        token = ['[CLS]'] + [x for x in question]
        token_ids = tokenizer.convert_tokens_to_ids(token)

        # input = tokenizer(question, padding=True, truncation=True, max_length=512, return_tensors="pt")
        input = torch.tensor([token_ids])
        output = model(input)
        tokens = output[0]
        post_tokens = tokens.squeeze(0).detach().numpy()

        question_arg_emb = []

        question_len = 0
        for tok_idx in range(len(one_q)):
            tok_len = len("".join(one_q[tok_idx]))
            question_arg_emb.append(post_tokens[question_len + 1:question_len + tok_len + 1])
            if tok_len == 1:
                q_val.append(question_arg_emb[tok_idx])
            else:
                q_val.append(sum(question_arg_emb[tok_idx]) / float(tok_len))
            question_len += tok_len
    else:
        question = "".join(one_q)
        token = ['[CLS]'] + [x for x in question]
        token_ids = tokenizer.convert_tokens_to_ids(token)

        # input = tokenizer(question, padding=True, truncation=True, max_length=512, return_tensors="pt")
        input = torch.tensor([token_ids])
        output = model(input)
        tokens = output[0]
        post_tokens = tokens.squeeze(0).detach().numpy()

        question_arg_emb = []

        question_len = 0
        for tok_idx in range(len(one_q)):
            tok_len = len(one_q[tok_idx])
            question_arg_emb.append(post_tokens[question_len + 1:question_len + tok_len + 1])
            if tok_len == 1:
                q_val.append(question_arg_emb[tok_idx])
            else:
                q_val.append(sum(question_arg_emb[tok_idx]) / float(tok_len))
            question_len += tok_len

    return q_val


if __name__ == '__main__':
    one_q = [['值', '我'], ['无', '是'], ['列', '谁']]
    q_val = transformers_tokens(one_q, is_list=True)
    print(q_val)
    val_embs = []
    val_embs.append(q_val)
    B = 1
    val_emb_array = np.zeros((B, len(q_val), 768), dtype=np.float32)
    for i in range(B):
        for t in range(len(val_embs[i])):
            val_emb_array[i, t, :] = val_embs[i][t]
    val_inp = torch.from_numpy(val_emb_array)
    print(val_inp.shape)
    print(val_inp)