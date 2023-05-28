import json
import argparse
import nltk
import os
import pickle
import jieba
from utils import symbol_filter, fully_part_header, group_header, partial_header, num2year, group_symbol, group_digital_alpha
# from utils import AGG
from utils import load_dataSets


def process_datas(datas):
    """
    :param datas:
    :param args:
    :return:
    """
    for entry in datas:
        entry['question_toks'] = symbol_filter(entry['question_toks'])
        question_toks = entry['question_toks']  # 句子字符列表

        table_names = []
        for y in entry['table_names']:  # 中文表名集合
            table_names.append(y)

        header_toks = []  # 中文列名集合
        header_toks_list = []  # 将列名分割成字符之后的列表

        for y in entry['col_set']:
            header_toks.append(y)  # 中文列名集合
            x = [tok for tok in y]
            header_toks_list.append(x)  # 将列名分割成字符之后的列表

        num_toks = len(question_toks)  # question字符个数
        idx = 0
        tok_concol = []
        type_concol = []

        while idx < num_toks:
            
            # fully header(question中有n-gram出现在列名中)
            end_idx, header = fully_part_header(question_toks, idx, num_toks, header_toks)
            if header:
                tok_concol.append(question_toks[idx: end_idx])
                type_concol.append(["列"])
                idx = end_idx
                continue

            # check for table(question中有n-gram出现在表名中)
            end_idx, tname = group_header(question_toks, idx, num_toks, table_names)
            if tname:
                tok_concol.append(question_toks[idx: end_idx])
                type_concol.append(["表"])
                idx = end_idx
                continue

            # check for column
            end_idx, header = group_header(question_toks, idx, num_toks, header_toks)
            if header:
                tok_concol.append(question_toks[idx: end_idx])
                type_concol.append(["列"])
                idx = end_idx
                continue

            # check for partial column(question中有n-gram部分匹配列名)
            end_idx, tname = partial_header(question_toks, idx, header_toks_list)
            if tname:
                tok_concol.append(tname)
                type_concol.append(["列"])
                idx = end_idx
                continue

            # string match for Time Format
            if num2year(question_toks[idx]):
                question_toks[idx] = '年份'
                end_idx, header = group_header(question_toks, idx, num_toks, header_toks)
                question_toks[idx] = '年'
                if header:
                    tok_concol.append(question_toks[idx: end_idx])
                    type_concol.append(["列"])
                    idx = end_idx
                    continue

            result = group_digital_alpha(question_toks, idx)
            if result is True:
                tok_concol.append(question_toks[idx: idx + 1])
                type_concol.append(["值"])
                idx += 1
                continue

            tok_concol.append([question_toks[idx]])
            type_concol.append(["无"])
            idx += 1
            continue

        entry['question_arg'] = tok_concol
        entry['question_arg_type'] = type_concol

    return datas


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, help='dataset', required=True)
    arg_parser.add_argument('--table_path', type=str, help='table dataset', required=True)
    arg_parser.add_argument('--output', type=str, help='output data')
    args = arg_parser.parse_args()

    # loading dataSets
    datas, table = load_dataSets(args)

    # process datasets
    process_result = process_datas(datas)

    with open(args.output, 'w', encoding='UTF-8') as f:
        json.dump(datas, f, ensure_ascii=False)
