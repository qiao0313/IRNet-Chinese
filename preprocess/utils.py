import os
import json
from nltk.tokenize import word_tokenize
import jieba
import re

# VALUE_FILTER = ['what', 'how', 'list', 'give', 'show', 'find', 'id', 'order', 'when']
# AGG = ['average', 'sum', 'max', 'min', 'minimum', 'maximum', 'between']


def load_dataSets(args):
    with open(args.table_path, 'r', encoding='UTF-8') as f:
        table_datas = json.load(f)
    with open(args.data_path, 'r', encoding='UTF-8') as f:
        datas = json.load(f)

    output_tab = {}
    tables = {}
    tabel_name = set()  # db_id 集合
    for i in range(len(table_datas)):
        table = table_datas[i]
        temp = {}
        temp['col_map'] = table['column_names']  # (table_id, 中文列名)
        temp['table_names'] = table['table_names']  # 中文表名集合
        tmp_col = []  # 中文列名集合
        for cc in [x[1] for x in table['column_names']]:
            if cc not in tmp_col:
                tmp_col.append(cc)
        table['col_set'] = tmp_col  # 中文列名集合
        db_name = table['db_id']
        tabel_name.add(db_name)
        table['schema_content'] = [col[1] for col in table['column_names']]  # 中文列名集合
        table['col_table'] = [col[0] for col in table['column_names']]  # 每一个col对应的table_id
        output_tab[db_name] = temp  # (table_id, 中文列名), 中文表名集合
        tables[db_name] = table  # col_set: 中文列名集合; schema_content: 中文列名集合; col_table: 每一列对应的table_id

    for d in datas:
        d['names'] = tables[d['db_id']]['schema_content']  # 中文列名集合
        d['table_names'] = tables[d['db_id']]['table_names']  # 中文表名集合
        d['col_set'] = tables[d['db_id']]['col_set']  # 中文列名集合
        d['col_table'] = tables[d['db_id']]['col_table']  # 每一个col对应的table_id
        question = d['question']
        new_question = question_filter(question)  # 去掉question中的标点符号
        d['question_toks'] = [tok for tok in new_question]  # question字符列表
        # query = d['query']
        # query_toks = word_tokenize(query)
        # q = []
        # for tok in query_toks:
        #     if tok != ' ':
        #         tok = filter_num(tok)
        #         q.append(tok)
        # d['query_toks_no_value'] = q
        keys = {}
        for kv in tables[d['db_id']]['foreign_keys']:
            keys[kv[0]] = kv[1]
            keys[kv[1]] = kv[0]
        for id_k in tables[d['db_id']]['primary_keys']:
            keys[id_k] = id_k
        d['keys'] = keys  # 主键与外键对应关系
    return datas, tables


def group_header(toks, idx, num_toks, header_toks):
    for endIdx in reversed(range(idx + 1, num_toks+1)):
        sub_toks = toks[idx: endIdx]
        sub_toks = "".join(sub_toks)
        if sub_toks in header_toks:
            return endIdx, sub_toks
    return idx, None


def fully_part_header(toks, idx, num_toks, header_toks):
    for endIdx in reversed(range(idx + 1, num_toks+1)):
        sub_toks = toks[idx: endIdx]
        if len(sub_toks) > 1:
            sub_toks = "".join(sub_toks)
            if sub_toks in header_toks:
                return endIdx, sub_toks
    return idx, None


def partial_header(toks, idx, header_toks):
    def check_in(list_one, list_two):
        if len(set(list_one) & set(list_two)) == len(list_one) and (len(list_two) <= 7):
            return True
    for endIdx in reversed(range(idx + 1, len(toks))):
        sub_toks = toks[idx: min(endIdx, len(toks))]
        if len(sub_toks) > 1:
            flag_count = 0
            tmp_heads = None
            for heads in header_toks:
                if check_in(sub_toks, heads):
                    flag_count += 1
                    tmp_heads = heads
            if flag_count == 1:
                return endIdx, tmp_heads
    return idx, None


def question_filter(question):
    # symbol = ["，", "、", "。", "！", "？", "：", ".", ":"]
    new_question = question.replace("，", '').replace("、", '').replace("。", '').replace("！", '').replace("？", '').\
        replace("：", '').replace(".", '').replace(":", '').replace(',', '').replace('!', '')
    return new_question


def symbol_filter(questions):
    question_tmp_q = []
    for q_id, q_val in enumerate(questions):
        if len(q_val) > 2 and q_val[0] in ["'", '"', '`', '鈥�', '鈥�'] and q_val[-1] in ["'", '"', '`', '鈥�']:
            question_tmp_q.append("'")
            question_tmp_q += ["".join(q_val[1:-1])]
            question_tmp_q.append("'")
        elif len(q_val) > 2 and q_val[0] in ["'", '"', '`', '鈥�']:
            question_tmp_q.append("'")
            question_tmp_q += ["".join(q_val[1:])]
        elif len(q_val) > 2 and q_val[-1] in ["'", '"', '`', '鈥�']:
            question_tmp_q += ["".join(q_val[0:-1])]
            question_tmp_q.append("'")
        elif q_val in ["'", '"', '`', '鈥�', '鈥�', '``', "''"]:
            question_tmp_q += ["'"]
        else:
            question_tmp_q += [q_val]
    return question_tmp_q


def group_digital_alpha(toks, idx):
    value = re.compile(r"[a-zA-Z0-9]")
    test = toks[idx]
    result = value.match(test)
    if result:
        return True
    else:
        return False


def filter_num(tok):
    value = re.compile(r"[0-9]+")
    has_num = value.match(tok)
    if has_num:
        return 'value'
    else:
        return tok


def group_symbol(toks, idx, num_toks):
    if toks[idx-1] == "'":
        for i in range(0, min(3, num_toks-idx)):
            if toks[i + idx] == "'":
                return i + idx, toks[idx:i+idx]
    return idx, None


def num2year(tok):
    if len(str(tok)) == 4 and str(tok).isdigit() and int(str(tok)[:2]) < 22 and int(str(tok)[:2]) > 15:
        return True
    return False


def set_header(toks, header_toks, tok_concol, idx, num_toks):
    def check_in(list_one, list_two):
        if set(list_one) == set(list_two):
            return True
    for endIdx in range(idx, num_toks):
        toks += tok_concol[endIdx]
        if len(tok_concol[endIdx]) > 1:
            break
        for heads in header_toks:
            if check_in(toks, heads):
                return heads
    return None


if __name__ == '__main__':
    question = "统计青海省已退运的变电站数量，。，。！：、,."
    new_question = question_filter(question)
    print([tok for tok in new_question])
