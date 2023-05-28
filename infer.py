import torch
import json
import copy
import numpy as np
import re as regex
from src.models.model import IRNet
from src.rule import semQL
from preprocess import utils
from src.utils import get_table_colNames, get_col_table_dict, load_word_emb
from src.dataset import Example
from src import args as arg
from src.rule.sem_utils import find_table, multi_equal, multi_option, random_choice
from sem2SQL import transform


def preprocess(question):

    # data process
    table_path = r'./data/db_schema.json'
    with open(table_path, 'r', encoding='utf-8') as inf:
        table_data = json.load(inf)
        table = table_data[0]

    data = {}
    data['query'] = ""
    data['question'] = question
    question_filter = utils.question_filter(question)
    data['question_toks'] = [tok for tok in question_filter]
    data['names'] = [col[1] for col in table['column_names']]
    data['table_names'] = table['table_names']
    tmp_col = []  # 中文列名集合
    for cc in [x[1] for x in table['column_names']]:
        if cc not in tmp_col:
            tmp_col.append(cc)
    data['col_set'] = tmp_col
    data['col_table'] = [col[0] for col in table['column_names']]
    data['origin_question_toks'] = data['question_toks']
    data['question_toks'] = utils.symbol_filter(data['question_toks'])

    keys = {}
    for kv in table['foreign_keys']:
        keys[kv[0]] = kv[1]
        keys[kv[1]] = kv[0]
    for id_k in table['primary_keys']:
        keys[id_k] = id_k
    data['keys'] = keys

    question_toks = data['question_toks']
    table_names = data['table_names']
    header_toks = data['col_set']  # 中文列名集合
    header_toks_list = []  # 将列名分割成字符之后的列表
    for y in data['col_set']:
        x = [tok for tok in y]
        header_toks_list.append(x)

    num_toks = len(question_toks)
    idx = 0
    tok_concol = []
    type_concol = []

    while idx < num_toks:

        # fully header(question中有n-gram出现在列名中)
        end_idx, header = utils.fully_part_header(question_toks, idx, num_toks, header_toks)
        if header:
            tok_concol.append(question_toks[idx: end_idx])
            type_concol.append(["列"])
            idx = end_idx
            continue

        # check for table(question中有n-gram出现在表名中)
        end_idx, tname = utils.group_header(question_toks, idx, num_toks, table_names)
        if tname:
            tok_concol.append(question_toks[idx: end_idx])
            type_concol.append(["表"])
            idx = end_idx
            continue

        # check for column
        end_idx, header = utils.group_header(question_toks, idx, num_toks, header_toks)
        if header:
            tok_concol.append(question_toks[idx: end_idx])
            type_concol.append(["列"])
            idx = end_idx
            continue

        # check for partial column(question中有n-gram部分匹配列名)
        end_idx, tname = utils.partial_header(question_toks, idx, header_toks_list)
        if tname:
            tok_concol.append(tname)
            type_concol.append(["列"])
            idx = end_idx
            continue
        
        # string match for Time Format
        if utils.num2year(question_toks[idx]):
            question_toks[idx] = '年份'
            end_idx, header = utils.group_header(question_toks, idx, num_toks, header_toks)
            question_toks[idx] = '年'
            if header:
                tok_concol.append(question_toks[idx: end_idx])
                type_concol.append(["列"])
                idx = end_idx
                continue

        result = utils.group_digital_alpha(question_toks, idx)
        if result is True:
            tok_concol.append(question_toks[idx: idx + 1])
            type_concol.append(["值"])
            idx += 1
            continue

        tok_concol.append([question_toks[idx]])
        type_concol.append(["无"])
        idx += 1
        continue

    data['question_arg'] = tok_concol
    data['question_arg_type'] = type_concol

    return data, table


def process(sql, table):

    process_dict = {}

    origin_sql = sql['question_toks']  # 句子分割成字符列表
    table_names = [[v for v in x] for x in table['table_names']]  # 所有表名分割成字符列表

    sql['pre_sql'] = copy.deepcopy(sql)

    tab_cols = [col[1] for col in table['column_names']]  # 每个表包含的列的列表
    tab_ids = [col[0] for col in table['column_names']]  # 列对应的表id

    col_set_iter = [[v for v in x] for x in sql['col_set']]  # col_set列名分割成字符列表
    col_iter = [[v for v in x] for x in tab_cols]  # 每个表包含的列分割成字符集合列表
    q_iter_small = [x for x in origin_sql]  # 句子分割成字符列表
    question_arg = copy.deepcopy(sql['question_arg'])  # [[], [], ...]
    question_arg_type = sql['question_arg_type']  # [[], [], [], ...]
    one_hot_type = np.zeros((len(question_arg_type), 3))  # question的3种匹配类型

    col_set_type = np.zeros((len(col_set_iter), 2))  # 列的2种匹配类型

    process_dict['col_set_iter'] = col_set_iter  # col_set列名分割成字符列表
    process_dict['q_iter_small'] = q_iter_small  # 句子分割成字符列表
    process_dict['col_set_type'] = col_set_type  # [len(col_set), 2]
    process_dict['question_arg'] = question_arg  # [[], [], ...]
    process_dict['question_arg_type'] = question_arg_type  # [[], [], [], ...]
    process_dict['one_hot_type'] = one_hot_type  # [len(question_arg_type), 3]
    process_dict['tab_cols'] = tab_cols  # 每个表包含的列集合
    process_dict['tab_ids'] = tab_ids  # 每个列对应的表id
    process_dict['col_iter'] = col_iter  # 每个表包含的列分割成字符集合列表
    process_dict['table_names'] = table_names  # 所有表名分割成字符列表

    return process_dict, sql


def schema_linking(question_arg, question_arg_type, one_hot_type, col_set_type, col_set_iter):

    for count_q, t_q in enumerate(question_arg_type):
        t = t_q[0]
        if t == '表':
            one_hot_type[count_q][0] = 1
            question_arg[count_q] = ['表'] + question_arg[count_q]
        elif t == '列':
            one_hot_type[count_q][1] = 1
            try:
                col_set_type[col_set_iter.index(question_arg[count_q])][1] = 5
                question_arg[count_q] = ['列'] + question_arg[count_q]
            except:
                print(col_set_iter, "".join(question_arg[count_q]))
                raise RuntimeError("not in col set")
        elif t == '值':
            one_hot_type[count_q][2] = 1
            question_arg[count_q] = ['值'] + question_arg[count_q]
        else:
            continue


def process_to_example(process_dict, sql):

    for c_id, col_ in enumerate(process_dict['col_set_iter']):  # 列名集合
        for q_id, ori in enumerate(process_dict['q_iter_small']):  # 句子分割成字符列表
            if ori in col_:
                process_dict['col_set_type'][c_id][0] += 1

    schema_linking(process_dict['question_arg'], process_dict['question_arg_type'],
                   process_dict['one_hot_type'], process_dict['col_set_type'], process_dict['col_set_iter'])

    col_table_dict = get_col_table_dict(process_dict['tab_cols'], process_dict['tab_ids'], sql)  # 每个列所属于的table_id
    table_col_name = get_table_colNames(process_dict['tab_ids'], process_dict['col_iter'])  # 每个table包含的列名字

    process_dict['col_set_iter'][0] = ['数', '数量', '很多']

    example = Example(
        src_sent=process_dict['question_arg'],  # 句子n-gram匹配列表
        col_num=len(process_dict['col_set_iter']),  # 列的个数
        vis_seq=(sql['question'], process_dict['col_set_iter'], sql['query']),  # (question, col_set列名分割成字符列表, SQL)
        tab_cols=process_dict['col_set_iter'],  # col_set列名分割成字符列表
        sql=sql['query'],  # SQL
        one_hot_type=process_dict['one_hot_type'],  # question的3种匹配类型
        col_hot_type=process_dict['col_set_type'],  # col_set的4种匹配类型
        table_names=process_dict['table_names'],  # 所有表名分割成字符列表
        table_len=len(process_dict['table_names']),  # 表的个数
        col_table_dict=col_table_dict,  # 每个列所属于的table_id
        cols=process_dict['tab_cols'],  # 每个表包含的列集合
        table_col_name=table_col_name,  # 每个table包含的列名字
        table_col_len=len(table_col_name),  # 表的个数
        tokenized_src_sent=process_dict['col_set_type'],  # col_set的4种匹配类型
        tgt_actions=None  # IR序列
    )
    example.sql_json = copy.deepcopy(sql)

    return example


def model_init(args):
    grammar = semQL.Grammar()
    model = IRNet(args, grammar)

    if args.cuda:
        model.cuda()

    pretrained_model_path = r'./saved_model/saved_model1673318065/best_model.model'
    pretrained_model = torch.load(pretrained_model_path, map_location="cpu")
    import copy
    pretrained_modeled = copy.deepcopy(pretrained_model)
    for k in pretrained_model.keys():
        if k not in model.state_dict().keys():
            del pretrained_modeled[k]

    model.load_state_dict(pretrained_modeled)

    glove_embed_path = r'./pretrained_models/sgns.baidubaike.bigram-char'
    model.word_emb = load_word_emb(glove_embed_path)

    return model


def infer(example, model):

    results = model.parse(example, beam_size=5)
    results = results[0]
    try:
        pred = " ".join([str(x) for x in results[0].actions])
    except Exception:
        pred = ""

    simple_json = example.sql_json['pre_sql']
    simple_json['model_result'] = pred

    return simple_json


def alter_not_in(data, table):
    if 'Filter(19)' in data['model_result']:
        current_table = table
        current_table['schema_content_clean'] = [x[1] for x in current_table['column_names']]
        current_table['col_table'] = [col[0] for col in current_table['column_names']]
        origin_table_names = [[x for x in names] for names in data['table_names']]
        question_arg_type = data['question_arg_type']
        question_arg = data['question_arg']
        pred_label = data['model_result'].split(' ')

        # get potiantial table
        cur_table = None
        for label_id, label_val in enumerate(pred_label):
            if label_val in ['Filter(19)']:
                cur_table = int(pred_label[label_id - 1][2:-1])
                break

        h_table = find_table(cur_table, origin_table_names, question_arg_type, question_arg)

        for label_id, label_val in enumerate(pred_label):
            if label_val in ['Filter(19)']:
                for primary in current_table['primary_keys']:
                    if int(current_table['col_table'][primary]) == int(pred_label[label_id - 1][2:-1]):
                        pred_label[label_id + 2] = 'C(' + str(
                            data['col_set'].index(current_table['schema_content_clean'][primary])) + ')'
                        break
                for pair in current_table['foreign_keys']:
                    if int(current_table['col_table'][pair[0]]) == h_table and data['col_set'].index(
                            current_table['schema_content_clean'][pair[1]]) == int(pred_label[label_id + 2][2:-1]):
                        pred_label[label_id + 8] = 'C(' + str(
                            data['col_set'].index(current_table['schema_content_clean'][pair[0]])) + ')'
                        pred_label[label_id + 9] = 'T(' + str(h_table) + ')'
                        break
                    elif int(current_table['col_table'][pair[1]]) == h_table and data['col_set'].index(
                            current_table['schema_content_clean'][pair[0]]) == int(pred_label[label_id + 2][2:-1]):
                        pred_label[label_id + 8] = 'C(' + str(
                            data['col_set'].index(current_table['schema_content_clean'][pair[1]])) + ')'
                        pred_label[label_id + 9] = 'T(' + str(h_table) + ')'
                        break
                pred_label[label_id + 3] = pred_label[label_id - 1]

        data['model_result'] = " ".join(pred_label)


def alter_inter(data):
    if 'Filter(0)' in data['model_result']:
        now_result = data['model_result'].split(' ')
        index = now_result.index('Filter(0)')
        c1 = None
        c2 = None
        for i in range(index + 1, len(now_result)):
            if c1 is None and 'C(' in now_result[i]:
                c1 = now_result[i]
            elif c1 is not None and c2 is None and 'C(' in now_result[i]:
                c2 = now_result[i]

        replace_result = ['Root1(0)'] + now_result[1:now_result.index('Filter(0)')]
        for r_id, r_val in enumerate(now_result[now_result.index('Filter(0)') + 2:]):
            if 'Filter' in r_val:
                break

        replace_result = replace_result + now_result[now_result.index('Filter(0)') + 1:r_id + now_result.index(
            'Filter(0)') + 2]
        replace_result = replace_result + now_result[1:now_result.index('Filter(0)')]

        replace_result = replace_result + now_result[r_id + now_result.index('Filter(0)') + 2:]
        replace_result = " ".join(replace_result)
        data['model_result'] = replace_result


def alter_column0(data):
    zero_count = 0
    count = 0
    result = []
    if 'C(0)' in data['model_result']:
        pattern = regex.compile('C\(.*?\) T\(.*?\)')
        result_pattern = list(set(pattern.findall(data['model_result'])))
        ground_col_labels = []
        for pa in result_pattern:
            pa = pa.split(' ')
            if pa[0] != 'C(0)':
                index = int(pa[1][2:-1])
                ground_col_labels.append(index)

        ground_col_labels = list(set(ground_col_labels))
        question_arg_type = data['question_arg_type']
        question_arg = data['question_arg']
        table_names = [[name for name in names] for names in data['table_names']]
        origin_table_names = [[x for x in names] for names in data['table_names']]
        count += 1
        easy_flag = False

        for q_ind, q in enumerate(data['question_arg']):
            q_str = "".join("".join(x) for x in data['question_arg'])
            if '有多少' in q_str or '几' in q_str or '数量' in q_str:
                easy_flag = True
        if easy_flag:
            # check for the last one is a table word
            for q_ind, q in enumerate(data['question_arg']):
                if (q_ind > 0 and q == ['少'] and data['question_arg'][q_ind - 1] == ['多']) or (
                        q_ind > 0 and q == ['量'] and data['question_arg'][q_ind - 1] == ['数']) or (
                        q_ind > 0 and q == ['几']):
                    re = multi_equal(question_arg_type, q_ind, ['表'], 2)
                    if re is not False:
                        # This step work for the number of [table] example
                        table_result = table_names[origin_table_names.index(question_arg[re])]
                        result.append((data['query'], data['question'], table_result, data))  # table_result: table_name分割成字符列表
                        break
                    else:
                        re = multi_option(question_arg, q_ind, data['table_names'], 2)
                        if re is not False:
                            table_result = re
                            result.append((data['query'], data['question'], table_result, data))
                            pass
                        else:
                            re = multi_equal(question_arg_type, q_ind, ['表'], len(question_arg_type))
                            if re is not False:
                                # This step work for the number of [table] example
                                table_result = table_names[origin_table_names.index(question_arg[re])]
                                result.append((data['query'], data['question'], table_result, data))
                                break
                            pass
                        table_result = random_choice(question_arg=question_arg,
                                                     question_arg_type=question_arg_type,
                                                     names=table_names,
                                                     ground_col_labels=ground_col_labels, q_ind=q_ind, N=2,
                                                     origin_name=origin_table_names)
                        result.append((data['query'], data['question'], table_result, data))

                        zero_count += 1
                    break
        else:
            M_OP = False
            for q_ind, q in enumerate(data['question_arg']):
                if M_OP is False and q in [['比'], ['至', '少'], ['最', '多'], ['最', '少']] or \
                        question_arg_type[q_ind] == ['M_OP']:
                    M_OP = True
                    re = multi_equal(question_arg_type, q_ind, ['表'], 3)
                    if re is not False:
                        # This step work for the number of [table] example
                        table_result = table_names[origin_table_names.index(question_arg[re])]
                        result.append((data['query'], data['question'], table_result, data))
                        break
                    else:
                        re = multi_option(question_arg, q_ind, data['table_names'], 3)
                        if re is not False:
                            table_result = re
                            #                             print(table_result)
                            result.append((data['query'], data['question'], table_result, data))
                            pass
                        else:
                            #                             zero_count += 1
                            re = multi_equal(question_arg_type, q_ind, ['表'], len(question_arg_type))
                            if re is not False:
                                # This step work for the number of [table] example
                                table_result = table_names[origin_table_names.index(question_arg[re])]
                                result.append((data['query'], data['question'], table_result, data))
                                break
                            table_result = random_choice(question_arg=question_arg,
                                                         question_arg_type=question_arg_type,
                                                         names=table_names,
                                                         ground_col_labels=ground_col_labels, q_ind=q_ind, N=2,
                                                         origin_name=origin_table_names)
                            result.append((data['query'], data['question'], table_result, data))
                            pass
            if M_OP is False:
                table_result = random_choice(question_arg=question_arg,
                                             question_arg_type=question_arg_type,
                                             names=table_names, ground_col_labels=ground_col_labels, q_ind=q_ind,
                                             N=2,
                                             origin_name=origin_table_names)
                result.append((data['query'], data['question'], table_result, data))

    for re in result:
        table_names = [[x for x in names] for names in re[3]['table_names']]
        origin_table_names = [[x for x in names] for names in re[3]['table_names']]
        if re[2] in table_names:
            re[3]['rule_count'] = table_names.index(re[2])
        else:
            re[3]['rule_count'] = origin_table_names.index(re[2])

    if 'rule_count' in data:
        str_replace = 'C(0) T(' + str(data['rule_count']) + ')'
        replace_result = regex.sub('C\(0\) T\(.\)', str_replace, data['model_result'])
        data['model_result_replace'] = replace_result
    else:
        data['model_result_replace'] = data['model_result']


def sem2SQL(data, table):
    alter_not_in(data, table=table)
    alter_inter(data)
    alter_column0(data)
    result = transform(data, table)
    return result[0]


def main():
    # initialize model
    arg_parser = arg.init_arg_parser()
    args = arg.init_config(arg_parser)
    model = model_init(args)

    while True:
        question = input("请输入你要查询的问题：")
        if question == "exit" or question == "stop":
            break
        data, table = preprocess(question)
        process_dict, sql = process(data, table)
        example = process_to_example(process_dict, sql)
        print("生成的SQL查询语句为：")
        simple_json = infer(example, model)
        pred_sql = sem2SQL(simple_json, table)
        print(pred_sql)


if __name__ == '__main__':
    main()
