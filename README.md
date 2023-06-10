# IRNet-CSgSQL

将IRNet应用于电网领域的Text-to-SQL数据集中，对其他中文领域数据集也适用。同时，本项目实现了问答demo。

## 环境配置

1.创建虚拟环境：

* `Python3.6`
* `Pytorch 0.4.0`

安装环境通过执行`pip install -r requirements.txt`命令。

2.下载[中文Glove](https://github.com/Embedding/Chinese-Word-Vectors)词向量到`./pretrained_models/`目录中，本项目使用了基于百度百科数据训练的Glove词向量。

## 数据集

CSgSQL数据集和[Spider](https://yale-lily.github.io/spider)数据集结构基本相同，具体包含`train.json`，`dev.json`以及`db_schema.json`三个文件。

1.`train.json`和`dev.json`对应于Spider数据集的训练集和验证集，只不过没有像Spider中的`"query_toks"`，`"query_toks_no_value"`和`"question_toks"`三个键，对于NL和SQL可以自己进行分词，并且将SQL中的数值用value单词替换。

2.`db_schema.json`对应于Spider数据集的`tables.json`文件，其中`"table_names_original"`对应于表的英文名字，`"table_names"`对应于表的中文名字，`"column_names_original"`和`"column_names"`也是类似。

## 预处理

将数据集放在`./data/`目录中，然后按照`./preprocess/`目录中的命令预处理训练集和验证集。

## 训练

运行`train.sh`训练IRNet。

    sh train.sh [GPU_ID] [SAVE_FOLD]

## 测试

运行`eval.sh`评估IRNet。

    sh eval.sh [GPU_ID] [OUTPUT_FOLD]

## 结果

| model | dev acc |
| :---: | :---: |
| IRNet + Glove | 80.3 |

## 部署

运行`infer.py`文件并将文件中相应的模型路径修改即可。

    python infer.py