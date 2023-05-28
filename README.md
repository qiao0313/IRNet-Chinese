# IRNet-CSgSQL

将IRNet应用于电网领域的Text-to-SQL数据集中，对其他中文领域数据集也适用。

## 环境配置

1.创建虚拟环境：

* `Python3.6`
* `Pytorch 0.4.0`

安装环境通过执行`pip install -r requirements.txt`命令。

2.下载[中文Glove](https://github.com/Embedding/Chinese-Word-Vectors)词向量到`./pretrained_models/`目录中，本项目使用了基于百度百科数据训练的Glove词向量。

## 数据集以及预处理

将数据集放在`./data/`目录中，然后按照`./preprocess/`目录中的命令预处理训练集和验证集。

## 训练

运行`train.sh`训练IRNet。

`sh train.sh [GPU_ID] [SAVE_FOLD]`

## 测试

运行`eval.sh`评估IRNet。

`sh eval.sh [GPU_ID] [OUTPUT_FOLD]`

## 结果

| model | dev acc |
| :---: | :---: |
| IRNet + GLOVE | 83.0 |
