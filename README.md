# 台北市政府服务机关问答系统

一基于HuggingFace的Transformer函数库调用，并以BERT模型进行分类任务。  

也验证并推测当使用简体中文训练出的BERT迁移至繁体中文的应用场景时可能会出现的问题。  

## 任务介绍

输入生活中常见的问题并预测，返回可以去哪个机关处理这个问题。  

## 环境

torch==1.5.1

transformers==3.0.2

## 架构说明

data/:存放数据

img/:存放README中用到的图片

pretrained_model/:预训练模型，BERT存放位置

src/:训练及加载的代码

result/:训练后的结果

## 运行说明

训练:src/train.py

应用:src/predict.py

## 结果

![image](https://github.com/sun830910/Taipei_QA_Matching/blob/master/img/result1.jpeg)

![image](https://github.com/sun830910/Taipei_QA_Matching/blob/master/img/result1.jpeg)

## 结果分析

训练30个epochs后，在训练集上的Acc约为0.948+，但在测试集上仅有约0.676+，推测原因有：

1. 由于使用的BERT预训练模型是基于简体中文语料进行训练的，所以迁移至繁体中文应用场景时有些token并不在预先学习的vocab之中，所以导致繁体中文的字可能多数是[UNK]的状态。
2. 由于多数token为unknown，导致模型随在训练集上fine tune了部分的token意思，但在对测试集上进行判别时，多数为未见过的字，也没有对tokenizer进行增强，导致在测试集上效果较差。