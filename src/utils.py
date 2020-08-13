# -*- coding: utf-8 -*-

"""
Created on 2020-08-13 09:05
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com

数据读取、模型加载、转存函数
"""

import torch
from torch.utils.data import TensorDataset
import pickle
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer


def read_data(data_path):
    """
    读取数据(.txt)
    :param data_path:数据路径
    :return: (List)问题、(List)答案
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.read()
    qa_pairs = data.split('\n')

    answers = []
    questions = []
    for qa_pair in qa_pairs:
        tmp = qa_pair.split()
        if len(tmp) == 2:
            answers.append(tmp[0])
            questions.append(tmp[1])
        else:
            print("样本格式不合规:{}".format(qa_pair))

    print("数据中共有 {} 个办公单位(labels)".format(len(set(answers))))

    assert len(questions) == len(answers)
    return questions, answers


def load_bert(model_path, config_path, vocab_path, num_labels):
    """
    实例化bert模型相关对象
    :param model_path: bert model 路径
    :param config_path: bert config 路径
    :param vocab_path: bert vocab 路径
    :param num_labels: 分类类别数(本语料为149)
    :return: 实例后的bert模型与tokernizer
    """
    config, model, tokenizer = (BertConfig, BertForSequenceClassification, BertTokenizer)
    config = config.from_pretrained(config_path, num_labels=num_labels)
    model = model.from_pretrained(model_path, config=config)
    tokenizer = tokenizer(vocab_file=vocab_path)
    return model, tokenizer


def compute_acc(y_pred, y_target):
    """
    计算准确率
    :param y_pred:模型预测结果
    :param y_target:
    :return:
    """
    _, y_pred_idx = y_pred.max(dim=1)
    num_correct = torch.eq(y_pred_idx, y_target).sum().item()
    return num_correct/len(y_pred_idx)


def make_torch_input(input_idxs, input_masks, input_segment_idxs, labels):
    """
    将数据转换为pytorch输入格式
    :param input_idxs: 语料tokens
    :param input_masks: 语料masks
    :param input_segment_idxs: 语料分词结果
    :param labels: 语料标签
    :return: pytorch格式
    """
    all_inputs_idxs = torch.tensor([input_idx for input_idx in input_idxs], dtype=torch.long)
    all_inputs_masks = torch.tensor([input_mask for input_mask in input_masks], dtype=torch.long)
    all_input_segment_idxs = torch.tensor([input_segment_idx for input_segment_idx in input_segment_idxs], dtype=torch.long)
    all_input_labels = torch.tensor([label for label in labels], dtype=torch.long)
    return TensorDataset(all_inputs_idxs, all_inputs_masks, all_input_segment_idxs, all_input_labels)


def split_dataset(dataset, split_ratio=0.8):
    """
    分割数据集
    :param dataset:数据集
    :param split_ratio: 分割比例，预设0.8
    :return: 训练集与测试集
    """
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    trainingSet, testingSet = torch.utils.data.random_split(dataset=dataset, lengths=[train_size, test_size])
    return trainingSet, testingSet


class DataFormat(object):
    """
    数据加载对象，主要可以对类别进行统筹，了解数据状况，也规范一些
    """
    def __init__(self, input_data):
        self.input_data = input_data
        self.unique_data = list(set(self.input_data))
        self.data_kinds = len(self.unique_data)
        self.data_list = []
        self._build_dict()

    def _build_dict(self):
        for data_idx, data in enumerate(self.unique_data):
            if data is not None:
                self.data_list.append((data_idx, data))

    def data_to_idx(self, text):
        for data_idx, data in self.data_list:
            if text == data:
                return data_idx

    def idx_to_data(self, idx):
        if idx in self.data_list:
            return self.input_data[idx]

    @property
    def data_length(self):
        return self.data_kinds

    @property
    def data(self):
        return self.input_data

    def __len__(self):
        return len(self.data)


def sentence_to_idx(tokenizer, sentence):
    """
    调用tokenizer将句子转换成tokens形式
    :param tokenizer: 实体化后的tokenizer
    :param sentence: 输入句子
    :return: 转换后的tokens
    """
    return tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence)))


def get_embedding(tokenizer, data_path):
    """
    将输入语料转换为bert预设的格式
    :param tokenizer: bert预设的tokenizer
    :param data_path: 语料路径
    :return: 数据特征
    """
    questions, answers = read_data(data_path)
    questions_dict = DataFormat(questions)
    answers_dict = DataFormat(answers)

    questions_tokens = []
    questions_max_length = 0

    for question in questions_dict.data:
        tokens = sentence_to_idx(tokenizer=tokenizer, sentence=question)
        if 512 > len(tokens) > questions_max_length:  # 最大tokens长度应小于预训练模型的预设最大长度，如BERT为512
            questions_max_length = len(tokens)
        questions_tokens.append(tokens)

    print("问题最长长度为: {} ".format(questions_max_length))
    assert questions_max_length <= 512  # 最大tokens长度应小于预训练模型的预设最大长度，如BERT为512

    # padding
    for tokens in questions_tokens:
        while len(tokens) < questions_max_length:
            tokens.append(0)

    labels = []
    for answer in answers_dict.data:
        labels.append(answers_dict.data_to_idx(answer))

    assert len(questions_tokens) == len(labels)

    input_masks = [[1]*questions_max_length for _ in range(len(questions_dict))]
    input_segment_idx = [[0]*questions_max_length for _ in range(len(questions_dict))]

    # 打成字典一次return一个变量就行
    data_features = {
        'input_idx': questions_tokens,
        'input_masks': input_masks,
        'input_segment_idx': input_segment_idx,
        'answer_labels': labels,
        'question_dict': questions_dict,
        'answer_dict': answers_dict
    }

    # 保存data的特征
    output = open('../result/data_features.pkl', 'wb')
    pickle.dump(data_features, output)
    return data_features


if __name__ == '__main__':
    data_path = '../data/Taipei_QA.txt'
    tmp = read_data(data_path)
    questions, answers = tmp[0], tmp[1]
    questions_dict = DataFormat(questions)
    answers_dict = DataFormat(answers)
    print(answers_dict.data_to_idx('臺北市商業處'))
