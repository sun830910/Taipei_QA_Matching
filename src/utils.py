# -*- coding: utf-8 -*-

"""
Created on 2020-08-13 09:05
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""


def read_data(data_path):
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

    print("数据中共有 {} 个办公单位".format(len(set(answers))))

    assert  len(questions) == len(answers)
    return questions, answers

class data_format(object):
    def __init__(self, data):
        self.data = data
        self.unique_data = list(set(self.data))
        self.data_length = len(self.unique_data)
        self.data_list = []
        self._build_dict()

    def _build_dict(self):
        for data_idx, data in enumerate(self.unique_data):
            if data is not None:
                self.data_list.append((data_idx, data))

    def data_to_idx(self, text):
        if text in self.data_list:
            return self.data_list.index(text)

    def idx_to_data(self, idx):
        if idx in self.data_list:
            return self.data[idx]

    @property
    def data_length(self):
        return self.data_length

    @property
    def data(self):
        return self.data

    @property
    def __len__(self):
        return len(self.data)


def sentence_to_idx(tokenizer, sentence):
    return tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence)))


def get_embedding(tokenizer, data_path):
    questions, answers = read_data(data_path)
    questions_dict = data_format(questions)
    answers_dict = data_format(answers)

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
    return questions_tokens, labels









if __name__ == '__main__':
    data_path = '../data/Taipei_QA.txt'
    tmp = read_data(data_path)

