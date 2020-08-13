# -*- coding: utf-8 -*-

"""
Created on 2020-08-13 15:43
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""

import torch
import pickle
from utils import sentence_to_idx, load_bert


class QAPredictor(object):

    def __init__(self):
        self.load_model()

    def load_model(self):
        model_setting = {
            'config_path': '../pretrained_model/chinese_L-12_H-768_A-12/bert_config.json',
            'model_path': '../pretrained_model/chinese_L-12_H-768_A-12/pytorch_model.bin',
            'vocab_path': '../pretrained_model/chinese_L-12_H-768_A-12/vocab.txt',
            'num_labels': 149  # 分几类
        }

        self.model, self.tokenizer = load_bert(**model_setting)
        self.model.eval()

    def load_label(self):
        pkl_path = open('../result/data_features.pkl', 'rb')
        data_features = pickle.load(pkl_path)
        self.answer_dict = data_features['answer_dict']

    def predict(self, sentence):
        tokens = sentence_to_idx(self.tokenizer, sentence)
        assert len(tokens) <= 512
        model_input = torch.LongTensor(tokens).unsqueeze(0)

        model_output = self.model(model_input)

        predict = model_output[0]
        max_val = torch.max(predict)
        label = (predict == max_val).nonzero().numpy()[0][1]
        return self.answer_dict.idx_to_data(label)


if __name__ == '__main__':
    predictor = QAPredictor()

    test1 = '為何路邊停車格有編號的要收費，無編號的不用收費'
    print(test1)
    print(predictor.predict(test1))

    test2 = '債權人可否向稅捐稽徵處申請查調債務人之財產、所得資料'
    print(test2)
    print(predictor.predict(test2))

    test3 = '想做大腸癌篩檢，不知如何辨理'
    print(test2)
    print(predictor.predict(test2))
