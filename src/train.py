# -*- coding: utf-8 -*-

"""
Created on 2020-08-13 13:40
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""


import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from utils import get_embedding, make_torch_input, split_dataset, compute_acc, load_bert


class QAMatcher(object):

    def __init__(self):
        self.setup()


    def load_model(self):
        model_setting = {
            'model_name': "bert",
            'config_path': '../pretrained_model/chinese_L-12_H-768_A-12/bert_config.json',
            'model_path': '../pretrained_model/chinese_L-12_H-768_A-12/pytorch_model.bin',
            'vocab_path': '../pretrained_model/chinese_L-12_H-768_A-12/vocab.txt',
            'num_labels': 149  # 分几类
        }

        self.model, self.tokenizer = load_bert(**model_setting)

        # setting device，将模型加载至设备中
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("using device:{}".format(self.device))
        self.model.to(self.device)


    def get_torch_input(self):
        data_path = '../data/Taipei_QA.txt'
        data_features = get_embedding(self.tokenizer, data_path)
        input_idx = data_features["input_idx"]
        input_masks = data_features["input_masks"]
        input_segment_idx = data_features["input_segment_idx"]
        answer_labels = data_features["answer_labels"]

        all_dataset = make_torch_input(input_idxs=input_idx, input_masks=input_masks,
                                       input_segment_idxs=input_segment_idx,
                                       labels=answer_labels)
        train_dataset, test_dataset = split_dataset(all_dataset, 0.9)
        self.train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)


    def set_optimizer(self):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=5e-6, eps=1e-8)
        self.model.zero_grad()


    def setup(self):
        self.load_model()
        self.get_torch_input()
        self.set_optimizer()


    def save_model(self):
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained('../result/trained_model')

    def train(self, epochs):
        for epoch in range(epochs):
            running_loss_val = 0.0
            running_acc = 0.0
            for batch_index, batch_dict in enumerate(self.train_dataloader):
                self.model.train()
                batch_dict = tuple(t.to(self.device) for t in batch_dict)
                outputs = self.model(
                    batch_dict[0],  # 取cls
                    labels=batch_dict[3]
                )
                loss, logits = outputs[:2]
                loss.sum().backward()
                self.optimizer.step()
                # scheduler.step()  # Update learning rate schedule
                self.model.zero_grad()

                # compute the loss
                loss_t = loss.item()
                running_loss_val += (loss_t - running_loss_val) / (batch_index + 1)

                # compute the accuracy
                acc_t = compute_acc(logits, batch_dict[3])
                running_acc += (acc_t - running_acc) / (batch_index + 1)

                # log
                print("epoch:%2d batch:%4d train_loss:%2.4f train_acc:%3.4f" % (
                epoch + 1, batch_index + 1, running_loss_val, running_acc))

            # 验证
            running_loss_val = 0.0
            running_acc = 0.0
            for batch_index, batch_dict in enumerate(self.test_dataloader):
                self.model.eval()
                batch_dict = tuple(t.to(self.device) for t in batch_dict)
                outputs = self.model(
                    batch_dict[0],
                    # attention_mask=batch_dict[1],
                    labels=batch_dict[3]
                )
                loss, logits = outputs[:2]

                # compute the loss
                loss_t = loss.item()
                running_loss_val += (loss_t - running_loss_val) / (batch_index + 1)

                # compute the accuracy
                acc_t = compute_acc(logits, batch_dict[3])
                running_acc += (acc_t - running_acc) / (batch_index + 1)

                # log
                print("epoch:%2d batch:%4d test_loss:%2.4f test_acc:%3.4f" % (
                epoch + 1, batch_index + 1, running_loss_val, running_acc))
        self.save_model()

if __name__ == '__main__':
    QAModel = QAMatcher()
    QAModel.train(30)






