# -*- coding: utf-8 -*-
# @Time    : 2022/4/18 13:24
# @Author  : WANG Ruheng
# @Email   : blwangheng@163.com
# @IDE     : PyCharm
# @FileName: CPPIF.py
# Comprehensive Protein and Peptide interaction prediction Framework - CPPIF
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig
import re

class BERT(nn.Module):
	def __init__(self, config):
		super(BERT, self).__init__()
		self.config = config
		# 加载预训练模型参数
		self.pretrainpath = '/home/sde3/wrh/PepBCL/prot_bert_bfd'

		self.setting = BertConfig.from_pretrained(
			self.pretrainpath,
			# num_labels=2,
			# finetuning_task="dnaprom",
			# cache_dir=None,
			output_attentions=True
		)
		self.tokenizer = BertTokenizer.from_pretrained(self.pretrainpath, do_lower_case=False)
		self.bert = BertModel.from_pretrained(self.pretrainpath, config=self.setting)

	def forward(self, prot_seqs, pep_seqs):
		prot_seqs = [re.sub(r"[UZOB]", "X", ' '.join(i)) + ' [SEP] ' for i in prot_seqs]
		pep_seqs = [re.sub(r"[UZOB]", "X", ' '.join(i)) for i in pep_seqs]
		sequences = [prot_seqs[i] + pep_seqs[i] for i in range(len(prot_seqs))]
		token_seq = self.tokenizer(sequences, return_tensors='pt')
		input_ids, token_type_ids, attention_mask = token_seq['input_ids'], token_seq['token_type_ids'], token_seq[
            'attention_mask']
		if self.config.cuda:
			representation = self.bert(input_ids.cuda(), token_type_ids.cuda(), attention_mask.cuda())
		else:
			representation = self.bert(input_ids, token_type_ids, attention_mask)
		return representation

class CNN(nn.Module):
	def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, window_size):
		super(CNN, self).__init__()
		pad = (window_size - 1) // 2
		self.relu = nn.ReLU()
		self.conv1d1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim1, kernel_size=window_size, padding=pad)
		self.conv1d2 = nn.Conv1d(in_channels=hidden_dim1, out_channels=hidden_dim2, kernel_size=window_size, padding=pad)
		self.conv1d3 = nn.Conv1d(in_channels=hidden_dim2, out_channels=output_dim, kernel_size=window_size, padding=pad)
		# self.max_pool = nn.MaxPool1d(seq_len)
	def forward(self, input):
		# connect = input # [Batch_size, input_dim, seq_len]
		output = self.conv1d1(input.permute(0, 2, 1))  # [Batch_size, hidden_dim1, seq_len]
		output = self.relu(output)
		output = self.conv1d2(output) # [Batch_size, hidden_dim2, seq_len]
		output = self.relu(output)
		output = self.conv1d3(output) # [Batch_size, output_dim, seq_len]
		output = self.relu(output)
		output = output.permute(0, 2, 1)  # [Batch_size, seq_len, output_dim]
		# output = self.max_pool(output)  # [Batch_size, output_dim, 1]
		return output

class traditional_feature_encoder(nn.Module):
    def __init__(self):
        super(traditional_feature_encoder, self).__init__()
        # parameters setting
        aa_out_dim = 128
        ss_out_dim = 128
        pp_out_dim = 128

        prot_dense_in_dim = 23
        prot_dense_out_dim = 128
        pep_dense_in_dim = 3
        pep_dense_out_dim = 128

        prot_cnn_in_dim = 512
        prot_cnn_h1_dim = 64
        prot_cnn_h2_dim = 128
        prot_cnn_out_dim = 512
        prot_seq_len = 679
        prot_cnn_win_size = 5

        pep_cnn_in_dim = 512
        pep_cnn_h1_dim = 64
        pep_cnn_h2_dim = 128
        pep_cnn_out_dim = 512
        pep_seq_len = 50
        pep_cnn_win_size = 5

        # protein
        # categorical channels
        self.prot_aa_embedd = nn.Embedding(21 + 1, aa_out_dim)
        self.prot_ss_embedd = nn.Embedding(63 + 1, ss_out_dim)
        self.prot_pp_embedd = nn.Embedding(7 + 1, pp_out_dim)

        # numerical channel
        self.fc1 = nn.Linear(prot_dense_in_dim, prot_dense_out_dim)

        # feature process
        self.cnn1 = CNN(input_dim=prot_cnn_in_dim, hidden_dim1=prot_cnn_h1_dim, hidden_dim2=prot_cnn_h2_dim, output_dim=prot_cnn_out_dim, window_size=prot_cnn_win_size)

        # peptide
        # categorical channels
        self.pep_aa_embedd = nn.Embedding(21 + 1, aa_out_dim)
        self.pep_ss_embedd = nn.Embedding(63 + 1, ss_out_dim)
        self.pep_pp_embedd = nn.Embedding(7 + 1, pp_out_dim)

        # numerical channel
        self.fc2 = nn.Linear(pep_dense_in_dim, pep_dense_out_dim)

        # feature process
        self.cnn2 = CNN(input_dim=pep_cnn_in_dim, hidden_dim1=pep_cnn_h1_dim, hidden_dim2=pep_cnn_h2_dim, output_dim=pep_cnn_out_dim, window_size=pep_cnn_win_size)

    def forward(self, X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p):
        # categorical channels
        prot_aa = self.prot_aa_embedd(X_p)
        pep_aa = self.pep_aa_embedd(X_pep)
        prot_ss = self.prot_ss_embedd(X_SS_p)
        pep_ss = self.pep_ss_embedd(X_SS_pep)
        prot_pp = self.prot_pp_embedd(X_2_p)
        pep_pp = self.pep_pp_embedd(X_2_pep)

        # numerical channels
        prot_dense = self.fc1(X_dense_p)
        pep_dense = self.fc2(X_dense_pep)

        # protein
        prot_all_fea = torch.cat([prot_aa, prot_ss, prot_pp, prot_dense], dim=2)
        prot_cnn_fea = self.cnn1(prot_all_fea)
        # peptide
        pep_all_fea = torch.cat([pep_aa, pep_ss, pep_pp, pep_dense], dim=2)
        pep_cnn_fea = self.cnn2(pep_all_fea)

        return prot_cnn_fea, pep_cnn_fea

class Model(nn.Module):
	def __init__(self, config):
		super(Model, self).__init__()
		prot_seq_len = 679
		pep_seq_len = 50

		self.config = config
		self.BERT = BERT(config)
		self.binary_classification = nn.Sequential(
			nn.Linear(2048, 1024),
			nn.Dropout(0.2),
			nn.ReLU(),
			nn.Linear(1024, 512),
			nn.Dropout(0.2),
			nn.ReLU(),
			nn.Linear(512, 2)
		)
		self.prot_bert_bi_linear = nn.Linear(1024, 512)
		self.pep_bert_bi_linear = nn.Linear(1024, 512)
		self.prot_bert_linear = nn.Linear(1024, 512)
		self.pep_bert_linear = nn.Linear(1024, 512)
		self.prot_site_cls = nn.Sequential(
			nn.Linear(1024, 1024),
			nn.Dropout(0.2),
			nn.ReLU(),
			nn.Linear(1024, 512),
			nn.Dropout(0.2),
			nn.ReLU(),
			nn.Linear(512, 2)
		)
		self.pep_site_cls = nn.Sequential(
			nn.Linear(1024, 1024),
			nn.Dropout(0.2),
			nn.ReLU(),
			nn.Linear(1024, 512),
			nn.Dropout(0.2),
			nn.ReLU(),
			nn.Linear(512, 2)
		)
		self.tra_fea_encoder = traditional_feature_encoder()
		self.max_pool1 = nn.MaxPool1d(prot_seq_len)
		self.max_pool2 = nn.MaxPool1d(pep_seq_len)
		self.max_pool3 = nn.MaxPool1d(prot_seq_len)
		self.max_pool4 = nn.MaxPool1d(pep_seq_len)

	def binary_predict(self, prot_seqs, pep_seqs):
		representation = self.BERT(prot_seqs, pep_seqs)
		pool_rep = representation["pooler_output"]
		binary_result = self.binary_classification(pool_rep)

		return binary_result

	def site_predict(self, X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p, prot_seqs, pep_seqs):
		with torch.no_grad():
			representation = self.BERT(prot_seqs, pep_seqs)
		site_bert_rep = representation[0]
		prot_bert_fea = site_bert_rep[:, 1:680]
		pep_bert_fea = site_bert_rep[:, 681:-1]
		prot_bert_fea = self.prot_bert_linear(prot_bert_fea)
		pep_bert_fea = self.pep_bert_linear(pep_bert_fea)
		prot_tra_fea, pep_tra_fea = self.tra_fea_encoder(X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p)
		prot_all_fea = torch.cat([prot_bert_fea, prot_tra_fea], dim=2)
		pep_all_fea = torch.cat([pep_bert_fea, pep_tra_fea], dim=2)
		prot_site_result = self.prot_site_cls(prot_all_fea)
		pep_site_result = self.pep_site_cls(pep_all_fea)

		return prot_site_result, pep_site_result

	def predict(self, X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p, prot_seqs, pep_seqs):
		representation = self.BERT(prot_seqs, pep_seqs)
		site_bert_rep = representation[0]
		prot_bert_fea = site_bert_rep[:, 1:680]
		pep_bert_fea = site_bert_rep[:, 681:-1]

		prot_tra_fea, pep_tra_fea = self.tra_fea_encoder(X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep,
														 X_dense_p)


		# binary prediction
		prot_cnn_fea = prot_tra_fea.permute(0, 2, 1)
		prot_cnn_fea = self.max_pool1(prot_cnn_fea)
		pep_cnn_fea = pep_tra_fea.permute(0, 2, 1)
		pep_cnn_fea = self.max_pool2(pep_cnn_fea)
		prot_cnn_fea = prot_cnn_fea.view(-1, 512)
		pep_cnn_fea = pep_cnn_fea.view(-1, 512)

		prot_bert_bi_fea = prot_bert_fea.permute(0, 2, 1)
		prot_bert_bi_fea = self.max_pool3(prot_bert_bi_fea)
		pep_bert_bi_fea = pep_bert_fea.permute(0, 2, 1)
		pep_bert_bi_fea = self.max_pool4(pep_bert_bi_fea)
		prot_bert_bi_fea = prot_bert_bi_fea.view(-1, 1024)
		pep_bert_bi_fea = pep_bert_bi_fea.view(-1, 1024)

		prot_bert_bi_fea = self.prot_bert_bi_linear(prot_bert_bi_fea)
		pep_bert_bi_fea = self.pep_bert_bi_linear(pep_bert_bi_fea)
		binary_fea = torch.cat([prot_cnn_fea, pep_cnn_fea, prot_bert_bi_fea, pep_bert_bi_fea], dim=1)
		binary_result = self.binary_classification(binary_fea)

		# site prediction
		prot_bert_fea = self.prot_bert_linear(prot_bert_fea)
		pep_bert_fea = self.pep_bert_linear(pep_bert_fea)
		prot_all_fea = torch.cat([prot_bert_fea, prot_tra_fea], dim=2)
		pep_all_fea = torch.cat([pep_bert_fea, pep_tra_fea], dim=2)
		prot_site_result = self.prot_site_cls(prot_all_fea)
		pep_site_result = self.pep_site_cls(pep_all_fea)

		return binary_result, prot_site_result, pep_site_result


