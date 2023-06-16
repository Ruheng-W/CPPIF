# -*- coding: utf-8 -*-
# @Time    : 2022/3/23 22:12
# @Author  : WANG Ruheng
# @Email   : blwangheng@163.com
# @IDE     : PyCharm
# @FileName: nc_model.py

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Self_Attention(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Self_Attention, self).__init__()
        self.output_dim = output_dim
        self.W_Q = nn.Linear(input_dim, output_dim)
        self.W_K = nn.Linear(input_dim, output_dim)
        self.W_V = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        q_s = self.W_Q(x)
        k_s = self.W_K(x)
        v_s = self.W_V(x)

        scores = torch.matmul(q_s, k_s.transpose(-1, -2)) / (self.output_dim**0.5)
        attn = nn.Softmax(dim=-1)(scores)  # [batch_size, seq_len, seq_len]
        output = torch.matmul(attn, v_s)

        return output

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
        input = input.permute(0, 2, 1)  # [Batch_size, input_dim, seq_len]
        output = self.conv1d1(input)  # [Batch_size, hidden_dim1, seq_len]
        output = self.relu(output)
        output = self.conv1d2(output) # [Batch_size, hidden_dim2, seq_len]
        output = self.relu(output)
        output = self.conv1d3(output) # [Batch_size, output_dim, seq_len]
        # output = output.permute(0, 2, 1)  # [Batch_size, seq_len, output_dim]
        output = self.relu(output)
        # output = self.max_pool(output)  # [Batch_size, output_dim, 1]
        return output

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
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
        prot_cnn_out_dim = 192
        prot_seq_len = 679
        prot_cnn_win_size = 5

        prot_atten_in_dim = 128
        prot_atten_out_dim = 128

        pep_cnn_in_dim = 512
        pep_cnn_h1_dim = 64
        pep_cnn_h2_dim = 128
        pep_cnn_out_dim = 192
        pep_seq_len = 50
        pep_cnn_win_size = 5

        pep_atten_in_dim = 128
        pep_atten_out_dim = 128

        bi_cls1_in_dim = 640
        bi_cls1_out_dim = 1024
        bi_cls2_in_dim = 1024
        bi_cls2_out_dim = 1024
        bi_cls3_in_dim = 1024
        bi_cls3_out_dim = 512
        bi_cls3_2_in_dim = 512
        bi_cls3_2_out_dim = 2
        site_cls4_in_dim = 192
        site_cls4_out_dim = 2
        dropout_rate = 0.5

        # protein
        # categorical channels
        self.prot_aa_embedd1 = nn.Embedding(21 + 1, aa_out_dim)
        self.prot_aa_embedd2 = nn.Embedding(21 + 1, aa_out_dim)
        self.prot_ss_embedd = nn.Embedding(63 + 1, ss_out_dim)
        self.prot_pp_embedd = nn.Embedding(7 + 1, pp_out_dim)

        # numerical channel
        self.fc1 = nn.Linear(prot_dense_in_dim, prot_dense_out_dim)

        # feature process
        self.cnn1 = CNN(input_dim=prot_cnn_in_dim, hidden_dim1=prot_cnn_h1_dim, hidden_dim2=prot_cnn_h2_dim, output_dim=prot_cnn_out_dim, window_size=prot_cnn_win_size)
        self.atten1 = Self_Attention(input_dim=prot_atten_in_dim, output_dim=prot_atten_out_dim)
        self.max_pool1 = nn.MaxPool1d(prot_seq_len)
        self.max_pool3 = nn.MaxPool1d(prot_seq_len)

        # peptide
        # categorical channels
        self.pep_aa_embedd1 = nn.Embedding(21 + 1, aa_out_dim)
        self.pep_aa_embedd2 = nn.Embedding(21 + 1, aa_out_dim)
        self.pep_ss_embedd = nn.Embedding(63 + 1, ss_out_dim)
        self.pep_pp_embedd = nn.Embedding(7 + 1, pp_out_dim)

        # numerical channel
        self.fc2 = nn.Linear(pep_dense_in_dim, pep_dense_out_dim)

        # feature process
        self.cnn2 = CNN(input_dim=pep_cnn_in_dim, hidden_dim1=pep_cnn_h1_dim, hidden_dim2=pep_cnn_h2_dim, output_dim=pep_cnn_out_dim, window_size=pep_cnn_win_size)
        self.atten2 = Self_Attention(input_dim=pep_atten_in_dim, output_dim=pep_atten_out_dim)
        self.max_pool2 = nn.MaxPool1d(pep_seq_len)
        self.max_pool4 = nn.MaxPool1d(pep_seq_len)

        # binary interaction prediction
        self.cls1 = nn.Linear(bi_cls1_in_dim, bi_cls1_out_dim)
        self.cls2 = nn.Linear(bi_cls2_in_dim, bi_cls2_out_dim)
        self.cls3 = nn.Linear(bi_cls3_in_dim, bi_cls3_out_dim)
        self.cls3_2 = nn.Linear(bi_cls3_2_in_dim, bi_cls3_2_out_dim)
        self.drop = nn.Dropout(dropout_rate)
        # self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        # peptide binding residue prediction
        self.cls4 = nn.Linear(site_cls4_in_dim, site_cls4_out_dim)
        # self.sigmoid2 = nn.Sigmoid()

    def forward(self, X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p):
        # categorical channels
        prot_aa1 = self.prot_aa_embedd1(X_p)
        pep_aa1 = self.pep_aa_embedd1(X_pep)
        prot_aa2 = self.prot_aa_embedd2(X_p)
        pep_aa2 = self.pep_aa_embedd2(X_pep)
        prot_ss = self.prot_ss_embedd(X_SS_p)
        pep_ss = self.pep_ss_embedd(X_SS_pep)
        prot_pp = self.prot_pp_embedd(X_2_p)
        pep_pp = self.pep_pp_embedd(X_2_pep)

        # numerical channels
        prot_dense = self.fc1(X_dense_p)
        pep_dense = self.fc2(X_dense_pep)

        # protein
        prot_all_fea = torch.cat([prot_aa1, prot_ss, prot_pp, prot_dense], dim=2)
        prot_cnn_fea = self.cnn1(prot_all_fea)
        prot_cnn_fea = self.max_pool3(prot_cnn_fea)
        prot_atten_fea = self.atten1(prot_aa2)
        prot_atten_fea = prot_atten_fea.permute(0, 2, 1)
        prot_atten_fea = self.max_pool1(prot_atten_fea)

        # peptide
        pep_all_fea = torch.cat([pep_aa1, pep_ss, pep_pp, pep_dense], dim=2)
        pep_cnn_fea_site = self.cnn2(pep_all_fea)
        pep_cnn_fea = pep_cnn_fea_site
        pep_cnn_fea = self.max_pool4(pep_cnn_fea)
        pep_atten_fea = self.atten2(pep_aa2)
        pep_atten_fea = pep_atten_fea.permute(0, 2, 1)
        pep_atten_fea = self.max_pool2(pep_atten_fea)

        # binary interaction prediction
        prot_cnn_fea = prot_cnn_fea.view(-1, 192)
        pep_cnn_fea = pep_cnn_fea.view(-1, 192)
        prot_atten_fea = prot_atten_fea.view(-1, 128)
        pep_atten_fea = pep_atten_fea.view(-1, 128)
        binary_fea = torch.cat([prot_cnn_fea, pep_cnn_fea, prot_atten_fea, pep_atten_fea], dim=1)
        # binary_fea = binary_fea.view(batch_size, -1)
        b_output = self.cls1(binary_fea)
        b_output = self.drop(b_output)
        b_output = self.relu(b_output)
        b_output = self.cls2(b_output)
        b_output = self.drop(b_output)
        b_output = self.relu(b_output)
        b_output = self.cls3(b_output)
        b_output = self.relu(b_output)
        b_output = self.cls3_2(b_output)
        out_bi_y = b_output
        # out_bi_y = self.sigmoid(b_output)    # [Batch_size, 1]

        # peptide binding residue prediction
        pep_cnn_fea_site = pep_cnn_fea_site.permute(0, 2, 1)
        out_site_y = self.cls4(pep_cnn_fea_site)
        # out_site_y = self.sigmoid2(out_site_y)  # [Batch_size, seq_len]
        # out_site_y = out_site_y.view(-1, 50)

        return out_bi_y, out_site_y

    def binary_forward(self, X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p):
        # categorical channels
        prot_aa1 = self.prot_aa_embedd1(X_p)
        pep_aa1 = self.pep_aa_embedd1(X_pep)
        prot_aa2 = self.prot_aa_embedd2(X_p)
        pep_aa2 = self.pep_aa_embedd2(X_pep)
        prot_ss = self.prot_ss_embedd(X_SS_p)
        pep_ss = self.pep_ss_embedd(X_SS_pep)
        prot_pp = self.prot_pp_embedd(X_2_p)
        pep_pp = self.pep_pp_embedd(X_2_pep)

        # numerical channels
        prot_dense = self.fc1(X_dense_p)
        pep_dense = self.fc2(X_dense_pep)

        # protein
        prot_all_fea = torch.cat([prot_aa1, prot_ss, prot_pp, prot_dense], dim=2)
        prot_cnn_fea = self.cnn1(prot_all_fea)
        prot_cnn_fea = self.max_pool3(prot_cnn_fea)
        prot_atten_fea = self.atten1(prot_aa2)
        prot_atten_fea = prot_atten_fea.permute(0, 2, 1)
        prot_atten_fea = self.max_pool1(prot_atten_fea)

        # peptide
        pep_all_fea = torch.cat([pep_aa1, pep_ss, pep_pp, pep_dense], dim=2)
        pep_cnn_fea = self.cnn2(pep_all_fea)
        pep_cnn_fea = self.max_pool4(pep_cnn_fea)
        pep_atten_fea = self.atten2(pep_aa2)
        pep_atten_fea = pep_atten_fea.permute(0, 2, 1)
        pep_atten_fea = self.max_pool2(pep_atten_fea)

        # binary interaction prediction
        prot_cnn_fea = prot_cnn_fea.view(-1, 192)
        pep_cnn_fea = pep_cnn_fea.view(-1, 192)
        prot_atten_fea = prot_atten_fea.view(-1, 128)
        pep_atten_fea = pep_atten_fea.view(-1, 128)
        binary_fea = torch.cat([prot_cnn_fea, pep_cnn_fea, prot_atten_fea, pep_atten_fea], dim=1)
        # binary_fea = binary_fea.view(batch_size, -1)
        b_output = self.cls1(binary_fea)
        b_output = self.drop(b_output)
        b_output = self.relu(b_output)
        b_output = self.cls2(b_output)
        b_output = self.drop(b_output)
        b_output = self.relu(b_output)
        b_output = self.cls3(b_output)
        b_output = self.relu(b_output)
        b_output = self.cls3_2(b_output)
        out_bi_y = self.sigmoid(b_output)  # [Batch_size, 1]

        return out_bi_y