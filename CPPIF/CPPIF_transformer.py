# -*- coding: utf-8 -*-
# @Time    : 2022/5/5 16:18
# @Author  : WANG Ruheng
# @Email   : blwangheng@163.com
# @IDE     : PyCharm
# @FileName: CPPIF_transformer.py

# Comprehensive Protein and Peptide interaction prediction Framework - CPPIF
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import re
import numpy as np
import math

def get_attn_pad_mask(seq):
    # seq = torch.cat([torch.ones([seq.size(0), 1], device=device), seq], dim=1)
    batch_size, seq_len = seq.size()
    pad_attn_mask = seq.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    pad_attn_mask_expand = pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]
    return pad_attn_mask_expand

class PositionalEncoding(nn.Module):
    def __init__(self, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# class Embedding(nn.Module):
#     def __init__(self):
#         super(Embedding, self).__init__()
#         self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding (look-up table)
#         self.pos_embed = nn.Embedding(800, d_model)  # position embedding
#         self.norm = nn.LayerNorm(d_model)
#
#     def forward(self, x):
#         seq_len = x.size(1)  # x: [batch_size, seq_len]
#         pos = torch.arange(seq_len, device=device, dtype=torch.long)  # [seq_len]
#         pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
#         embedding = self.pos_embed(pos)
#         embedding = embedding + self.tok_embed(x)
#         embedding = self.norm(embedding)
#         return embedding


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_head, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)  # [batch_size, n_head, seq_len, seq_len]
        context = torch.matmul(attn, V)  # [batch_size, n_head, seq_len, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_head, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_head, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_head, bias=False)
        self.linear = nn.Linear(n_head * d_v, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # q_s: [batch_size, n_head, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # k_s: [batch_size, n_head, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_head, d_v).transpose(1, 2)  # v_s: [batch_size, n_head, seq_len, d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_head, 1, 1)
        context, attention_map = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        # context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_head * d_v)  # context: [batch_size, seq_len, n_head * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, n_head * d_v)
        output = self.linear(context)
        output = self.norm(output + residual)
        return output, attention_map


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        residual = x
        output = self.fc2(self.relu(self.fc1(x)))
        output = self.norm(output + residual)
        return output


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
        self.attention_map = None

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attention_map = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                                        enc_self_attn_mask)  # enc_inputs to same Q,K,V
        self.attention_map = attention_map
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pretrainpath = '/home/sde3/wrh/PepBCL/prot_bert_bfd'
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrainpath, do_lower_case=False)
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding (look-up table)
        self.pos_embed = PositionalEncoding()  # position embedding
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, seqs, attn_mask):
        seqs = [re.sub(r"[UZOB]", "X", ' '.join(i)) for i in seqs]
        token_seq = self.tokenizer(seqs, return_tensors='pt')
        input_ids, token_type_ids, attention_mask = token_seq['input_ids'], token_seq['token_type_ids'], token_seq[
            'attention_mask']
        input_ids = input_ids.cuda()
        input_ids = input_ids[:, 1:-1]

        output = self.tok_embed(input_ids)  # [bach_size, seq_len, d_model]
        output = self.pos_embed(output.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(attn_mask)  # [batch_size, maxlen, maxlen]
        for layer in self.layers:
            output = layer(output, enc_self_attn_mask)
            # output: [batch_size, max_len, d_model]

        return output


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, dec_self_attn_mask):  # dec_inputs = enc_outputs
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        #         self.tgt_emb = nn.Embedding(d_model * 2, d_model)
        self.pos_embed = PositionalEncoding()
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
        self.tgt_len = tgt_len

    def forward(self, dec_inputs):  # dec_inputs = enc_outputs (batch_size, peptide_hla_maxlen_sum, d_model)
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        '''
        #         dec_outputs = self.tgt_emb(dec_inputs) # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_embed(dec_inputs.transpose(0, 1)).transpose(0, 1).to(device)  # [batch_size, tgt_len, d_model]
        #         dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda() # [batch_size, tgt_len, tgt_len]
        dec_self_attn_pad_mask = torch.LongTensor(np.zeros((dec_inputs.shape[0], tgt_len, tgt_len))).bool().to(device)

        # dec_self_attns = []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs = layer(dec_outputs, dec_self_attn_pad_mask)
            # dec_self_attns.append(dec_self_attn)

        return dec_outputs


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        global n_layers, n_head, d_model, d_ff, d_k, d_v, vocab_size, device, tgt_len
        # max_len = config.max_len
        n_layers = config.num_layer
        n_head = config.num_head
        d_model = config.dim_embedding
        d_ff = config.dim_feedforward
        d_k = config.dim_k
        d_v = config.dim_v
        vocab_size = 31  # bert tokenizer vocabulary number
        device = torch.device("cuda" if config.cuda else "cpu")
        tgt_len = config.pad_pep_len + config.pad_prot_len

        self.prot_encoder = Encoder().to(device)
        self.pep_encoder = Encoder().to(device)
        self.decoder = Decoder().to(device)
        self.tgt_len = tgt_len
        self.pre_projection = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.Dropout(0.2),
            nn.ReLU(True),
            nn.Linear(32, 1)
        ).to(device)
        self.projection = nn.Sequential(
            nn.Linear(tgt_len * d_model, 256),
            nn.ReLU(True),

            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(True),

            # output layer
            nn.Linear(64, 2)
        ).to(device)
        self.prot_site_cls = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.Dropout(0.2),
            nn.ReLU(True),
            nn.Linear(256, 2)
        ).to(device)
        self.pep_site_cls = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.Dropout(0.2),
            nn.ReLU(True),
            nn.Linear(256, 2)
        ).to(device)

    def forward(self, prot_inputs, pep_inputs, prot_attn_mask, pep_attn_mask):
        '''
        pep_inputs: [batch_size, pep_len]
        hla_inputs: [batch_size, hla_len]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        prot_enc_outputs = self.prot_encoder(prot_inputs, prot_attn_mask)
        pep_enc_outputs = self.pep_encoder(pep_inputs, pep_attn_mask)
        enc_outputs = torch.cat((prot_enc_outputs, pep_enc_outputs), 1)  # concat pep & hla embedding

        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs = self.decoder(enc_outputs)
        # dec_outputs = dec_outputs.view(dec_outputs.shape[0], -1)  # Flatten [batch_size, tgt_len * d_model]

        # binary interaction prediction
        # dec_bi = self.pre_projection(dec_outputs)
        dec_logits = self.projection(dec_outputs.view(dec_outputs.shape[0], -1))  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]

        # binding sites prediction
        # prot_site_logits = self.prot_site_cls(dec_outputs[:, :679])
        pep_site_logits = self.pep_site_cls(dec_outputs[:, 679:])

        return dec_logits.view(-1, dec_logits.size(-1)), pep_site_logits
        #, pep_enc_self_attns, hla_enc_self_attns, dec_self_attns

class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()
        self.config = config
		# 加载预训练模型参数
        self.pretrainpath = '/home/sde3/wrh/PepBCL/prot_bert_bfd'
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrainpath, do_lower_case=False)
        self.bert = BertModel.from_pretrained(self.pretrainpath)

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
        prot_cnn_out_dim = 192
        prot_seq_len = 679
        prot_cnn_win_size = 5

        pep_cnn_in_dim = 512
        pep_cnn_h1_dim = 64
        pep_cnn_h2_dim = 128
        pep_cnn_out_dim = 192
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
        # self.BERT = BERT(config)
        self.transformer_encoder1 = Encoder()
        self.transformer_encoder2 = Encoder()
        self.transformer_encoder3 = Encoder()
        self.binary_classification = nn.Sequential(
			nn.Linear(640, 1024),
			nn.Dropout(0.2),
			nn.ReLU(),
            nn.Linear(1024, 1024),
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
			nn.Linear(192 + 128, 512),
			nn.Dropout(0.2),
			nn.ReLU(),
			nn.Linear(512, 2)
		)
        self.pep_site_cls = nn.Sequential(
			nn.Linear(192 + 128, 512),
			nn.Dropout(0.2),
			nn.ReLU(),
			nn.Linear(512, 2)
		)
        self.tra_fea_encoder = traditional_feature_encoder()
        self.max_pool1 = nn.MaxPool1d(prot_seq_len)
        self.max_pool2 = nn.MaxPool1d(pep_seq_len)
        self.max_pool3 = nn.MaxPool1d(prot_seq_len)
        self.max_pool4 = nn.MaxPool1d(pep_seq_len)

    def predict(self, X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p, prot_seqs, pep_seqs, prot_mask, pep_mask):
        # representation = self.BERT(prot_seqs, pep_seqs)
        prot_trans = self.transformer_encoder1(prot_seqs, prot_mask)
        pep_trans = self.transformer_encoder1(pep_seqs, pep_mask)
        prot_bert_fea = prot_trans[:, 1:]
        pep_bert_fea = pep_trans[:, 1:]

        prot_tra_fea, pep_tra_fea = self.tra_fea_encoder(X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep,
														 X_dense_p)


        # binary prediction
        prot_cnn_fea = prot_tra_fea.permute(0, 2, 1)
        prot_cnn_fea = self.max_pool1(prot_cnn_fea)
        pep_cnn_fea = pep_tra_fea.permute(0, 2, 1)
        pep_cnn_fea = self.max_pool2(pep_cnn_fea)
        prot_cnn_fea = prot_cnn_fea.view(-1, 192)
        pep_cnn_fea = pep_cnn_fea.view(-1, 192)

        prot_bert_bi_fea = prot_bert_fea.permute(0, 2, 1)
        prot_bert_bi_fea = self.max_pool3(prot_bert_bi_fea)
        pep_bert_bi_fea = pep_bert_fea.permute(0, 2, 1)
        pep_bert_bi_fea = self.max_pool4(pep_bert_bi_fea)
        prot_bert_bi_fea = prot_bert_bi_fea.view(-1, 128)
        pep_bert_bi_fea = pep_bert_bi_fea.view(-1, 128)

        # prot_bert_bi_fea = self.prot_bert_bi_linear(prot_bert_bi_fea)
        # pep_bert_bi_fea = self.pep_bert_bi_linear(pep_bert_bi_fea)
        binary_fea = torch.cat([prot_cnn_fea, pep_cnn_fea, prot_bert_bi_fea, pep_bert_bi_fea], dim=1)
        binary_result = self.binary_classification(binary_fea)

        # site prediction
        # prot_bert_fea = self.prot_bert_linear(prot_bert_fea)
        # pep_bert_fea = self.pep_bert_linear(pep_bert_fea)
        prot_all_fea = torch.cat([prot_bert_fea, prot_tra_fea], dim=2)
        pep_all_fea = torch.cat([pep_bert_fea, pep_tra_fea], dim=2)
        prot_site_result = self.prot_site_cls(prot_all_fea)
        pep_site_result = self.pep_site_cls(pep_all_fea)

        return binary_result, prot_site_result, pep_site_result