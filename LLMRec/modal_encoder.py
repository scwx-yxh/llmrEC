"""
在大模型输出后实现模态信息融合的模态专家模块
基于transformer encoder构建
"""

import torch.nn as nn
import torch
import copy
from torch import Tensor
from typing import Optional


class MultiheadAttention(nn.Module):
    def __init__(self, word_emb_dim, nheads, dropout_prob=0.):
        super(MultiheadAttention, self).__init__()
        self.word_emb_dim = word_emb_dim
        self.num_heads = nheads
        self.dropout_prob = dropout_prob
        self.head_dim = word_emb_dim // nheads
        assert self.head_dim * nheads == self.word_emb_dim  # embed_dim must be divisible by num_heads

        self.q_in_proj = nn.Linear(word_emb_dim, word_emb_dim)
        self.k_in_proj = nn.Linear(word_emb_dim, word_emb_dim)
        self.v_in_proj = nn.Linear(word_emb_dim, word_emb_dim)

        self.q_in_modal_proj = nn.Linear(word_emb_dim, word_emb_dim)
        self.k_in_modal_proj = nn.Linear(word_emb_dim, word_emb_dim)
        self.v_in_modal_proj = nn.Linear(word_emb_dim, word_emb_dim)

        self.out_proj = nn.Linear(word_emb_dim, word_emb_dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, ls_len=None, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):

        # 获取query的shape,这里按照torch源码要求，按照tgt_sequence_size, batch_size, word_emb_dim顺序排列
        tgt_len, batch_size, word_emb_dim = query.size()
        num_heads = self.num_heads
        assert word_emb_dim == self.word_emb_dim
        head_dim = word_emb_dim // num_heads

        # 检查word_emb_dim是否可以被num_heads整除
        assert head_dim * num_heads == word_emb_dim
        scaling = float(head_dim) ** -0.5

        query, query_modal = query.split(ls_len, dim=0)
        key, key_modal = key.split(ls_len, dim=0)
        value, value_modal = value.split(ls_len, dim=0)

        # 三个Q、K、V的全连接层
        q, q_modal = self.q_in_proj(query), self.q_in_modal_proj(query_modal)
        k, k_modal = self.k_in_proj(key), self.k_in_modal_proj(key_modal)
        v, v_modal = self.v_in_proj(value), self.v_in_modal_proj(value_modal)

        q = torch.cat((q, q_modal), dim=0)
        k = torch.cat((k, k_modal), dim=0)
        v = torch.cat((v, v_modal), dim=0)

        # 这里对Q进行一个统一常数放缩
        q = q * scaling

        # multihead运算技巧，将word_emb_dim切分为num_heads个head_dim，并且让num_heads与batch_size暂时使用同一维度
        # 切分word_emb_dim后将batch_size * num_heads转换至第0维，为三维矩阵的矩阵乘法（bmm）做准备
        q = q.contiguous().view(tgt_len, batch_size * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, batch_size * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, batch_size * num_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)

        # Q、K进行bmm批次矩阵乘法，得到权重矩阵
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [batch_size * num_heads, tgt_len, src_len]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(batch_size, num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(batch_size * num_heads, tgt_len, src_len)

        # 权重矩阵进行softmax，使得单行的权重和为1
        attn_output_weights = torch.softmax(attn_output_weights, dim=-1)
        attn_output_weights = torch.dropout(attn_output_weights, p=self.dropout_prob, train=self.training)

        # 权重矩阵与V矩阵进行bmm操作，得到输出
        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [batch_size * num_heads, tgt_len, head_dim]

        # 转换维度，将num_heads * head_dim reshape回word_emb_dim，并且将batch_size调回至第1维
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, batch_size, word_emb_dim)

        # 最后一层全连接层，得到最终输出
        attn_output = self.out_proj(attn_output)
        return attn_output

class ModalEncoderLayer(nn.Module):
    def __init__(self, word_emb_dim, nhead, dim_feedforward=2048, dropout_prob=0.1):
        super(ModalEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(word_emb_dim, nhead, dropout_prob=dropout_prob)

        # self.linear1 = nn.Linear(word_emb_dim, dim_feedforward)
        # self.dropout = nn.Dropout(dropout_prob)
        # self.linear2 = nn.Linear(dim_feedforward, word_emb_dim)

        self.norm1 = nn.LayerNorm(word_emb_dim)
        self.norm2_1 = nn.LayerNorm(word_emb_dim)
        self.norm2_2 = nn.LayerNorm(word_emb_dim)
        self.dropout1 = nn.Dropout(dropout_prob)
        # self.dropout2 = nn.Dropout(dropout_prob)

        # self.activation = torch.relu

        self.ffn1 = nn.Sequential(
            nn.Linear(word_emb_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(dim_feedforward, word_emb_dim),
            nn.Dropout(dropout_prob)
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(word_emb_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(dim_feedforward, word_emb_dim),
            nn.Dropout(dropout_prob)
        )

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, ls_len=None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        # self attention
        src2 = self.self_attn(src, src, src, ls_len=ls_len,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # 两层全连接
        src, src_modal = src.split(ls_len, dim=0)

        src = self.norm2_1(src + self.ffn1(src))
        src_modal = self.norm2_2(src_modal + self.ffn2(src_modal))

        src = torch.cat((src, src_modal), dim=0)

        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # src = src + self.dropout2(src2)
        # src = self.norm2(src)
        return src


class ModalEncoder(nn.Module):
    __constants__ = ['norm']
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(ModalEncoder, self).__init__()
        # 将同一个encoder_layer进行deepcopy n次
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, ls_len=None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src
        # 串行n个encoder_layer
        for mod in self.layers:
            output = mod(output, src_mask=mask, ls_len=ls_len, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)
        return output

