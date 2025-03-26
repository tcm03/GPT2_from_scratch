import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Callable

class Attention(nn.Module):

    def __init__(
        self,
        d_in: int,
        d_out: int,
        qkv_bias: bool = False
    ) -> None:
        super().__init__()
        self.d_in: int = d_in
        self.d_out: int = d_out
        self.qkv_bias: bool = qkv_bias

    def get_linear(self, d_in, d_out, qkv_bias):
        return nn.Linear(
            in_features = d_in,
            out_features = d_out,
            bias = qkv_bias
        )

class CausalAttention(Attention):

    def __init__(
        self,
        d_in: int,
        d_out: int,
        max_context_length: int,
        dropout_rate: int = 0,
        qkv_bias: bool = False
    ) -> None:
        super().__init__(d_in, d_out, qkv_bias)

        self.W_q: nn.Linear = super().get_linear(d_in, d_out, qkv_bias)
        self.W_k: nn.Linear = super().get_linear(d_in, d_out, qkv_bias)
        self.W_v: nn.Linear = super().get_linear(d_in, d_out, qkv_bias)
        self.dropout = nn.Dropout(p=dropout_rate) # reduce overfitting during training
        self.register_buffer(
            name = 'mask',
            tensor = torch.triu(torch.ones(max_context_length, max_context_length), diagonal = 1).bool(),
            persistent = True
        )

    def forward(
        self,
        X: torch.Tensor
    ) -> None:
        B, seq_len, dim = X.shape
        queries = self.W_q(X)
        keys = self.W_k(X)
        values = self.W_v(X)
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask[:seq_len, :seq_len], -torch.inf)
        attn_weights = F.softmax(
            input = attn_scores / self.d_out**.5,
            dim = -1
        )
        attn_weights = 1/(1 - self.dropout.p) * self.dropout(attn_weights)
        contexts = attn_weights @ values

        return contexts

class MultiHeadAttention(CausalAttention):

    def __init__(
        self,
        d_in: int,
        d_out: int,
        qkv_bias: int,
        max_context_length: int,
        num_heads: int,
        dropout_rate: int = 0
    ) -> None:
        super().__init__(
            d_in = d_in,
            d_out = d_out,
            max_context_length = max_context_length,
            dropout_rate = dropout_rate,
            qkv_bias = qkv_bias
        )
        self.num_heads = num_heads
        self.d_head = d_out // num_heads
        self.out_proj = nn.Linear(d_out, d_out) # optional

    def forward(
        self,
        X: torch.Tensor
    ) -> None:
        assert X.ndim == 3, "Mismatch: (batch_size, sequence length, input dimension)"
        bs, seq_len, d_in = X.shape
        queries = self.W_q(X)
        keys = self.W_k(X)
        values = self.W_v(X)
        queries = queries.view(bs, seq_len, self.num_heads, self.d_head)
        queries = queries.transpose(1, 2) # (bs, num_heads, seq_len, d_head)
        keys = keys.view(bs, seq_len, self.num_heads, self.d_head)
        keys = keys.transpose(1, 2)
        attn_scores = queries @ keys.transpose(2, 3) # (bs, num_heads, seq_len, seq_len)
        attn_scores = attn_scores.masked_fill_(self.mask[:seq_len, :seq_len], -torch.inf)
        attn_weights = F.softmax(attn_scores / self.d_out**0.5, dim = -1)
        attn_weights = 1/(1 - self.dropout.p) * self.dropout(attn_weights)

        values = values.view(bs, seq_len, self.num_heads, self.d_head).transpose(1, 2) # (bs, num_heads, seq_len, d_head)
        contexts = attn_weights @ values # (bs, num_heads, seq_len, d_head)
        contexts = contexts.transpose(1, 2).flatten(2, 3) # (bs, seq_len, d_out)
        contexts = self.out_proj(contexts)
        assert contexts.ndim == 3, "contexts.shape: (bs, seq_len, d_out)"

        return contexts