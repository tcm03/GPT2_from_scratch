from attention import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Callable

class LayerNorm(nn.Module):

    def __init__(
        self,
        emb_dim: int
    ) -> None:
        super().__init__()
        self.scale = nn.Parameter(data = torch.ones(emb_dim))
        self.shift = nn.Parameter(data = torch.zeros(emb_dim))
        self.eps = 1e-5

    def forward(
        self,
        x: torch.Tensor
    ) -> None:
        x_mean = x.mean(dim = -1, keepdim = True)
        x_var = x.var(dim = -1, keepdim = True, unbiased = False)
        x_norm = (x - x_mean) / torch.sqrt(x_var + self.eps)
        x_ss = self.scale * x_norm + self.shift
        return x_ss
    
class FFN(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.GELU()
        )
    def forward(
        self,
        x: torch.Tensor
    ) -> None:
        assert x.shape[-1] == self.input_dim, "input dimension mismatch"
        return self.layers(x)
    
class TransformerBlock(nn.Module):

    def __init__(
        self,
        config: Dict[str, Any]
    ) -> None:
        super().__init__()
        self.embed_dim = config['emb_dim']
        self.layernorm1 = LayerNorm(emb_dim = self.embed_dim)
        self.layernorm2 = LayerNorm(emb_dim = self.embed_dim)
        self.mha = MultiHeadAttention(
            d_in = self.embed_dim,
            d_out = self.embed_dim,
            qkv_bias = config['qkv_bias'],
            max_context_length = config['context_length'],
            num_heads = config['n_heads'],
            dropout_rate = config['mha_drop_rate']
        )
        self.dropout = nn.Dropout(p = config['trf_drop_rate'])
        # inverted bottleneck
        self.ffn = FFN(
            input_dim = self.embed_dim,
            hidden_dim = 4 * self.embed_dim
        )


    def forward(
        self,
        x: torch.Tensor
    ):
        assert x.shape[-1] == self.embed_dim, "input dimension mismatch"
        x1 = self.layernorm1(x)
        x1 = self.mha(x1)
        x1 = self.dropout(x1)
        x1 = x1 + x # skip connection

        x2 = self.layernorm2(x1)
        x2 = self.ffn(x2)
        x2 = self.dropout(x2)
        o = x2 + x1
        return o
    
class GPTModel(nn.Module):
    def __init__(
        self,
        config: Dict[str, Any]
    ) -> None:
        super().__init__()

        self.vocab_size = config['vocab_size']
        self.embed_dim = config['emb_dim']
        self.context_length = config['context_length']
        self.dropout_rate = config['gpt_drop_rate']
        self.num_layers = config['n_layers']
        self.token_embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.pos_embed = nn.Embedding(self.context_length, self.embed_dim)
        self.dropout = nn.Dropout(p = self.dropout_rate)
        self.trf_blocks = nn.Sequential(*[
            TransformerBlock(config) for _ in range(self.num_layers)
        ])
        self.layernorm = LayerNorm(emb_dim = self.embed_dim)
        self.lm_head = nn.Linear(self.embed_dim, self.vocab_size, bias = False)

    def forward(
        self,
        text_ids: torch.Tensor
    ) -> None:
        assert text_ids.ndim == 2, "expected shape: (batch size, sequence length)"
        bs, seq_len = text_ids.shape
        text_embs = self.token_embed(text_ids)
        pos_embs = self.pos_embed(torch.arange(seq_len, device = text_ids.device))

        x = text_embs + pos_embs
        x = self.dropout(x)
        x = self.trf_blocks(x)
        x = self.layernorm(x)
        logits = self.lm_head(x)
        return logits