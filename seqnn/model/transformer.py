import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        seq_len_max,
        num_heads=4,
        dim_embed=64,
        dropout_attn=0.0,
        dropout_resid=0.0,
    ):
        super().__init__()

        # key, query, value projections for all heads
        hidden_size = dim_embed * num_heads
        self.key = nn.Linear(hidden_size, hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        # regularization
        self.dropout_attn = nn.Dropout(dropout_attn)
        self.dropout_resid = nn.Dropout(dropout_resid)
        # output projection
        self.proj = nn.Linear(hidden_size, hidden_size)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        m = seq_len_max
        self.register_buffer("mask", torch.tril(torch.ones(m, m)).view(1, 1, m, m))
        self.num_heads = num_heads

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim;
        # all of these tensors will have size (B, num_heads, T, dim_embed)
        dim_embed = hidden_size // self.num_heads
        k = (
            self.key(x)
            .view(batch_size, seq_len, self.num_heads, dim_embed)
            .transpose(1, 2)
        )
        q = (
            self.query(x)
            .view(batch_size, seq_len, self.num_heads, dim_embed)
            .transpose(1, 2)
        )
        v = (
            self.value(x)
            .view(batch_size, seq_len, self.num_heads, dim_embed)
            .transpose(1, 2)
        )

        # causal self-attention
        #  (B, Nh, T, D) x (B, Nh, D, T) -> (B, Nh, T, T)
        # where:
        #  D = dim_embed
        #  Nh = num_heads
        #  B = batch_size
        #  T = seq_len
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        att = att.softmax(dim=-1)
        att = self.dropout_attn(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)

        # output projection
        y = self.dropout_resid(self.proj(y))
        return y


class TransformerBlock(nn.Module):
    """ Transformer block """

    def __init__(
        self,
        seq_len_max,
        dim_embed=64,
        num_heads=4,
        layer_norm_last=True,
        dropout_resid=0.0,
        dropout_attn=0.0,
    ):
        super().__init__()
        hidden_size = dim_embed * num_heads
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.attn = CausalSelfAttention(
            seq_len_max,
            num_heads=num_heads,
            dim_embed=dim_embed,
            dropout_resid=dropout_resid,
            dropout_attn=dropout_attn,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.ReLU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout_resid),
        )
        self.layer_norm_last = layer_norm_last

    def forward(self, x):
        if self.layer_norm_last:
            x = self.ln1(x + self.attn(x))
            x = self.ln2(x + self.mlp(x))
        else:
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        seq_len_max,
        continuous_tokens=False,
        vocab_size=None,
        num_blocks=4,
        dim_embed=64,
        num_heads=4,
        layer_norm_last=True,
        output_layer_bias=False,
        dropout_embed=0.0,
        dropout_resid=0.0,
        dropout_attn=0.0,
    ):
        super().__init__()
        self.continuous_tokens = continuous_tokens
        # input embedding
        hidden_size = num_heads * dim_embed
        if continuous_tokens:
            vocab_size = 1
            self.tok_emb = nn.Linear(1, hidden_size)
        else:
            assert (
                vocab_size is not None
            ), "vocab_size must be provided unless continuous_tokens = True"
            self.tok_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len_max, hidden_size))
        self.dropout = nn.Dropout(dropout_embed)
        # transformer
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    seq_len_max,
                    dim_embed=dim_embed,
                    num_heads=num_heads,
                    layer_norm_last=layer_norm_last,
                    dropout_resid=dropout_resid,
                    dropout_attn=dropout_attn,
                )
                for _ in range(num_blocks)
            ]
        )
        # decoder head
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=output_layer_bias)
        self.block_size = seq_len_max

    def forward(self, tokens):
        batch_size, seq_len = tokens.size()
        assert (
            seq_len <= self.block_size
        ), "Cannot forward, model block size is exhausted."

        if self.continuous_tokens:
            token_embeddings = self.tok_emb(tokens.view(*tokens.shape, 1))
        else:
            token_embeddings = self.tok_emb(tokens)
        position_embeddings = self.pos_emb[:, :seq_len, :]
        #position_embeddings = self.pos_emb[:, -seq_len:, :] # would this be better? not sure..

        x = self.dropout(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        output = self.head(x)
        return output

    def unroll(self, x, steps, sample=False, full_sequence=False, temperature=1.0):
        """
        take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
        the sequence, feeding the predictions back into the model each time. Clearly the sampling
        has quadratic complexity unlike an RNN that is only linear, and has a finite context window
        of block_size, unlike an RNN that has an infinite context window.
        """
        if self.continuous_tokens:
            assert x.ndim == 3
        else:
            assert x.ndim == 2

        for k in range(steps):

            # crop sequence length if needed
            x_cond = x if x.size(1) <= self.block_size else x[:, -self.block_size :]
            output = self(x_cond)

            if self.continuous_tokens:
                if sample:
                    raise NotImplementedError
                else:
                    x = torch.cat((x, output[:, -1, :]), dim=1)
            else:
                if sample:
                    logits = output[:, -1, :] / temperature
                    # optionally crop probabilities to only the top k options
                    # if top_k is not None:
                    #    logits = top_k_logits(logits, top_k)
                    probs = torch.softmax(logits, dim=-1)
                    ix = torch.multinomial(probs, num_samples=1)
                    x = torch.cat((x, ix), dim=1)
                else:
                    raise NotImplementedError
            
        if full_sequence:
            return x
        else:
            return x[:, -steps:]
