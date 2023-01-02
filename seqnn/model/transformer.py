import numpy as np
import torch
import torch.nn as nn


class GenerativeTransformerBlock(nn.Module):
    def __init__(
        self,
        max_seq_len,
        num_features,
        num_heads,
        num_hidden_ff=1024,
        dropout=0.1,
        layernorm_last=False,
    ):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            num_features, num_heads, dropout=dropout, batch_first=True
        )
        self.create_attention_mask(max_seq_len)
        self.mlp = nn.Sequential(
            nn.Linear(num_features, num_hidden_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_hidden_ff, num_features),
        )
        self.ln1 = nn.LayerNorm(num_features)
        self.ln2 = nn.LayerNorm(num_features)
        self.layernorm_last = layernorm_last

    def create_attention_mask(self, max_seq_len):
        m = max_seq_len
        allowed = torch.tril(torch.ones(m, m)) > 0
        self.register_buffer("attention_mask", ~allowed)

    def forward(self, x):
        seq_len = x.shape[1]
        mask = self.attention_mask[:seq_len, :seq_len]
        if self.layernorm_last:
            # the original formulation with layer norm applied after attention/mlp
            output, _ = self.self_attention(x, x, x, attn_mask=mask, need_weights=False)
            x = self.ln1(x + output)
            x = self.ln2(x + self.mlp(x))
        else:
            # alternative formulation where the layer norm is applied 'inside' the residual block
            x = self.ln1(x)
            output, _ = self.self_attention(x, x, x, attn_mask=mask, need_weights=False)
            x = x + output
            x = x + self.mlp(self.ln2(x))
        return x


class GenerativeTransformer(nn.Module):
    def __init__(
        self,
        max_seq_len,
        num_features=512,
        num_heads=8,
        num_blocks=4,
        num_hidden_ff=1024,
        dropout=0.1,
        layernorm_last=False,
        learn_pos_encoding=True,
        reverse_pos_encoding=False,
    ):
        """
        Args:
            vocab_size: Number of tokens in the vocabulary.
            max_seq_len: Maximum sequence length (i.e. size of the look-back memory) of the model.
            num_features: Number of features to be used for word embedding and further in all layers of the decoder.
                Must be divisible by num_heads.
            num_heads: Number of attention heads inside the transformer block.
            num_blocks: Number of transformer blocks.
            num_hidden_: Number of hidden units in the feedforward layer of the transformer block.
            dropout: Dropout level used in the transformer block.
        """
        super().__init__()
        self.num_features = num_features
        if learn_pos_encoding:
            self.positional_encoding = PositionalEncodingLearnable(
                num_features, max_seq_len, reverse=reverse_pos_encoding
            )
        else:
            self.positional_encoding = PositionalEncoding(
                num_features, max_seq_len, reverse=reverse_pos_encoding
            )
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                GenerativeTransformerBlock(
                    max_seq_len,
                    num_features,
                    num_heads,
                    num_hidden_ff=num_hidden_ff,
                    dropout=dropout,
                    layernorm_last=layernorm_last,
                )
                for _ in range(num_blocks)
            ]
        )
        self.ln = nn.LayerNorm(num_features)

    def forward(self, x):
        assert x.ndim == 3
        assert x.shape[2] == self.num_features
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        return x

    # @torch.no_grad()
    # def generate(self, x, max_new_tokens, temperature=1.0, sample=False, top_k=None):
    #    '''
    #    This function is a modified version of
    #    https://github.com/karpathy/minGPT/blob/7218bcfa527c65f164de791099de715b81a95106/mingpt/model.py#L283
    #    '''
    #    for _ in range(max_new_tokens):
    #        # if the sequence length is too large, we need to crop it to the memory length of the model
    #        x_crop = x if x.shape[1] <= self.memory_len else x[:, -self.memory_len :]
    #        logits = self(x_crop)
    #        logits = logits[:, -1, :] / temperature
    #        if top_k is not None:
    #            v, _ = torch.topk(logits, top_k)
    #            logits[logits < v[:, [-1]]] = -float("Inf")
    #        probs = torch.nn.functional.softmax(logits, dim=-1)
    #        if sample:
    #            x_next = torch.multinomial(probs, num_samples=1)
    #        else:
    #            _, x_next = torch.topk(probs, k=1, dim=-1)
    #        x = torch.cat((x, x_next), dim=1)
    #    return x


class PositionalEncoding(nn.Module):
    """
    This implementation is the same as in the Annotated transformer blog post
    See https://nlp.seas.harvard.edu/2018/04/03/attention.html for more detail.
    """

    def __init__(self, num_features, max_len, reverse=False):
        super().__init__()
        assert (num_features % 2) == 0, "num_features should be an even number."
        pe = torch.zeros(max_len, num_features)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, num_features, 2).float() * (-np.log(10000.0) / num_features)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("positional_encoding", pe)
        self.reverse = reverse

    def forward(self, x):
        seq_len = x.size(1)
        if self.reverse:
            return (x.flip(1) + self.positional_encoding[:, :seq_len, :]).flip(1)
        return x + self.positional_encoding[:, :seq_len, :]


class PositionalEncodingLearnable(nn.Module):
    def __init__(self, num_features, max_len, scale_init=0.1, reverse=False):
        super().__init__()
        assert (num_features % 2) == 0, "num_features should be an even number."
        self.positional_encoding = torch.nn.Embedding(max_len, num_features)
        self.reverse = reverse

    def forward(self, x):
        seq_len = x.size(1)
        # TODO: use same device as x
        position = torch.arange(0, seq_len, dtype=torch.long, device="cpu").unsqueeze(0)
        encoding = self.positional_encoding(position)
        return x + encoding
        # TODO: implement the reverse logic
        # if self.reverse:
        #    return (x.flip(1) + self.positional_encoding[:, : x.size(1), :]).flip(1)
        # return x + self.positional_encoding[:, : x.size(1), :]


# def get_relative_positional_encoding(length1:int, length2:int, d_model:int, device:torch.device):
#    xs = torch.arange(length1, device=device).unsqueeze(1)
#    ys = torch.arange(length2, device=device).unsqueeze(0)
#
#    position = ys - xs
#    div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
#    angle = position.unsqueeze(-1) * div_term.view(1, 1, -1)
#    positional_encoding = torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1)
#
#    return positional_encoding / math.sqrt(d_model)
