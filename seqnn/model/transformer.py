import numpy as np
import torch
import torch.nn as nn


class GenerativeTransformerBlock(nn.Module):
    def __init__(
        self, max_seq_len, num_features, num_heads, num_hidden_mlp=512, dropout=0.1
    ):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            num_features, num_heads, dropout=dropout, batch_first=True
        )
        self.create_attention_mask(max_seq_len)
        self.mlp = nn.Sequential(
            nn.Linear(num_features, num_hidden_mlp),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_hidden_mlp, num_features),
        )
        self.ln1 = nn.LayerNorm(num_features)
        self.ln2 = nn.LayerNorm(num_features)

    def create_attention_mask(self, max_seq_len):
        m = max_seq_len
        not_allowed = torch.triu(torch.ones(m, m)) - torch.diag(torch.ones(m))
        self.register_buffer("attention_mask", not_allowed > 0)

    def forward(self, x):
        seq_len = x.shape[1]
        mask = self.attention_mask[:seq_len, :seq_len]
        output, _ = self.self_attention(x, x, x, attn_mask=mask, need_weights=False)
        x = self.ln1(x + output)
        x = self.ln2(x + self.mlp(x))
        return x


class GenerativeTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        num_blocks=4,
        num_features=256,
        num_heads=4,
        num_hidden_mlp=1024,
        dropout=0.1,
    ):
        """
        Args:
          vocab_size: Number of tokens in the vocabulary.
          n_blocks: Number of EncoderBlock blocks.
          n_features: Number of features to be used for word embedding and further in all layers of the decoder.
          n_heads: Number of attention heads inside the DecoderBlock.
          n_hidden: Number of hidden units in the Feedforward block of DecoderBlock.
          dropout: Dropout level used in DecoderBlock.
        """
        super().__init__()
        # max_tgt_len = MAX_LENGTH + 1 # +1 due to SOS token
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=num_features
        )
        self.positional_encoding = PositionalEncoding(num_features, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                GenerativeTransformerBlock(
                    max_seq_len,
                    num_features,
                    num_heads,
                    num_hidden_mlp=num_hidden_mlp,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.readout = nn.Linear(num_features, vocab_size)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        logits = self.readout(x)
        return logits
        # return self.log_softmax(logits)


class PositionalEncoding(nn.Module):
    """
    This implementation is the same as in the Annotated transformer blog post
    See https://nlp.seas.harvard.edu/2018/04/03/attention.html for more detail.
    """

    def __init__(self, num_features, max_len):
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

    def forward(self, x):
        return x + self.positional_encoding[:, : x.size(1), :]
