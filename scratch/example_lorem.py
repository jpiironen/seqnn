import numpy as np
import torch

from seqnn import SeqNNConfig, SeqNN
from seqnn.data.dataset import CharacterMap, DictSeqDataset

# load some example data
with open("data/lorem/sample1.txt", "r") as file:
    text = ""
    for line in file:
        text += line

character_map = CharacterMap(text)
vocab_size = character_map.get_vocab_size()
tokens = character_map.to_indices(text)

# truncate dataset size for faster testing
tokens = tokens[:1000]

# create dataset
n_valid = 500
data_train = DictSeqDataset(
    {"token": torch.tensor(tokens[:-n_valid]).view(-1, 1)}, seq_len=128
)
data_valid = DictSeqDataset(
    {"token": torch.tensor(tokens[-n_valid:]).view(-1, 1)}, seq_len=128
)


# setup model
config = SeqNNConfig(
    targets="token",
    controls=None,
    horizon_past=128,
    horizon_future=0,
    model="Transformer",
    likelihood="LikCategorical",
    likelihood_args={"num_classes": vocab_size},
    optimizer="Adam",
    optimizer_args={"lr": 0.001},
    validate_every_n_steps=5,
)
model = SeqNN(config)

# train
model.train(data_train, data_valid, overfit_batches=1, max_epochs=50)
