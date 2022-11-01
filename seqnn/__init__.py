from seqnn.config import SeqNNConfig
from seqnn.model.seqmodel import SeqNN


def load(path):
    # just a convenient interface to model loading
    return SeqNN.load(path)
