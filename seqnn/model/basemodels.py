import torch
import torch.nn as nn
import copy
from seqnn.utils import ensure_list


class MLP(nn.Module):
    """
    Multilayer perceptron with an arbitrary number of layers.
    """

    def __init__(
        self,
        sizes,
        dropout=0.0,
        dropout_input=0.0,
        act=nn.ReLU(),
        dropout_before_act=True,
    ):
        super(MLP, self).__init__()
        self.sizes = sizes
        self.n_inputs = sizes[0]
        self.n_outputs = sizes[-1]
        nlayers = len(sizes)

        layers = []
        layers.append(nn.Dropout(dropout_input))
        for i in range(nlayers - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < nlayers - 2:
                if dropout_before_act:
                    layers.append(nn.Dropout(dropout))
                    layers.append(act)
                else:
                    layers.append(act)
                    layers.append(nn.Dropout(dropout))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CNN1d(nn.Module):
    """
    1D convolutional net, applies convolution along the last dimension of the given tensor
    """

    def __init__(
        self,
        seq_len,
        conv_sizes,
        fc_sizes,
        kernel_size=3,
        dropout=0.0,
        dropout_between=None,
        dropout_before_act=True,
        act=nn.ReLU(),
    ):
        super().__init__()
        if dropout_between is None:
            dropout_between = dropout
        conv_sizes = ensure_list(conv_sizes)
        fc_sizes = ensure_list(fc_sizes)
        assert len(conv_sizes) > 1
        num_conv_layers = len(conv_sizes) - 1
        padding = int((kernel_size - 1) / 2)
        layers = []

        for i in range(num_conv_layers):
            layer = nn.Conv1d(
                conv_sizes[i],
                conv_sizes[i + 1],
                kernel_size=kernel_size,
                padding=padding,
            )
            layers.append(layer)
            layers.append(act)
        self.conv_layers = nn.ModuleList(layers)
        self.fc = MLP(
            [seq_len * conv_sizes[-1]] + fc_sizes,
            dropout=dropout,
            dropout_input=dropout_between,
            dropout_before_act=dropout_before_act,
        )
        self.seq_len = seq_len

    def forward(self, x):
        assert (
            x.shape[-1] == self.seq_len
        ), "Last input dimension must be the sequence length"
        batch_size = x.shape[0]
        for layer in self.conv_layers:
            x = layer(x)
        x = x.reshape(batch_size, -1)
        out = self.fc(x)
        return out
