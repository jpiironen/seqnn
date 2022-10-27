import pathlib
import torch
import torch.nn as nn
import seqnn
from seqnn.model.transformer import GenerativeTransformer
from seqnn.utils import get_cls


class ModelCore(nn.Module):
    def __init__(
        self,
        likelihood,
        num_target,
        num_control,
        horizon_past,
        horizon_future,
    ):
        super().__init__()
        self.likelihood = likelihood
        self.num_target = num_target
        self.num_control = num_control
        self.num_output = self.get_num_outputs()
        self.horizon_past = horizon_past
        self.horizon_future = horizon_future

    @staticmethod
    def create(config):
        likelihood = config.get_likelihood()
        num_target = config.get_num_targets()
        num_control = config.get_num_controls()
        model = get_cls(config.model.cls)(
            likelihood,
            num_target,
            num_control,
            config.task.horizon_past,
            config.task.horizon_future,
            **config.model.args,
        )
        return model

    def save(self, path):
        path = pathlib.Path(path)
        if not path.exists():
            path.mkdir(parents=True)
        torch.save(self.state_dict(), path / "model_core.pt")

    def get_num_outputs(self):
        return self.likelihood.get_num_parameters() * self.num_target

    def forward(
        self, target_past, control_past, control_future, target_future=None, **kwargs
    ):
        raise NotImplementedError

    def get_loss(self, target, control):
        raise NotImplementedError


class RNN(ModelCore):
    def __init__(
        self,
        likelihood,
        num_target,
        num_control,
        horizon_past,
        horizon_future,
        model_type=nn.GRU,
        num_layers=2,
        hidden_size=128,
    ):
        super().__init__(
            likelihood,
            num_target,
            num_control,
            horizon_past,
            horizon_future,
        )
        self.model = model_type(
            input_size=num_control + num_target,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.readout = nn.Linear(hidden_size, self.num_output)

    def forward(
        self,
        target_past,
        control_past,
        control_future,
        target_future=None,
        teacher_forcing=False,
        **kwargs
    ):

        # batch_size, n_past, _ = target_past.shape
        _, n_future, _ = control_future.shape

        # encoding / burn-in
        x = torch.cat((control_past, target_past), dim=2)
        _, state = self.model(x)

        y = target_past[:, -1:, :]
        outputs_future = []
        for i in range(n_future):
            x_i = torch.cat((control_future[:, i : i + 1, :], y), dim=2)
            output, state = self.model(x_i, state)
            output = self.readout(output)
            if teacher_forcing:
                y = target_future[:, i : i + 1, :]
            else:
                y = self.likelihood.sample(output)
            outputs_future.append(output)

        return torch.cat(outputs_future, dim=1)


class Transformer(ModelCore):
    def __init__(
        self,
        likelihood,
        num_target,
        num_control,
        horizon_past,
        horizon_future,
        num_features=512,
        num_heads=8,
        num_blocks=4,
        num_hidden_ff=1024,
        dropout=0.1,
    ):
        assert isinstance(
            likelihood, seqnn.model.likelihood.LikCategorical
        ), "Transformer currently supports only categorical likelihood"
        super().__init__(
            likelihood, num_target, num_control, horizon_past, horizon_future
        )
        vocab_size = likelihood.get_num_parameters()
        max_seq_len = horizon_past + horizon_future
        self.model = GenerativeTransformer(
            vocab_size,
            max_seq_len,
            num_blocks=num_blocks,
            num_features=num_features,
            num_heads=num_heads,
            num_hidden_ff=num_hidden_ff,
            dropout=dropout,
        )

    def get_loss(self, target, control):
        tokens = target.squeeze(dim=2)
        pred = self.model(tokens)
        # prediction loss for every token (except the first one) conditioned on all preceding tokens
        losses = self.likelihood.get_loss(pred[:, :-1, :], tokens[:, 1:])
        return losses
