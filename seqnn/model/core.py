import pathlib
import torch
import torch.nn as nn
import seqnn
from seqnn.model.basemodels import MLP, CNN1d
from seqnn.model.transformer import GenerativeTransformer
from seqnn.utils import get_cls, ensure_list


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

    def save_state(self, dir):
        dir = pathlib.Path(dir)
        if not dir.exists():
            dir.mkdir(parents=True)
        torch.save(self.state_dict(), dir / "model_core.pt")

    def load_state(self, dir):
        dir = pathlib.Path(dir)
        self.load_state_dict(torch.load(dir / "model_core.pt"))

    def get_num_outputs(self):
        return self.likelihood.get_num_parameters() * self.num_target

    def split_past_and_future(self, tensor):
        if tensor is None:
            return None, None
        return torch.split(tensor, (self.horizon_past, self.horizon_future), dim=1)

    def forward(
        self, target_past, control_past, control_future, target_future=None, **kwargs
    ):
        raise NotImplementedError

    def get_loss(self, target, control, aux=None, teacher_forcing=False):
        target_past, target_future = self.split_past_and_future(target)
        control_past, control_future = self.split_past_and_future(control)
        aux_past, _ = self.split_past_and_future(aux)
        output = self(
            target_past,
            control_past,
            control_future,
            aux_past=aux_past,
            target_future=target_future,
            teacher_forcing=teacher_forcing,
        )
        losses = self.likelihood.get_loss(output, target_future)
        return losses


class NLDS(ModelCore):
    """Non-linear dynamical system (NLDS), where the intial latent state
    is estimated using a CNN, and dynamics are computed via an MLP"""

    def __init__(
        self,
        likelihood,
        num_target,
        num_control,
        horizon_past,
        horizon_future,
        num_auxiliary=0,
        num_filters=[64, 64],
        num_hidden_dynamics=[128],
        num_hidden_readout=[128],
        kernel_size=3,
        latent_size=32,
        dropout=0.1,
        dropout_latent=0.0,
        act=nn.ReLU(),
    ):
        super().__init__(
            likelihood,
            num_target,
            num_control,
            horizon_past,
            horizon_future,
        )
        num_filters = ensure_list(num_filters)
        num_hidden_dynamics = ensure_list(num_hidden_dynamics)
        num_hidden_readout = ensure_list(num_hidden_readout)
        seq_len_enc = horizon_past
        self.encoder = CNN1d(
            seq_len=seq_len_enc,
            conv_sizes=[num_control + num_target + num_auxiliary] + num_filters,
            fc_sizes=[latent_size],
            kernel_size=kernel_size,
            act=act,
            dropout=dropout,
        )
        self.dynamics_model = MLP(
            [latent_size + num_control] + num_hidden_dynamics + [latent_size],
            dropout=dropout,
            act=act,
        )
        self.dropout_latent = nn.Dropout(dropout_latent)
        self.readout_model = MLP(
            [latent_size] + num_hidden_readout + [self.num_output],
            dropout=dropout,
            act=act,
        )

    def encode(self, control_past, target_past, aux_past=None):
        # encode the past sequences using CNN to estimate the current latent state
        if aux_past is not None:
            x = torch.cat((target_past, control_past, aux_past), dim=2)
        else:
            x = torch.cat((target_past, control_past), dim=2)
        x = x.permute(0, 2, 1)
        state = self.encoder(x)
        return state

    def forward_one_step(self, state, control):
        assert state.ndim == 2
        assert control.ndim == 2
        x = torch.cat((state, control), dim=1)
        return state + self.dynamics_model(x)

    def forward(
        self,
        target_past,
        control_past,
        control_future,
        target_future=None,
        aux_past=None,
        **kwargs
    ):
        # encode the past, and unroll the predictions about how the
        # hidden state evolves, and finally map the hidden states to target predictions
        state = self.encode(control_past, target_past, aux_past)
        control_all = torch.cat((control_past, control_future), dim=1)
        batch_size, horizon_future, num_control = control_future.shape
        hidden_states = []
        for i in range(horizon_future):
            control = control_all[
                :,
                self.horizon_past + i : self.horizon_past + i + 1,
                :,
            ].view(batch_size, -1)
            state = self.forward_one_step(state, control)
            hidden_states.append(state.view(batch_size, 1, -1))
        hidden_states = torch.cat(hidden_states, dim=1)
        hidden_states = self.dropout_latent(hidden_states)
        output = self.readout_model(hidden_states)
        return output


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
        dropout=0.1,
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
        self.dropout = nn.Dropout(dropout)
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
            output = self.readout(self.dropout(output))
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

    def get_loss(self, target, control, **kwargs):
        tokens = target.squeeze(dim=2)
        pred = self.model(tokens)
        # prediction loss for every token (except the first one) conditioned on all preceding tokens
        losses = self.likelihood.get_loss(pred[:, :-1, :], tokens[:, 1:])
        return losses
