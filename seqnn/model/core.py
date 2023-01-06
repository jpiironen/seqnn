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

    def get_num_outputs(self):
        return self.likelihood.get_num_parameters() * self.num_target

    def has_controls(self):
        return self.num_control > 0

    def split_past_and_future(self, tensor):
        if tensor is None:
            return None, None
        return torch.split(tensor, (self.horizon_past, self.horizon_future), dim=1)

    def forward(
        self, target_past, control_past, control_future, target_future=None, **kwargs
    ):
        raise NotImplementedError

    def generate(
        self, target_past, control_past, control_future, sample=False, **kwargs
    ):
        raise NotImplementedError

    def get_loss(
        self,
        target_past,
        control_past,
        target_future,
        control_future,
        aux_past=None,
        teacher_forcing=False,
    ):
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

    def generate(
        self, target_past, control_past, control_future, sample=False, **kwargs
    ):
        return self(
            target_past, control_past, control_future, target_future=None, **kwargs
        )


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


class CategoricalTransformer(ModelCore):
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
        # TODO: this function no longer satisfies the super class method signature
        tokens = target.squeeze(dim=2)
        pred = self.model(tokens)
        # prediction loss for every token (except the first one) conditioned on all preceding tokens
        losses = self.likelihood.get_loss(pred[:, :-1, :], tokens[:, 1:])
        return losses


class Transformer(ModelCore):
    def __init__(
        self,
        likelihood,
        num_target,
        num_control,
        horizon_past,
        horizon_future,
        sequence_dims=True,
        num_features=512,
        num_heads=8,
        num_blocks=4,
        num_hidden_ff=1024,
        dropout=0.1,
        learn_pos_encoding=True,
    ):
        assert (
            horizon_future == 1
        ), "To train Transformer, one must set horizon_future = 1"
        super().__init__(
            likelihood, num_target, num_control, horizon_past, horizon_future
        )
        self.num_features = num_features
        self.sequence_dims = sequence_dims

        if self.sequence_dims:
            max_seq_len = horizon_past * (self.num_target + self.num_control)
        else:
            max_seq_len = horizon_past * (1 + self.has_controls())
        self.model = GenerativeTransformer(
            max_seq_len,
            num_blocks=num_blocks,
            num_features=num_features,
            num_heads=num_heads,
            num_hidden_ff=num_hidden_ff,
            dropout=dropout,
            learn_pos_encoding=learn_pos_encoding,
        )
        self.embedding_target, self.embedding_control = self.create_embeddings()
        self.readout_target, self.readout_control = self.create_readouts()

    def create_embeddings(self):
        if self.likelihood.is_discrete():
            embedding_target = nn.Embedding(
                self.likelihood.get_num_parameters(), self.num_features
            )
            # TODO: implement control embedding
            embedding_control = None
        else:
            if self.sequence_dims:
                embedding_target = nn.ModuleList(
                    [nn.Linear(1, self.num_features) for _ in range(self.num_target)]
                )
                embedding_control = nn.ModuleList(
                    [nn.Linear(1, self.num_features) for _ in range(self.num_control)]
                )
            else:
                embedding_target = nn.Linear(self.num_target, self.num_features)
                embedding_control = (
                    nn.Linear(self.num_control, self.num_features)
                    if self.has_controls()
                    else None
                )
        return embedding_target, embedding_control

    def create_readouts(self):
        num_outputs_per_target = self.likelihood.get_num_parameters()
        if self.sequence_dims:
            readout_target = nn.ModuleList(
                [
                    nn.Linear(self.num_features, num_outputs_per_target)
                    for _ in range(self.num_target)
                ]
            )
            readout_control = nn.ModuleList(
                [
                    # TODO: this is not quite appropriate if we have categorical controls
                    nn.Linear(self.num_features, num_outputs_per_target)
                    for _ in range(self.num_control)
                ]
            )
        else:
            num_target_outputs_total = num_outputs_per_target * self.num_target
            num_control_outputs_total = num_outputs_per_target * self.num_control
            readout_target = nn.Linear(self.num_features, num_target_outputs_total)
            if self.num_control > 0:
                # TODO: this is not quite appropriate if we have categorical controls
                readout_control = nn.Linear(
                    self.num_features, num_control_outputs_total
                )
            else:
                readout_control = []
        return readout_target, readout_control

    def variables_to_sequence(self, target, control, control_last=False, embed=True):
        batch_size = target.shape[0]
        if self.sequence_dims:
            # given m-dimensional control u, and n-dimensional target y, this will sequence everything as
            # (..., u_t[1], u_t[2], ..., u_t[m], y_t[1], y_t[2], ..., y_t[n], ...) or
            # (..., y_t[1], y_t[2], ..., y_t[n], u_t[1], u_t[2], ..., u_t[m], ...) if control_last == True
            target = [
                embedding(target[:, :, k : k + 1]) if embed else target[:, :, k : k + 1]
                for k, embedding in enumerate(self.embedding_target)
            ]
            target = torch.stack(target, dim=-1)
            if self.has_controls():
                control = [
                    embedding(control[:, :, k : k + 1])
                    if embed
                    else control[:, :, k : k + 1]
                    for k, embedding in enumerate(self.embedding_control)
                ]
                control = torch.stack(control, dim=-1)
                vars = (target, control) if control_last else (control, target)
                vars = torch.cat(vars, dim=3)
            else:
                vars = target
            last_dim = self.num_features if embed else 1
            sequence = vars.transpose(2, 3).contiguous().view(batch_size, -1, last_dim)
        else:
            target = self.embedding_target(target) if embed else target
            if self.has_controls():
                control = self.embedding_control(control) if embed else control
                vars = (target, control) if control_last else (control, target)
                vars = torch.cat(vars, dim=2)
            else:
                vars = target
            last_dim = self.num_features if embed else 1
            sequence = vars.view(batch_size, -1, last_dim)

        return sequence

    def read_output(self, sequence, control_last=False, offset=0):
        batch_size = sequence.shape[0]
        if self.sequence_dims:
            # combine readouts into a list and order them appropriately so that each output representation
            # will be processed by correct readout layer.
            #
            # if control_last == False and offset == 0,
            # then the input sequence is assumed to be
            #   u_t[1], u_t[2], ..., u_t[m], y_t[1], y_t[2] ..., y_t[n],
            # and if offset == 1,
            #   u_t[2], ..., u_t[m], y_t[1], y_t[2] ..., y_t[n], u_{t+1}[1]
            # etc.
            #
            # if control_last == True and offset 0,  the input sequence is assumed to be
            #   y_t[1], y_t[2] ..., y_t[n], u_t[1], u_t[2], ..., u_t[m]
            # and if offset == 1,
            #   y_t[2] ..., y_t[n], u_t[1] u_t[2], ..., u_t[m], y_{t+1}[1]
            # etc.
            #
            if control_last:
                readouts_all = list(self.readout_target) + list(self.readout_control)
            else:
                readouts_all = list(self.readout_control) + list(self.readout_target)
            # here +1 comes from the fact that the prediction is always for the next symbol in the sequence
            readouts_all = readouts_all[1 + offset :] + readouts_all[: 1 + offset]
            seq_reshaped = sequence.view(
                batch_size, -1, self.num_target + self.num_control, self.num_features
            )
            readouts = [
                readout(seq_reshaped[:, :, k, :])
                for k, readout in enumerate(readouts_all)
            ]
            seq_output = torch.stack(readouts, dim=-1)
            seq_output = seq_output.transpose(2, 3)
            seq_output = seq_output.reshape(batch_size, -1, seq_output.shape[3])
        else:
            if control_last:
                readouts_all = [self.readout_target] + ensure_list(self.readout_control)
            else:
                readouts_all = ensure_list(self.readout_control) + [self.readout_target]
            readouts_all = readouts_all[1 + offset :] + readouts_all[: 1 + offset]
            seq_reshaped = sequence.view(
                batch_size, -1, 1 + (self.num_control > 0), self.num_features
            )
            readouts = [
                readout(seq_reshaped[:, :, k, :])
                for k, readout in enumerate(readouts_all)
            ]
            seq_output = torch.stack(readouts, dim=-1)
            seq_output = seq_output.transpose(2, 3)
            seq_output = seq_output.reshape(batch_size, -1, seq_output.shape[3])
        return seq_output

    def get_loss(
        self,
        target_past,
        control_past,
        target_future,
        control_future,
        aux_past=None,
        teacher_forcing=False,
    ):
        target_all = torch.cat((target_past, target_future), dim=1)
        control_all = torch.cat((control_past, control_future), dim=1)
        if self.sequence_dims:
            offset = torch.randint(self.num_target + self.num_control, (1,)).item()
        else:
            offset = torch.randint(1 + (self.num_control > 0), (1,)).item()
        n = self.model.get_memory_length()
        x_all = self.variables_to_sequence(target_all, control_all)
        y_all = self.variables_to_sequence(target_all, control_all, embed=False)

        # prediction loss for every token conditioned on all preceding tokens
        # (notice that the first prediction is for the second token in the sequence)
        # TODO: should we ignore the prediction loss for the controls?
        x = x_all[:, offset : offset + n, :]
        y = y_all[:, 1 + offset : 1 + offset + n]
        output = self.model(x)
        pred = self.read_output(output, offset)
        losses = self.likelihood.get_loss(pred, y)
        return losses

    def generate(
        self, target_past, control_past, control_future, sample=False, **kwargs
    ):
        control_all = torch.cat((control_past, control_future), dim=1)
        n_future = control_future.shape[1]
        n_past = control_past.shape[1]
        x = self.variables_to_sequence(
            target_past, control_all[:, 1 : n_past + 1, :], control_last=True
        )
        memory_length = self.model.get_memory_length()

        preds = []
        targets_generated = []
        for i in range(n_future):

            if self.sequence_dims:

                # target_next = []
                pred = []
                for k in range(self.num_target):
                    # generate target variables one at a time
                    x_crop = (
                        x if x.shape[1] <= memory_length else x[:, -memory_length:, :]
                    )
                    output = self.model(x_crop)
                    pred_k = self.read_output(output, control_last=True, offset=k)[
                        :, -1:, :
                    ]
                    pred.append(pred_k)
                    if sample:
                        target_k_next = self.likelihood.sample(pred_k)
                    else:
                        target_k_next = self.likelihood.most_probable(pred_k)

                    x_next = self.embedding_target[k](target_k_next)
                    x = torch.cat((x, x_next), dim=1)
                pred = torch.stack(pred, dim=-1).transpose(2, 3)
                preds.append(pred)

                # add controls into the sequence one-by-one
                if i + 1 < n_future:
                    control_next = control_future[:, i + 1 : i + 2, :]
                    for k in range(self.num_control):
                        x_next = self.embedding_control[k](
                            control_next[:, :, k : k + 1]
                        )
                        x = torch.cat((x, x_next), dim=1)
            else:

                # if the sequence length is too large, we need to crop it to the memory length of the model
                x_crop = x if x.shape[1] <= memory_length else x[:, -memory_length:, :]
                output = self.model(x_crop)
                pred = self.read_output(output, control_last=True)[:, -1:, :]
                preds.append(pred)

                if sample:
                    target_next = self.likelihood.sample(pred)
                else:
                    target_next = self.likelihood.most_probable(pred)
                targets_generated.append(target_next)

                if i + 1 < n_future:
                    x_next = self.variables_to_sequence(
                        target_next,
                        control_future[:, i + 1 : i + 2, :],
                        control_last=True,
                    )
                    x = torch.cat((x, x_next), dim=1)

        if sample:
            return torch.cat(targets_generated, dim=1)
        return torch.cat(preds, dim=1)
