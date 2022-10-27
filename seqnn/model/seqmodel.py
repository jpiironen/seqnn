import numpy as np
import pandas as pd
import torch
import torch.utils.data
import pytorch_lightning as pl
from seqnn.model.core import ModelCore
from seqnn.model.data_handler import DataHandler
from seqnn.utils import get_cls


class SeqNNLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_core = ModelCore.create(config)
        self.data_handler = DataHandler(config)

    def forward(self, *args, **kwargs):
        return self.model_core(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = get_cls(self.config.optimizer.cls)(
            self.model_core.parameters(), **self.config.optimizer.args
        )
        scheduler = get_cls(self.config.scheduler.cls)(
            optimizer, **self.config.scheduler.args
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # past, future = batch
        # (
        #    target_past,
        #    target_future,
        #    control_past,
        #    control_future,
        # ) = self.data_handler.prepare_data(past, future, augment=True)
        # teacher_forcing = np.random.rand() < self.config.training.teacher_forcing_prob
        # output = self.forward(
        #    target_past,
        #    control_past,
        #    control_future,
        #    target_future=target_future,
        #    teacher_forcing=teacher_forcing,
        # )
        # loss = self.model_core.likelihood.get_loss(output, target_future)

        target, control = self.data_handler.prepare_data(batch, augment=True)
        losses = self.model_core.get_loss(target, control)
        loss = losses.mean()

        self.log("train_loss", loss.item())
        # self.log("train_loss", loss)
        # self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        target, control = self.data_handler.prepare_data(batch, augment=True)
        losses = self.model_core.get_loss(target, control)
        return losses

    def validation_epoch_end(self, outputs):
        losses = torch.cat(outputs, dim=0)
        self.log("valid_loss", losses.mean().item())


class SeqNN:
    def __init__(self, config):
        self.config = config
        self.model = SeqNNLightning(config)

    def train(
        self,
        data_train,
        data_valid=None,
        max_epochs=1,
        max_steps=-1,
        overfit_batches=0.0,
        dev_run=False,
    ):
        loader_train = self.data_to_loader(data_train, train=True)
        loader_valid = self.data_to_loader(data_valid)
        trainer = pl.Trainer(
            fast_dev_run=dev_run,
            gradient_clip_val=self.config.training.max_grad_norm,
            max_epochs=max_epochs,
            max_steps=max_steps,
            enable_progress_bar=True,
            log_every_n_steps=1,
            num_sanity_val_steps=-1,
            track_grad_norm=2,
            val_check_interval=self.config.training.validate_every_n_steps,
            check_val_every_n_epoch=None,
            overfit_batches=overfit_batches,
            limit_val_batches=0.0 if overfit_batches > 0 else 1.0,
        )
        trainer.fit(self.model, loader_train, loader_valid)

    def data_to_loader(self, data, train=False):
        if data is None:
            assert not train, "Training data cannot be None."
            return None
        if isinstance(data, torch.utils.data.DataLoader):
            return data
        if isinstance(data, torch.utils.data.Dataset):
            if train:
                if len(data) == 0:
                    raise RuntimeError("Training set has zero samples.")
                return torch.utils.data.DataLoader(
                    data,
                    batch_size=self.config.training.batch_size,
                    shuffle=True,
                    drop_last=True,
                )
            else:
                return torch.utils.data.DataLoader(
                    data,
                    batch_size=self.config.training.batch_size_valid,
                    shuffle=False,
                    drop_last=False,
                )
        if isinstance(data, (list, tuple)):
            return [self.data_to_loader(d, train=train) for d in data]
        if isinstance(data, pd.DataFrame):
            raise NotImplementedError

        raise NotImplementedError

    # def forward(self, *args, **kwargs):
    #    return self.model_core(*args, **kwargs)
