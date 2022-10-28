import pathlib
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import pytorch_lightning as pl
from seqnn import SeqNNConfig
from seqnn.data.dataset import CombinationDataset
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
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        target, control = self.data_handler.prepare_data(batch, augment=True)
        teacher_forcing = np.random.rand() < self.config.training.teacher_forcing_prob
        losses = self.model_core.get_loss(target, control, teacher_forcing=teacher_forcing)
        loss = losses.mean()

        self.log("train_loss", loss.item())
        # self.log("train_loss", loss)
        # self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        target, control = self.data_handler.prepare_data(batch, augment=False)
        losses = self.model_core.get_loss(target, control, teacher_forcing=False)
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
            raise NotImplementedError
            # TODO: this would not work because data_to_loader returns a loader...
            # return CombinationDataset(
            #    [self.data_to_loader(d, train=train) for d in data]
            # )
        if isinstance(data, pd.DataFrame):
            dataset = DataHandler.df_to_dataset(data, self.config)
            return self.data_to_loader(dataset, train=train)

        raise NotImplementedError

    def save(self, dir):
        dir = pathlib.Path(dir)
        self.config.save(dir / "config.yaml")
        self.model.model_core.save_state(dir)

    @staticmethod
    def load(dir):
        dir = pathlib.Path(dir)
        config = SeqNNConfig.load(dir / "config.yaml")
        model = SeqNN(config)
        model.model.model_core.load_state(dir)
        return model
