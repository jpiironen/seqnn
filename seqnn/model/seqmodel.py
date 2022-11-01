import pathlib
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import pytorch_lightning as pl

import seqnn.data.scalers
from seqnn.config import SeqNNConfig
from seqnn.data.dataset import CombinationDataset
from seqnn.model.core import ModelCore
from seqnn.model.data_handler import DataHandler
from seqnn.utils import get_cls, save_torch_state, load_torch_state


class SeqNNLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        pl.seed_everything(self.config.training.seed)
        self.model_core = ModelCore.create(config)
        self.scaler = self.create_scaler(config)
        self.data_handler = DataHandler(config)
        self.save_hyperparameters(config.to_dict())

    def create_scaler(self, config):
        scalers = {}
        for group, scaler_spec in config.scalers.groups.items():
            cls_name = scaler_spec["cls"]
            args = scaler_spec["args"]
            ndim = len(config.task.grouping[group])
            scalers[group] = get_cls(cls_name)(ndim, **args)
        return seqnn.data.scalers.ScalerCollection(scalers)

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

    def zero_scaler_grad(self):
        # zero out the gradient of the scaler parameters during training, so as to
        # not affect the gradient statistics (will not have effect on the training since
        # the scaler parameters are not optimized)
        for p in self.scaler.parameters():
            if p.grad is not None:
                p.grad.zero_()

    def to_scaled(self, seq):
        return self.scaler.to_scaled(seq)

    def to_native(self, seq):
        return self.scaler.to_native(seq)

    def training_step(self, batch, batch_idx):
        past, future = batch
        past = self.to_scaled(past)
        future = self.to_scaled(future)
        (
            target_past,
            control_past,
            target_future,
            control_future,
        ) = self.data_handler.prepare_data(past, future, augment=True)
        teacher_forcing = np.random.rand() < self.config.training.teacher_forcing_prob
        losses = self.model_core.get_loss(
            target_past,
            control_past,
            target_future,
            control_future,
            teacher_forcing=teacher_forcing,
        )
        loss = losses.mean()
        self.log("train_loss", loss.item())
        self.zero_scaler_grad()
        return loss

    def validation_step(self, batch, batch_idx):
        past, future = batch
        (
            target_past,
            control_past,
            target_future,
            control_future,
        ) = self.data_handler.prepare_data(past, future, augment=False)
        losses = self.model_core.get_loss(
            target_past,
            control_past,
            target_future,
            control_future,
            teacher_forcing=False,
        )
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
        logdir=None,
    ):
        loader_train = self.data_to_loader(data_train, train=True)
        loader_valid = self.data_to_loader(data_valid)
        self.fit_scalers(loader_train)
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
            enable_checkpointing=False,
            default_root_dir=logdir,
        )
        trainer.fit(self.model, loader_train, loader_valid)

    def fit_scalers(self, dataloader):
        for batch in dataloader:
            for seq in batch:
                self.model.scaler.update_stats(seq)

    def data_to_loader(self, data, train=False):
        if isinstance(data, torch.utils.data.DataLoader):
            # do not modify if already got data loader as an input
            return data
        dataset = self.get_dataset(data)
        if dataset is None:
            assert not train, "Training data/loader cannot be None."
            return None
        if train:
            if len(dataset) == 0:
                raise RuntimeError("Training set has zero samples.")
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=self.config.training.batch_size,
                shuffle=True,
                drop_last=True,
            )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.training.batch_size_valid,
            shuffle=False,
            drop_last=False,
        )

    def get_dataset(self, data):
        if data is None:
            return None
        if isinstance(data, torch.utils.data.DataLoader):
            return data.dataset
        if isinstance(data, torch.utils.data.Dataset):
            return data
        if isinstance(data, (list, tuple)):
            return CombinationDataset([self.get_dataset(d) for d in data])
        if isinstance(data, pd.DataFrame):
            return DataHandler.df_to_dataset(data, self.config)
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, past, future, native=True):
        self.model.eval()
        past = self.model.to_scaled(past)
        future = self.model.to_scaled(future)
        (
            target_past,
            control_past,
            _,
            control_future,
        ) = self.model.data_handler.prepare_data(past, future, augment=False)
        pred = self.model(target_past, control_past, control_future)
        pred_params = self.model.model_core.likelihood.parametrize_model_output(pred)
        params_per_target = {
            key: self.model.data_handler.split_target(tensor)
            for key, tensor in pred_params.items()
        }
        if native:
            params_per_target = self.model.model_core.likelihood.to_native(
                params_per_target, self.model.scaler
            )
        return params_per_target

    def save(self, dir):
        dir = pathlib.Path(dir)
        self.config.save(dir / "config.yaml")
        save_torch_state(self.model.model_core, dir / "model_core.pt")
        save_torch_state(self.model.scaler, dir / "scaler.pt")

    @staticmethod
    def load(dir):
        dir = pathlib.Path(dir)
        config = SeqNNConfig.load(dir / "config.yaml")
        model = SeqNN(config)
        load_torch_state(model.model.model_core, dir / "model_core.pt")
        load_torch_state(model.model.scaler, dir / "scaler.pt")
        return model
