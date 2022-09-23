import pprint
import seqnn
from seqnn.utils import ensure_list


class Config:
    def __init__(self, **kwargs):
        self.update(**kwargs)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return pprint.pformat(self.__dict__)


class SeqNNConfig(Config):
    def __init__(
        self,
        targets,
        controls,
        horizon_past,
        horizon_future,
        likelihood="LikGaussian",
        model="RNN",
        model_args={},
        optimizer="SGD",
        optimizer_args={"lr": 0.001, "momentum": 0.9},
        lr_scheduler="StepLR",
        lr_scheduler_args={"gamma": 0.5, "step_size": 2000},
        batch_size=32,
        batch_size_valid=32,
        validate_every_nsteps=100,
    ):
        task_cfg = Config(
            targets=ensure_list(targets),
            controls=ensure_list(controls),
            horizon_past=horizon_past,
            horizon_future=horizon_future,
            likelihood=likelihood,
        )
        model_cfg = Config(
            cls="seqnn.model.core." + model,
            args=model_args,
        )
        optimizer_cfg = Config(
            cls="torch.optim." + optimizer,
            args=optimizer_args,
        )
        scheduler_cfg = Config(
            cls="torch.optim.lr_scheduler." + lr_scheduler,
            args=lr_scheduler_args,
        )
        training_cfg = Config(
            batch_size=batch_size,
            batch_size_valid=batch_size_valid,
            validate_every_nsteps=validate_every_nsteps,
        )
        super().__init__(
            task=task_cfg,
            model=model_cfg,
            optimizer=optimizer_cfg,
            scheduler=scheduler_cfg,
            training=training_cfg,
        )

    def get_likelihood(self):
        return getattr(seqnn.model.likelihood, self.task.likelihood)()

    def get_num_targets(self):
        # TODO: DUMMY
        return len(self.task.targets)

    def get_num_controls(self):
        # TODO: DUMMY
        return len(self.task.controls)
