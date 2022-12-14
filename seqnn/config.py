import pathlib
import pprint
from seqnn.utils import ensure_list, get_cls, save_yaml, load_yaml


class Config:
    def __init__(self, **kwargs):
        self.update(**kwargs)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return pprint.pformat(self.__dict__)

    def to_dict(self):
        return {
            key: value.to_dict() if isinstance(value, Config) else value
            for key, value in self.__dict__.items()
        }

    @staticmethod
    def from_dict(d):
        return Config(**d)


class SeqNNConfig(Config):
    def __init__(
        self,
        targets,
        horizon_past,
        horizon_future,
        controls_continuous=None,
        controls_categorical=None,
        likelihood="LikGaussian",
        likelihood_args={},
        model="NLDS",
        model_args={},
        optimizer="SGD",
        optimizer_args={"lr": 0.001, "momentum": 0.9},
        lr_scheduler="StepLR",
        lr_scheduler_args={"gamma": 1.0, "step_size": 2000},
        batch_size=32,
        batch_size_valid=32,
        validate_every_n_steps=100,
        teacher_forcing_prob=0.5,
        max_grad_norm=100.0,
        seed=42,
    ):
        targets_grouped = self.standardize_variable_set_continuous(targets)
        controls_grouped = self.standardize_variable_set_continuous(controls_continuous)
        (
            controls_cat_grouped,
            num_categories,
        ) = self.standardize_variable_set_categorical(controls_categorical)
        grouping = targets_grouped | controls_grouped | controls_cat_grouped
        task_cfg = Config(
            targets=list(targets_grouped.keys()),
            controls_cont=list(controls_grouped.keys()),
            controls_cat=list(controls_cat_grouped.keys()),
            horizon_past=horizon_past,
            horizon_future=horizon_future,
            grouping=grouping,
            num_categories=num_categories,
        )
        lik_cfg = Config(
            cls="seqnn.model.likelihood." + likelihood,
            args=likelihood_args,
        )
        model_cfg = Config(
            cls="seqnn.model.core." + model,
            args=model_args,
        )
        scaler_cfg = Config(
            groups={
                group: dict(
                    cls="seqnn.data.scalers.MinMaxScaler",
                    args={},
                )
                for group in task_cfg.grouping.keys()
                if group not in task_cfg.controls_cat
            }
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
            max_grad_norm=max_grad_norm,
            seed=seed,
            teacher_forcing_prob=teacher_forcing_prob,
            validate_every_n_steps=validate_every_n_steps,
        )
        super().__init__(
            task=task_cfg,
            lik=lik_cfg,
            model=model_cfg,
            scalers=scaler_cfg,
            optimizer=optimizer_cfg,
            scheduler=scheduler_cfg,
            training=training_cfg,
        )

    def standardize_variable_set_continuous(self, vars):
        if vars is None:
            return {}
        if isinstance(vars, (list, tuple)):
            for v in vars:
                assert isinstance(v, str)
            return {v: [v] for v in vars}
        if isinstance(vars, dict):
            for key, value in vars.items():
                assert isinstance(
                    key, str
                ), "If targets/controls is a dict, then each key must be a string"
                for v in ensure_list(value):
                    assert isinstance(
                        v, str
                    ), "If targets/controls is a dict, then each value must be a list of strings"
            vars = {key: ensure_list(value) for key, value in vars.items()}
            return vars
        raise NotImplementedError(
            f"Got unknown type for targets/controls: {type(vars)}"
        )

    def standardize_variable_set_categorical(self, vars):
        if vars is None:
            return {}, {}
        assert isinstance(vars, dict)

        for key, value in vars.items():
            assert isinstance(
                key, str
            ), "In categorical variable dict, each key must be a string denoting the variable name"
            assert isinstance(
                value, int
            ), "In categorical variable dict, each value must be an integer denoting the number of categories"

        grouping = {key: [key] for key in vars.keys()}
        num_categ = {key: value for key, value in vars.items()}
        return grouping, num_categ

    def get_likelihood(self):
        return get_cls(self.lik.cls)(**self.lik.args)

    def get_num_targets(self):
        return sum(len(self.task.grouping[group]) for group in self.task.targets)

    def get_num_controls(self):
        # number of controls from the model's perspective; so when using one-hot encoding,
        # the contribution of each categorical variable to the final sum equals its number of categories
        n = sum(len(self.task.grouping[group]) for group in self.task.controls_cont)
        n += sum(
            self.task.num_categories[tag]
            for group in self.task.controls_cat
            for tag in self.task.grouping[group]
        )
        return n

    def save(self, path):
        path = pathlib.Path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        save_yaml(self.to_dict(), path)

    @staticmethod
    def load(path):
        config_dict = load_yaml(path)
        return SeqNNConfig.from_dict(config_dict)

    @staticmethod
    def from_dict(d):
        config = SeqNNConfig(targets=None, horizon_past=None, horizon_future=None)
        d = {
            key: Config.from_dict(value) if isinstance(value, dict) else value
            for key, value in d.items()
        }
        config.update(**d)
        return config
