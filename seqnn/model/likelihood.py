import torch
import torch.nn as nn


class Likelihood:
    def real_to_positive(self, x):
        return nn.functional.softplus(x + 0.5)

    def get_num_parameters(self):
        raise NotImplementedError

    def parametrize_model_output(self, model_output):
        raise NotImplementedError

    def to_native(self, param, scaler):
        raise NotImplementedError

    def get_loss(self, model_output, target):
        raise NotImplementedError

    def sample(self, model_output):
        raise NotImplementedError

    def quantile(self, model_output, prob):
        raise NotImplementedError


class LikGaussian(Likelihood):
    def __init__(self):
        super().__init__()

    def get_num_parameters(self):
        return 2

    def parametrize_model_output(self, x):
        assert x.ndim == 3
        num_obs = int(x.shape[2] / 2)
        mean, scale_unconstrained = torch.split(x, (num_obs, num_obs), dim=2)
        scale = self.real_to_positive(scale_unconstrained)
        return {"mean": mean, "scale": scale}

    def to_native(self, param, scaler):
        param_native = {}
        param_native["mean"] = scaler.to_native(param["mean"])
        if isinstance(param["mean"], dict) and isinstance(param["scale"], dict):
            mean_plus_scale = {
                key: param["mean"][key] + param["scale"][key] for key in param["mean"]
            }
            mean_plus_scale_native = scaler.to_native(mean_plus_scale)
            param_native["scale"] = {
                key: mean_plus_scale_native[key] - param_native["mean"][key]
                for key in mean_plus_scale_native
            }
        else:
            mean_plus_scale = param["mean"] + param["scale"]
            mean_plus_scale_native = scaler.to_native(mean_plus_scale)
            param_native["scale"] = mean_plus_scale_native - param_native["mean"]
        return param_native

    def get_loss(self, model_output, target):
        p = self.parametrize_model_output(model_output)
        loss = -torch.distributions.Normal(p["mean"], p["scale"]).log_prob(target)
        return loss

    def sample(self, model_output):
        p = self.parametrize_model_output(model_output)
        return torch.distributions.Normal(p["mean"], p["scale"]).sample()

    def quantile(self, model_output, prob):
        p = self.parametrize_model_output(model_output)
        return torch.distributions.Normal(p["mean"], p["scale"]).icdf(prob)


class LikCategorical(Likelihood):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def get_num_parameters(self):
        return self.num_classes

    def parametrize_model_output(self, model_output):
        return {"probs": torch.softmax(model_output, dim=-1)}

    def get_loss(self, model_output, target):
        loss = -torch.distributions.Categorical(logits=model_output).log_prob(target)
        return loss

    def sample(self, model_output):
        return torch.distributions.Categorical(logits=model_output).sample()
