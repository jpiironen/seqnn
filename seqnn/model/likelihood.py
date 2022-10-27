import torch
import torch.nn as nn


class Likelihood:
    def real_to_positive(self, x):
        return nn.functional.softplus(x + 0.5)

    def get_num_parameters(self):
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

    def get_loss(self, model_output, target, reduce="mean"):
        mean, scale = self.split_model_output(model_output)
        loss = -torch.distributions.Normal(mean, scale).log_prob(target)
        if reduce == "mean":
            return loss.mean()
        elif reduce == "sum":
            return loss.sum()
        return loss

    def split_model_output(self, x):
        assert x.ndim == 3
        num_obs = int(x.shape[2] / 2)
        mean, scale_unconstrained = torch.split(x, (num_obs, num_obs), dim=2)
        scale = self.real_to_positive(scale_unconstrained)
        return mean, scale

    def sample(self, model_output):
        mean, scale = self.split_model_output(model_output)
        return torch.distributions.Normal(mean, scale).sample()

    def quantile(self, model_output, prob):
        mean, scale = self.split_model_output(model_output)
        return torch.distributions.Normal(mean, scale).icdf(prob)
