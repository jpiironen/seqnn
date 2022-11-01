import torch
import torch.nn as nn
import numpy as np


class Scaler(nn.Module):
    def __init__(self):
        super().__init__()

    def update_stats(self, x):
        # update needed statistics based on one batch of data
        raise NotImplementedError

    def to_scaled(self, x):
        raise NotImplementedError

    def to_native(self, x):
        raise NotImplementedError


class MinMaxScaler(Scaler):
    def __init__(self, ndim, bounds=(-2, 2), center_median=True):
        super().__init__()
        self.ndim = ndim
        self.min = nn.Parameter(np.inf * torch.ones(ndim))
        self.max = nn.Parameter(-np.inf * torch.ones(ndim))
        self.median = nn.Parameter(torch.zeros(ndim))
        self.bounds = bounds
        self.center_median = center_median
        self.batches_read = 0

    def get_min_max_dim(self, x, dim):
        assert x.ndim == 2
        valid = ~x[:, dim].isnan()
        if not valid.any():
            min_value = self.min[dim]
            max_value = self.max[dim]
        else:
            min_value = x[valid, dim].min()
            max_value = x[valid, dim].max()
        return min_value, max_value

    def get_median_dim(self, x, dim):
        assert x.ndim == 2
        valid = ~x[:, dim].isnan()
        if not valid.any():
            median = self.median[dim]
        else:
            median = x[valid, dim].median()
        return median

    def get_min_max(self, x):
        assert x.ndim == 2
        nfeat = x.shape[1]
        min_values = torch.zeros(nfeat)
        max_values = torch.zeros(nfeat)
        for dim in range(nfeat):
            min_values[dim], max_values[dim] = self.get_min_max_dim(x, dim)
        return min_values, max_values

    def get_median(self, x):
        assert x.ndim == 2
        nfeat = x.shape[1]
        median = torch.zeros(nfeat)
        for dim in range(nfeat):
            median[dim] = self.get_median_dim(x, dim)
        return median

    def update_stats(self, x):
        assert x.ndim == 3
        self.batches_read += 1
        with torch.no_grad():
            batch_min, batch_max = self.get_min_max(x.view(-1, x.shape[2]))
            batch_median = self.get_median(x.view(-1, x.shape[2]))
            self.min[:] = torch.min(self.min, batch_min)
            self.max[:] = torch.max(self.max, batch_max)
            # approximate online estimate for the median
            nb = self.batches_read
            self.median[:] = (nb - 1) / nb * self.median + 1 / nb * batch_median

    def to_scaled(self, x):
        assert x.ndim == 3
        if self.center_median:
            native_range = torch.max(self.max - self.median, self.median - self.min)
            scaled_range = 0.5 * (self.bounds[1] - self.bounds[0])
            return (x - self.median) / native_range * scaled_range
        else:
            native_range = self.max - self.min
            scaled_range = self.bounds[1] - self.bounds[0]
            return (x - self.min) / native_range * scaled_range + self.bounds[0]

    def to_native(self, x):
        assert x.ndim == 3
        if self.center_median:
            native_range = torch.max(self.max - self.median, self.median - self.min)
            scaled_range = 0.5 * (self.bounds[1] - self.bounds[0])
            return x / scaled_range * native_range + self.median
        else:
            native_range = self.max - self.min
            scaled_range = self.bounds[1] - self.bounds[0]
            return (x - self.bounds[0]) / scaled_range * native_range + self.min


class ScalerCollection(Scaler):
    def __init__(self, scalers):
        super().__init__()
        assert isinstance(scalers, dict)
        for _, scaler in scalers.items():
            assert isinstance(scaler, Scaler)
        self.scalers = nn.ModuleDict(scalers)

    def update_stats(self, data):
        for label in self.scalers.keys():
            self.scalers[label].update_stats(data[label])

    def to_scaled(self, data):
        return {
            label: self.scalers[label].to_scaled(tensor)
            if label in self.scalers.keys()
            else tensor
            for label, tensor in data.items()
        }

    def to_native(self, data):
        return {
            label: self.scalers[label].to_native(tensor)
            if label in self.scalers.keys()
            else tensor
            for label, tensor in data.items()
        }