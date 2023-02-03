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


class PastFutureScalerCollection(nn.Module):
    def __init__(self, scalers):
        super().__init__()
        assert isinstance(scalers, dict)
        for _, scaler in scalers.items():
            assert isinstance(scaler, PastFutureScaler)
        self.scalers = nn.ModuleDict(scalers)

    def update_stats(self, past, future):
        for label in self.scalers.keys():
            self.scalers[label].update_stats(past[label], future[label])

    def to_scaled(self, past, future):
        for key in future:
            assert key in past
        past_scaled = {}
        future_scaled = {}
        for label in past.keys():
            if label in self.scalers.keys():
                if label in future:
                    past_scaled[label], future_scaled[label] = self.scalers[
                        label
                    ].to_scaled(past[label], future[label])
                else:
                    past_scaled[label], _ = self.scalers[label].to_scaled(past[label])
            else:
                past_scaled[label] = past[label] # TODO: should we copy?
                if label in future:
                    future_scaled[label] = future[label]
        return past_scaled, future_scaled

    def to_native(self, past, future):
        for key in future:
            assert key in past
        past_native = {}
        future_native = {}
        for label in past.keys():
            if label in self.scalers.keys():
                if label in future:
                    past_native[label], future_native[label] = self.scalers[
                        label
                    ].to_native(past[label], future[label])
                else:
                    past_native[label], _ = self.scalers[label].to_native(past[label])
            else:
                past_native[label] = past[label]
                if label in future:
                    future_native[label] = future[label]
        return past_native, future_native


class PastFutureScaler(nn.Module):
    def __init__(
        self,
        ndim,
        bounds=(-2, 2),
        scale_future=2.0,
        center_future=False,
        diff_future=False,
        eps=1e-9,
    ):
        super().__init__()
        self.ndim = ndim
        self.min_past = nn.Parameter(np.inf * torch.ones(ndim))
        self.max_past = nn.Parameter(-np.inf * torch.ones(ndim))
        self.abs_max_future = nn.Parameter(0.0 * torch.ones(ndim))
        self.bounds = bounds
        self.scale_future = scale_future
        self.identical_scaling = not diff_future and not center_future
        self.diff_future = diff_future
        self.center_future = center_future
        self.batches_read = 0
        self.eps = eps

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

    def get_min_max(self, x):
        assert x.ndim == 2
        nfeat = x.shape[1]
        min_values = torch.zeros(nfeat)
        max_values = torch.zeros(nfeat)
        for dim in range(nfeat):
            min_values[dim], max_values[dim] = self.get_min_max_dim(x, dim)
        return min_values, max_values

    def transform_future(self, future, past):
        if self.diff_future:
            # differencing, i.e. make future trajectories stationary
            future = torch.cat([past[:, -1:, :], future], dim=1).diff(dim=1)
        elif self.center_future:
            # center to the last value in past
            future = future - past[:, -1:, :]
        return future

    def inv_transform_future(self, future, past):
        if self.diff_future:
            future = past[:, -1:, :] + future.cumsum(dim=1)
        elif self.center_future:
            # center to the last value in past
            future = future + past[:, -1:, :]
        return future

    def update_stats(self, past, future):
        assert past.ndim == 3
        assert future.ndim == 3

        seq_all = torch.cat((past, future), dim=1)
        future = self.transform_future(future, past)

        self.batches_read += 1
        with torch.no_grad():
            # past; use all the data to learn the statistics
            batch_min, batch_max = self.get_min_max(seq_all.view(-1, seq_all.shape[2]))
            self.min_past[:] = torch.min(self.min_past, batch_min)
            self.max_past[:] = torch.max(self.max_past, batch_max)

            # future
            batch_min, batch_max = self.get_min_max(future.view(-1, future.shape[2]))
            batch_abs_max = torch.max(batch_min.abs(), batch_max.abs())
            self.abs_max_future[:] = torch.max(self.abs_max_future, batch_abs_max)

    def to_scaled(self, past, future=None):
        assert past.ndim == 3
        if future is not None:
            assert future.ndim == 3
            assert past.shape[2] == future.shape[2]

        # past
        scaled_range = self.bounds[1] - self.bounds[0]
        native_range = self.max_past - self.min_past
        past_scaled = (past - self.min_past) / (
            native_range + self.eps
        ) * scaled_range + self.bounds[0]

        if future is None:
            return past_scaled, None

        # future
        if self.identical_scaling:
            future_scaled = (future - self.min_past) / (
                native_range + self.eps
            ) * scaled_range + self.bounds[0]
        else:
            future = self.transform_future(future, past)
            future_scaled = (
                future / (self.abs_max_future + self.eps) * self.scale_future
            )
        return past_scaled, future_scaled

    def to_native(self, past, future=None):
        assert past.ndim == 3
        if future is not None:
            assert future.ndim == 3
            assert past.shape[2] == future.shape[2]

        # past
        scaled_range = self.bounds[1] - self.bounds[0]
        native_range = self.max_past - self.min_past
        past_native = (past - self.bounds[0]) / (
            scaled_range + self.eps
        ) * native_range + self.min_past

        if future is None:
            return past_native, None

        # future
        if self.identical_scaling:
            future_native = (future - self.bounds[0]) / (
                scaled_range + self.eps
            ) * native_range + self.min_past
        else:
            future = future / (self.scale_future + self.eps) * self.abs_max_future
            future_native = self.inv_transform_future(future, past_native)
        return past_native, future_native
