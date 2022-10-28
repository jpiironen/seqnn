import numpy as np
import pandas as pd
import torch
import seqnn.data.dataset


class DataHandler:
    def __init__(self, config) -> None:
        self.config = config
        self.targets = config.task.targets
        self.controls = config.task.controls
        self.tagindex = {tag: i for i, tag in enumerate(self.targets + self.controls)}

    def get_tags(self, tensor, tags):
        assert tensor.ndim == 3
        assert isinstance(tags, list)
        indices = [self.tagindex[tag] for tag in tags]
        return tensor[:, :, indices]

    def get_control(self, data_dict):
        if len(self.controls) == 0:
            # no controls, return empty tensor with appropriate dimensions
            some_target_group = self.targets[0]
            batch_size, seq_len, _ = data_dict[some_target_group].shape
            return torch.tensor([]).view(batch_size, seq_len, 0)
        return torch.cat([data_dict[control] for control in self.controls], dim=2)

    def get_target(self, data_dict):
        if self.targets[0] in data_dict:
            return torch.cat([data_dict[target] for target in self.targets], dim=2)
        return None

    def prepare_data(self, batch, augment=False):

        # if augment:
        #    past, future = self.augment_native(past, future)

        # if self.scaler:
        #    past = self.to_scaled(past)
        #    future = self.to_scaled(future)

        # past, future = self.impute(past, future)

        # target_past = self.get_target(past)
        # target_future = self.get_target(future)
        # control_past = self.get_control(past)
        # control_future = self.get_control(future)
        # return target_past, target_future, control_past, control_future

        return self.get_target(batch), self.get_control(batch)

    @staticmethod
    def get_variable_grouping(config):
        # TODO: DUMMY
        return {tag: [tag] for tag in config.task.targets + config.task.controls}

    @staticmethod
    def df_to_dataset(df, config):
        if isinstance(df, (list, tuple)):
            # multiple data frames, so create a combination dataset
            datasets = [
                DataHandler.df_to_dataset(d, config)
                for d in df
            ]
            return seqnn.data.dataset.CombinationDataset(datasets)
        assert isinstance(
            df, pd.DataFrame
        ), "Expected pandas data frame, got %s" % type(df)
        seq_len = config.task.horizon_past + config.task.horizon_future
        groups = DataHandler.get_variable_grouping(config)
        data_dict = {
            group_name: torch.tensor(np.array(df[tags]), dtype=torch.float)
            for group_name, tags in groups.items()
        }
        return seqnn.data.dataset.DictSeqDataset(
            data_dict, seq_len, index=df.index
        )