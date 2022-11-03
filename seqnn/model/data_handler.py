import numpy as np
import pandas as pd
import torch
import seqnn.data.dataset
from seqnn.utils import ensure_list


class DataHandler:
    def __init__(self, config) -> None:
        self.config = config
        self.targets = config.task.targets
        self.controls = config.task.controls
        self.tag_to_group_and_index = self.create_tag_index()

    def create_tag_index(self):
        # create a dictionary where:
        #   key = <tag name>
        #   value = (<group_name>, <index within group>)
        mapping = {}
        for group_name, tags in self.config.task.grouping.items():
            for i, tag in enumerate(tags):
                mapping[tag] = (group_name, i)
        return mapping

    def get_tags(self, data_dict, tags):
        tags = ensure_list(tags)
        data_out = []
        for tag in tags:
            group_name, index_within_group = self.tag_to_group_and_index[tag]
            tensor = data_dict[group_name]
            assert tensor.ndim == 3, "Expected a dictionary of 3d-tensors."
            data_out.append(tensor[:, :, index_within_group : index_within_group + 1])
        return torch.cat(data_out, dim=-1)

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

    def split_target(self, data_tensor):
        groups = self.config.task.grouping
        target_group_sizes = [len(groups[target]) for target in self.targets]
        if data_tensor is not None:
            data_splitted = torch.split(data_tensor, target_group_sizes, dim=2)
        else:
            data_splitted = [None] * len(target_group_sizes)
        return {group: data for group, data in zip(self.targets, data_splitted)}

    def prepare_data(self, past, future, augment=False):

        # if augment:
        #    past, future = self.augment_native(past, future)

        # past, future = self.impute(past, future)
        return (
            self.get_target(past),
            self.get_control(past),
            self.get_target(future),
            self.get_control(future),
        )

    @staticmethod
    def df_to_dataset(df, config):
        if isinstance(df, (list, tuple)):
            # multiple data frames, so create a combination dataset
            datasets = [DataHandler.df_to_dataset(d, config) for d in df]
            return seqnn.data.dataset.CombinationDataset(datasets)
        assert isinstance(
            df, pd.DataFrame
        ), "Expected pandas data frame, got %s" % type(df)
        data_dict = {
            group_name: torch.tensor(np.array(df[tags]), dtype=torch.float)
            for group_name, tags in config.task.grouping.items()
        }
        return seqnn.data.dataset.DictSeqDataset(
            data_dict,
            (config.task.horizon_past, config.task.horizon_future),
            index=df.index,
        )
