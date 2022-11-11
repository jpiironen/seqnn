import numpy as np
import pandas as pd
import torch
import seqnn.data.dataset
from seqnn.utils import ensure_list


class DataHandler:
    def __init__(self, config) -> None:
        self.config = config
        self.targets = config.task.targets
        self.targets_diff = config.task.targets_diff
        self.controls_cont = config.task.controls_cont
        self.controls_cat = config.task.controls_cat
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

    def get_group_and_index(self, tag):
        return self.tag_to_group_and_index[tag]

    def get_tags(self, data_dict, tags):
        tags = ensure_list(tags)
        data_out = []
        for tag in tags:
            group_name, index_within_group = self.get_group_and_index(tag)
            tensor = data_dict[group_name]
            assert tensor.ndim == 3, "Expected a dictionary of 3d-tensors."
            data_out.append(tensor[:, :, index_within_group : index_within_group + 1])
        return torch.cat(data_out, dim=-1)

    def set_tags(self, data_dict, tags, values):
        tags = ensure_list(tags)
        assert values.ndim == 3
        assert len(tags) == values.shape[2]
        for i, tag in enumerate(tags):
            group_name, index_within_group = self.tag_to_group_and_index[tag]
            tensor = data_dict[group_name]
            assert tensor.ndim == 3, "Expected a dictionary of 3d-tensors."
            tensor[:, :, index_within_group : index_within_group + 1] = values[
                :, :, i : i + 1
            ]

    def get_control(self, data_dict, batch_size, seq_len):

        if len(self.controls_cont) == 0:
            # no continuous controls, create empty tensor with appropriate dimensions
            control_cont = torch.tensor([]).view(batch_size, seq_len, 0)
        else:
            control_cont = torch.cat(
                [data_dict[control] for control in self.controls_cont], dim=2
            )
        if len(self.controls_cat) == 0:
            # no categorical controls, create empty tensor with appropriate dimensions
            control_cat = torch.tensor([]).view(batch_size, seq_len, 0)
        else:
            # one-hot encoding
            control_cat = torch.cat(
                [
                    torch.nn.functional.one_hot(
                        self.get_tags(data_dict, tag).squeeze(dim=-1).to(torch.long),
                        num_classes=self.config.task.num_categories[tag],
                    )
                    for group in self.controls_cat
                    for tag in self.config.task.grouping[group]
                ],
                dim=2,
            ).to(torch.float)
        return torch.cat([control_cont, control_cat], dim=2)

    def get_target(self, data_dict):
        if self.targets[0] in data_dict:
            return torch.cat([data_dict[target] for target in self.targets], dim=2)
        return None
    
    def get_target_diff(self, data_dict):
        if self.targets_diff[0] in data_dict:
            return torch.cat([data_dict[target] for target in self.targets_diff], dim=2)
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
        # TODO: implement augmentation
        # if augment:
        #    past, future = self.augment(past, future)

        # past, future = self.impute(past, future)
        batch_size, len_past, _ = past[self.targets[0]].shape
        len_future = (
            future[self.targets[0]].shape[1]
            if self.targets[0] in future
            else self.config.task.horizon_future
        )
        return (
            self.get_target(past),
            self.get_target_diff(past),
            self.get_control(past, batch_size, len_past),
            self.get_target(future),
            self.get_target_diff(future),
            self.get_control(future, batch_size, len_future),
        )

    @staticmethod
    def df_to_dataset(df, config, past_only=False):
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
        if past_only:
            seq_partitioning = config.task.horizon_past
        else:
            seq_partitioning = (config.task.horizon_past, config.task.horizon_future)
        return seqnn.data.dataset.DictSeqDataset(
            data_dict,
            seq_partitioning,
            index=df.index,
        )
