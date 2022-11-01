import numpy as np
import torch

from seqnn.utils import ensure_list


class DictSeqDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_len, index=None):
        self.data = data
        self.seq_parts = ensure_list(seq_len, flatten_tuple=True)
        self.seq_len = sum(self.seq_parts)
        for _, v in data.items():
            if index is not None:
                assert (
                    len(index) == v.shape[0]
                ), "Number of items in index must match with the size of the dataset"
                self.index = index
            else:
                self.index = np.arange(v.shape[0])
            self.length = max(0, v.shape[0] - self.seq_len + 1)
            break

    def get_data(self):
        return self.data

    # def get_index(self, i):
    #    # TODO: this needs to be implemented
    #    if self.split_past_future:
    #        past_index = self.index[i : i + self.horizon_past]
    #        future_index = self.index[i + self.horizon_past : i + self.seq_len]
    #        return past_index, future_index
    #    return self.index[i : i + self.seq_len]

    def __getitem__(self, i):
        if i < 0:
            # map negative indices to their equivalent positive counterparts
            i = len(self) + i

        seq_dict = {
            name: values[i : i + self.seq_len] for name, values in self.data.items()
        }
        n_splits = len(self.seq_parts)

        if n_splits == 1:
            return seq_dict

        seq_dicts = [{} for _ in range(n_splits)]
        for name, tensor in seq_dict.items():
            splits = torch.split(tensor, self.seq_parts)
            for k, split in enumerate(splits):
                seq_dicts[k][name] = split
        return seq_dicts

    def __len__(self):
        return self.length


class CombinationDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = self.filter_datasets(datasets)
        self.lengths = [len(ds) for ds in self.datasets]
        self.indices = self.create_indices()

    def filter_datasets(self, datasets):
        """
        Returns list of datasets where datasets with zero length are filtered out.
        """
        datasets_filtered = []
        for ds in datasets:
            if len(ds) > 0:
                datasets_filtered.append(ds)
        return datasets_filtered

    def create_indices(self):
        """
        Returns a list whose length is equal to the length of the combined dataset,
        where entry i is a two-element tuple with first value giving the index of a dataset
        and the second value the index within that dataset.
        """
        cum_lengths = np.cumsum(self.lengths)
        indices = []
        for i in range(len(self)):

            dataset_idx = np.nonzero(i < cum_lengths)[0][0]
            if dataset_idx == 0:
                idx_within_dataset = i
            else:
                idx_within_dataset = i - cum_lengths[dataset_idx - 1]
            indices.append((dataset_idx, idx_within_dataset))
        return indices

    def get_index(self, i):
        dataset_idx, idx_within_dataset = self.indices[i]
        return self.datasets[dataset_idx].get_index(idx_within_dataset)

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, i):
        dataset_idx, idx_within_dataset = self.indices[i]
        return self.datasets[dataset_idx][idx_within_dataset]
