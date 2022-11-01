import yaml
import torch
from importlib import import_module


def get_cls(path_as_str):
    parts = path_as_str.split(".")
    if len(parts) == 1:
        raise ValueError(
            "Cannot figure out the class if the full module path is not given"
        )
    module = import_module(".".join(parts[:-1]))
    return getattr(module, parts[-1])


def ensure_list(arg, flatten_tuple=False):
    if arg is None:
        return []
    if flatten_tuple:
        if isinstance(arg, tuple):
            arg = list(arg)
    return arg if isinstance(arg, list) else [arg]


def save_yaml(dictionary, path):
    with open(path, "w") as file:
        yaml.dump(dictionary, file, default_flow_style=False)


def load_yaml(path):
    with open(path, "rb") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data

def save_torch_state(module, path):
    torch.save(module.state_dict(), path)

def load_torch_state(module, path):
    module.load_state_dict(torch.load(path))

def get_data_sample(dataset, indices):
    indices = ensure_list(indices)
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=len(indices))
    for data in loader:
        break
    return data
