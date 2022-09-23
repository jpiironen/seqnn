from importlib import import_module

def get_cls(path_as_str):
    parts = path_as_str.split(".")
    if len(parts) == 1:
        raise ValueError("Cannot figure out the class if the full module path is not given")
    module = import_module(".".join(parts[:-1]))
    return getattr(module, parts[-1])

def ensure_list(arg):
    if arg is None:
        return []
    return arg if isinstance(arg, list) else [arg]