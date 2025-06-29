import argparse
import torch

class dict2obj(dict):
    def __init__(self, d, default=None):
        self.__d = d
        self.__default = default
        super(self.__class__, self).__init__(d)

    def __getattr__(self, k):
        if k in self.__d:
            v = self.__d[k]
            if isinstance(v, dict):
                v = self.__class__(v)
            setattr(self, k, v)
            return v
        return self.__default

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    if isinstance(x, int):
        return x
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool(v)
        parser.add_argument(f"--{k}", default=v, type=v_type)   

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def create_karras_grid(t_min, t_max, num_steps, rho=7):
    steps = (
        t_max ** (1 / rho)
        + torch.arange(num_steps) / (num_steps - 1) * (t_min ** (1 / rho) - t_max ** (1 / rho))
    ) ** rho
    return steps