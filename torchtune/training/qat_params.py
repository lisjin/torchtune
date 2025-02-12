from torchtune.training import get_world_size_and_rank

from typing import Any, Dict, List, Optional, Set, Tuple
from torch import nn, Tensor

NORM_LAYERS = (nn.modules.batchnorm._BatchNorm, nn.LayerNorm, nn.RMSNorm)


def get_param_groups(
    model: torch.nn.Module,
    params_quant: Dict[str, Tensor],
    params_no_wd: Dict[str, Tensor],
    params_wd: Dict[str, Tensor],
    skip_wd_names: Optional[Set[str]] = None,
    prefix="",
    force_full_prec=False,
) -> None:
    """Recurse over children of model to extract quantizable weights, as well as
    non-quantizable params (params_no_wd, params_wd).
    """
    for mn, module in model.named_children():
        cur_prefix = f"{prefix}.{mn}" if len(prefix) else mn

        use_full_prec = force_full_prec or isinstance(module, NORM_LAYERS)
        for pn, param in module.named_parameters(recurse=False):
            param_name = f"{cur_prefix}.{pn}"
            for attr in ("_orig_mod", "module"):
                param_name = param_name.rsplit(f"{attr}.", 1)[-1]

            if not use_full_prec and param.dim() > 1:
                params_quant[param_name] = param
            elif pn == "bias" or skip_wd_names and param_name in skip_wd_names:
                params_no_wd[param_name] = param
            else:
                params_wd[param_name] = param
        get_param_groups(
            module,
            params_quant,
            params_no_wd,
            params_wd,
            skip_wd_names,
            cur_prefix,
            use_full_prec,
        )


def split_param_groups(
    model: torch.nn.Module, skip_wd_names: Optional[Set[str]] = None
) -> Tuple[List[Any], List[Any], List[Any]]:
    """Splits model parameters into 3 groups, described below.

    Returns:
        params_quant: quantized, weight decay
        params_no_wd: unquantized, no weight decay
        params_wd: unquantized, weight decay
    """
    params_quant, params_no_wd, params_wd = {}, {}, {}
    get_param_groups(model, params_quant, params_no_wd, params_wd, skip_wd_names)
    n_found_params = len(params_quant) + len(params_no_wd) + len(params_wd)
    assert n_found_params == len(list(model.parameters()))

    _, rank = get_world_size_and_rank()
    if rank == 0:
        for name, dct in zip(
            ("quant", "no_wd", "wd"), (params_quant, params_no_wd, params_wd)
        ):
            print(f"[params_{name}], {len(dct)}: {tuple(dct.keys())}")
    return (
        list(params_quant.values()),
        list(params_no_wd.values()),
        list(params_wd.values()),
    )


class AverageMeter(object):
    """Computes and stores the average and current value

    The (global) average is synced across all worker processes. The main process
    can optionally have a fixed length buffer to average over its most recent
    `window_size` values.
    """

    def __init__(self, window_size=0):
        self.reset(window_size)

    def reset(self, window_size):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.buf = (
            deque(maxlen=window_size) if is_main_process() and window_size > 0 else None
        )

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.buf is not None:
            self.buf.extend([val] * n)

    @property
    def avg(self):
        """Average across all values. Reduces across workers if applicable."""
        if dist.is_available() and dist.is_initialized():
            # Average over worker values if using distributed training
            t = torch.tensor([self.count, self.sum], device="cuda")
            dist.barrier()
            dist.all_reduce(t)

            self.count = int(t[0].item())
            self.sum = t[1].item()
        return self.sum / self.count

    @property
    def window_avg(self):
        """Worker-specific average over the most recent `window_size` elements"""
        return sum(self.buf) / len(self.buf) if self.buf is not None else -1


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
