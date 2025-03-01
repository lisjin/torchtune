from torchtune.utils import get_logger, log_rank_zero

from typing import Any, Dict, List, Optional, Set, Tuple
from torch import nn, Tensor

log = get_logger("INFO")

FULL_PREC_LAYERS = (
    nn.modules.batchnorm._BatchNorm,
    nn.LayerNorm,
    nn.RMSNorm,
    nn.Embedding,
)


def get_param_groups(
    model: nn.Module,
    params_quant: Dict[str, Tensor],
    params_no_quant: Dict[str, Tensor],
    prefix="",
    force_full_prec=False,
) -> None:
    """Recurse over children of model to extract quantizable weights, as well as
    non-quantizable params (params_no_quant, params_wd).
    """
    if hasattr(model, "_checkpoint_wrapped_module"):
        model = model._checkpoint_wrapped_module

    for mn, module in model.named_children():
        cur_prefix = f"{prefix}.{mn}" if len(prefix) else mn

        use_full_prec = force_full_prec or isinstance(module, FULL_PREC_LAYERS)
        for pn, param in module.named_parameters(recurse=False):
            param_name = f"{cur_prefix}.{pn}"
            for attr in ("_orig_mod", "module"):
                param_name = param_name.rsplit(f"{attr}.", 1)[-1]

            if not use_full_prec and param.dim() > 1:
                params_quant[param_name] = param
            else:
                params_no_quant[param_name] = param
        get_param_groups(
            module,
            params_quant,
            params_no_quant,
            cur_prefix,
            use_full_prec,
        )


def split_param_groups(model: nn.Module) -> Tuple[List[Any], List[Any], List[Any]]:
    """Splits model parameters into quantized and un-quantized groups."""
    params_quant, params_no_quant = {}, {}
    get_param_groups(model, params_quant, params_no_quant)
    n_found_params = len(params_quant) + len(params_no_quant)
    assert n_found_params == len(list(model.parameters()))

    for name, dct in zip(("quant", "no_quant"), (params_quant, params_no_quant)):
        log_rank_zero(log, f"[params_{name}], {len(dct)}: {tuple(dct.keys())}")
    return list(params_quant.values()), list(params_no_quant.values())
