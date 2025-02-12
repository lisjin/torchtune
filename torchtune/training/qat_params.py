import re

from torchtune.utils import get_logger, log_rank_zero

from typing import Any, Dict, List, Optional, Tuple
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
    prefix: str = "",
    force_full_prec: bool = False,
    full_prec_pat: Optional[str] = None,
) -> None:
    """Recurse over children of model to extract quantizable weights, as well as
    non-quantizable params (params_no_quant).
    """
    if hasattr(model, "_checkpoint_wrapped_module"):
        model = model._checkpoint_wrapped_module

    for mn, module in model.named_children():
        cur_prefix = f"{prefix}.{mn}" if len(prefix) else mn
        for attr in ("_orig_mod", "module"):
            cur_prefix = cur_prefix.rsplit(f"{attr}.", 1)[-1]

        use_full_prec = force_full_prec or isinstance(module, FULL_PREC_LAYERS)
        if full_prec_pat is not None:
            use_full_prec |= re.search(full_prec_pat, cur_prefix) is not None

        for pn, param in module.named_parameters(recurse=False):
            param_name = f"{cur_prefix}.{pn}"
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
            full_prec_pat,
        )


def split_param_groups(
    model: nn.Module, full_prec_pat: Optional[str] = None
) -> Tuple[List[Any], List[Any]]:
    """Splits model parameters into quantized and un-quantized groups."""
    params_quant, params_no_quant = {}, {}
    get_param_groups(model, params_quant, params_no_quant, full_prec_pat=full_prec_pat)
    n_found_params = len(params_quant) + len(params_no_quant)
    assert n_found_params == sum(1 for p in model.parameters())

    for name, dct in zip(("quant", "no_quant"), (params_quant, params_no_quant)):
        log_rank_zero(log, f"[params_{name}], {len(dct)}: {tuple(dct.keys())}")
    return list(params_quant.values()), list(params_no_quant.values())
