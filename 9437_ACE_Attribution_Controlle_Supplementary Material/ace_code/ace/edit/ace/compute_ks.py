from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .compute_zs import get_modules_input_output_at_words, get_module_input_output_at_words
from .ace_hparams import aceHyperParams


def compute_ks_parallel(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: Dict,
    hparams: aceHyperParams,
    layer: int,
    context_templates: List[str],
):
    """
    Compute K (input representations required for equation 19) for multiple writable modules 
    simultaneously (typically attn and mlp). When JSON specifies only one module, this function 
    can degrade to single-module behavior.
    """
    layers_ks = dict()

    # Key: merge rewrite_module_tmp (single, usually attn) + rewrite_module_tmps (list, usually includes mlp)
    rewrite_module_tmps = [hparams.rewrite_module_tmp] + list(hparams.rewrite_module_tmps or [])
    rewrite_module_tmps = list(dict.fromkeys(rewrite_module_tmps))  # Remove duplicates while preserving order

    if len(rewrite_module_tmps) == 1:
        # Degrade to compute_ks for single module
        single = rewrite_module_tmps[0]
        layer_ks = get_module_input_output_at_words(
            model,
            tok,
            layer,
            context_templates=[
                context.format(request["prompt"])
                for request in requests
                for context_type in context_templates
                for context in context_type
            ],
            words=[
                request["subject"]
                for request in requests
                for context_type in context_templates
                for _ in context_type
            ],
            module_template=single,
            fact_token_strategy=hparams.fact_token,
        )[0]

        context_type_lens = [0] + [len(context_type) for context_type in context_templates]
        context_len = sum(context_type_lens)
        context_type_csum = np.cumsum(context_type_lens).tolist()

        ans = []
        for i in range(0, layer_ks.size(0), context_len):
            tmp = []
            for j in range(len(context_type_csum) - 1):
                start, end = context_type_csum[j], context_type_csum[j + 1]
                tmp.append(layer_ks[i + start: i + end].mean(0))
            ans.append(torch.stack(tmp, 0).mean(0))
        layers_ks[single] = torch.stack(ans, dim=0)
        return layers_ks

    # Parallel processing for dual/multiple modules
    ks_tuple = get_modules_input_output_at_words(
        model,
        tok,
        layer,
        context_templates=[
            context.format(request["prompt"])
            for request in requests
            for context_type in context_templates
            for context in context_type
        ],
        words=[
            request["subject"]
            for request in requests
            for context_type in context_templates
            for _ in context_type
        ],
        module_templates=rewrite_module_tmps,
        fact_token_strategy=hparams.fact_token,
    )

    # Aggregate parallel results by context
    context_type_lens = [0] + [len(context_type) for context_type in context_templates]
    context_len = sum(context_type_lens)
    context_type_csum = np.cumsum(context_type_lens).tolist()

    for idx, rewrite_module_tmp in enumerate(rewrite_module_tmps):
        cur_ks = ks_tuple[idx]
        ans = []
        for i in range(0, cur_ks.size(0), context_len):
            tmp = []
            for j in range(len(context_type_csum) - 1):
                start, end = context_type_csum[j], context_type_csum[j + 1]
                tmp.append(cur_ks[i + start: i + end].mean(0))
            ans.append(torch.stack(tmp, 0).mean(0))
        layers_ks[rewrite_module_tmp] = torch.stack(ans, dim=0)

    return layers_ks

def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: Dict,
    hparams: aceHyperParams,
    rewrite_module_tmp: str,
    layer: int,
    context_templates: List[str],
):
    layers_ks = dict()
    layer_ks = get_module_input_output_at_words(
        model,
        tok,
        layer,
        context_templates=[
            context.format(request["prompt"])
            for request in requests
            for context_type in context_templates
            for context in context_type
        ],
        words=[
            request["subject"]
            for request in requests
            for context_type in context_templates
            for _ in context_type
        ],
        module_template=rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
         )[0]

    context_type_lens = [0] + [len(context_type) for context_type in context_templates]
    context_len = sum(context_type_lens)
    context_type_csum = np.cumsum(context_type_lens).tolist()

    ans = []
    for i in range(0, layer_ks.size(0), context_len):
        tmp = []
        for j in range(len(context_type_csum) - 1):
            start, end = context_type_csum[j], context_type_csum[j + 1]
            tmp.append(layer_ks[i + start : i + end].mean(0))
        ans.append(torch.stack(tmp, 0).mean(0))
    layers_ks[rewrite_module_tmp] = torch.stack(ans, dim=0)
    return layers_ks