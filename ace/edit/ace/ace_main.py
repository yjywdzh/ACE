from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from pathlib import Path

from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast
from util.globals import *

from .compute_ks import compute_ks, compute_ks_parallel
from .compute_zs import compute_zs, compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .ace_hparams import aceHyperParams

# Cache
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}
KZ_CACHE = {}


def apply_ace_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: aceHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Inject weight increments computed by ace into the model.
    """
    device = next(model.parameters()).device
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas = execute_ace(model, tok, requests, hparams, cache_template=cache_template)

    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            upd_matrix = upd_matrix.to(device)
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix.float()

    print(f"\nNew weights successfully inserted into {list(deltas.keys())}")
    return model, weights_copy


def execute_ace(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: aceHyperParams,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Execute ace update: compute and return increment matrices for all weights 
    according to specified layers and modules.
    """
    device = next(model.parameters()).device
    deltas = {}

    # Flatten/record layer grouping
    if isinstance(hparams.layers[0], list):
        layer_groups = hparams.layers
        group_weights = [1.0, 1.0]  # Can add different group weights in external JSON if needed; keep consistent with original logic here
        flat_layers = []
        for group in hparams.layers:
            flat_layers.extend(group)
        hparams.layers = flat_layers
        print(f"Flattened layers: {hparams.layers}")
        print(f"Layer groups: {layer_groups}")
        print(f"Group weights: {group_weights}")
    else:
        layer_groups = [hparams.layers]
        group_weights = [1.0]

    # Normalize target text (add leading space)
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"] and request["target_new"]["str"][0] != " ":
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in requests[:10]:
        print(
            f"ace request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # **Unified writable module list** (attn + mlp; if JSON only contains mlp then only write mlp)
    rewrite_module_names = [hparams.rewrite_module_tmp] + list(hparams.rewrite_module_tmps or [])
    rewrite_module_names = list(dict.fromkeys(rewrite_module_names))  # Remove duplicates while preserving order

    # Collect all weight handles that need modification
    weights = {
        f"{rewrite_module_name.format(layer)}.weight": nethook.get_parameter(
            model, f"{rewrite_module_name.format(layer)}.weight"
        )
        for layer in hparams.layers
        for rewrite_module_name in rewrite_module_names
    }
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute z* (at final layer z_layer)
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = {name: [] for name in rewrite_module_names}

    for request in requests:
        for rewrite_module_name in rewrite_module_names:
            block_name = "attn" if "attn" in rewrite_module_name else "mlp"
            cache_fname = (
                Path(
                    str(cache_template).format(
                        z_layer, block_name, hparams.clamp_norm_factor, request["case_id"]
                    )
                )
                if cache_template is not None
                else None
            )
            data_loaded = False
            if (cache_fname is not None) and cache_fname.exists():
                try:
                    data = np.load(cache_fname)
                    z_list[rewrite_module_name].append(torch.from_numpy(data["v_star"]).to(device))
                    data_loaded = True
                except Exception as e:
                    print(f"Error reading cache file due to {e}. Recomputing...")

            if not data_loaded:
                cur_z_attn, cur_z_mlp = compute_zs(
                    model, tok, request, hparams, z_layer, context_templates
                )
                cur_z = cur_z_attn if block_name == "attn" else cur_z_mlp
                z_list[rewrite_module_name].append(cur_z)
                if cache_fname is not None:
                    cache_fname.parent.mkdir(exist_ok=True, parents=True)
                    np.savez(cache_fname, **{"v_star": cur_z.detach().cpu().numpy()})
                    print(f"Cached k/v pair at {cache_fname}")

    for k, v in z_list.items():
        z_list[k] = torch.stack(v, dim=1)

    # Write layer by layer
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        current_group = None
        group_idx = 0
        for g_idx, group in enumerate(layer_groups):
            if layer in group:
                current_group = group
                group_idx = g_idx
                break

        if current_group:
            in_group_idx = current_group.index(layer)
            layers_remaining_in_group = len(current_group) - in_group_idx
            diffusion_factor = np.sqrt(layers_remaining_in_group)
            group_weight = group_weights[group_idx]
            print(f"Layer {layer} in group {group_idx}, position {in_group_idx+1}/{len(current_group)}")
        else:
            diffusion_factor = np.sqrt(len(hparams.layers) - i)
            group_weight = 1.0
            print(f"Layer {layer} not in any defined group")

        print(f"Diffusion factor: {diffusion_factor:.3f}, Group weight: {group_weight:.3f}")

        layers_ks = None
        # **Use parallel K retrieval as long as there are dual modules** (no longer limited to gpt-j)
        if len(rewrite_module_names) >= 2:
            if layers_ks is None:
                layers_ks = compute_ks_parallel(model, tok, requests, hparams, layer, context_templates)

        for rewrite_module_name in rewrite_module_names:
            if layers_ks is None:
                layers_ks = compute_ks(model, tok, requests, hparams, rewrite_module_name, layer, context_templates)

            print(f"Writing {layers_ks[rewrite_module_name].size(0)} key/value pair(s) into layers")

            cur_zs = get_module_input_output_at_words(
                model,
                tok,
                z_layer,
                context_templates=[request["prompt"] for request in requests],
                words=[request["subject"] for request in requests],
                module_template=rewrite_module_name,
                fact_token_strategy=hparams.fact_token,
            )[1].T  # h_i^L

            targets = z_list[rewrite_module_name] - cur_zs  # z_i - h_i^L

            layer_ks = layers_ks[rewrite_module_name].T.double().to(device)
            targets = targets.double().to(device)

            cov = get_cov(
                model,
                tok,
                rewrite_module_name.format(layer),
                hparams.mom2_dataset,
                hparams.mom2_n_samples,
                hparams.mom2_dtype,
                force_recompute=False,
            ).to(device)

            repeat_factor = (layer_ks.size(1) // targets.size(1))
            targets = targets.repeat_interleave(repeat_factor, dim=1)

            upd_matrix = (targets / diffusion_factor * group_weight) @ layer_ks.T @ torch.inverse(
                layer_ks @ layer_ks.T + hparams.mom2_update_weight * cov.double()
            )

            weight_name = f"{rewrite_module_name.format(layer)}.weight"
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

            print(weight_name, ":\norig norm", torch.linalg.norm(weights[weight_name]))
            print("upd norm", torch.linalg.norm(upd_matrix))

            with torch.no_grad():
                weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float().to(device)
                deltas[weight_name] = upd_matrix

            for x in [layer_ks, cur_zs, targets]:
                x.cpu()
                del x
            torch.cuda.empty_cache()

    # Restore original weights (deltas returned for use by outer apply function)
    with torch.no_grad():
        for k, _ in weights.items():
            nethook.get_parameter(model, k)[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")
    return deltas

def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by ace does not match original weight shape. "
            "Check for bugs in the code?"
        )

def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieve covariance statistics (or their inverse), cached/returned on same device as model.
    """
    device = next(model.parameters()).device
    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples if not force_recompute else mom2_n_samples // 10,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    cov = COV_CACHE[key].to(device)
    return torch.inverse(cov) if inv else cov


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                ) # Use model to generate sentences
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE