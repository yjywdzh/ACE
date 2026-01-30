from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import repr_tools
from util import nethook

from .ace_hparams import aceHyperParams


def compute_zs(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: aceHyperParams,
    layer: int,
    context_templates: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute target right vectors v* (one for attn and one for mlp) by optimizing 
    through inserting learnable delta near the rewrite layer.
    Adapted for models: hidden_size, module naming, single device.
    """
    device = next(model.parameters()).device

    # Read lm_head / ln_f
    if ("neo" in model.config._name_or_path) or ("gpt2" in model.config._name_or_path):
        ln_f = nethook.get_module(model, hparams.ln_f_module)
        lm_head_module = nethook.get_module(model, hparams.lm_head_module)
        lm_w = nethook.get_parameter(lm_head_module, "weight").T
    else:
        lm_w = nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T
        ln_f = nethook.get_module(model, hparams.ln_f_module)

    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size).to(device)

    print("Computing right vector (v)")

    # Target tokens
    target_ids = tok(request["target_new"]["str"], return_tensors="pt").to(device)["input_ids"][0]

    # Compose rewrite and KL prompts
    rewriting_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context_types in context_templates
        for context in context_types
    ]
    kl_prompts = ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to(device)

    # Rewrite target labels
    rewriting_targets = torch.tensor(-100, device=device).repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids): ex_len] = target_ids

    # Find fact lookup positions
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Bind loss layer
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Initialize learnable variables (dimension from hidden_size or n_embd)
    embed_dim = getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd", None)
    if embed_dim is None:
        raise ValueError("Cannot find hidden size from model.config (expected 'hidden_size' or 'n_embd').")

    delta_attn = torch.zeros((embed_dim,), requires_grad=True, device=device)
    delta_mlp = torch.zeros((embed_dim,), requires_grad=True, device=device)
    target_init_attn, target_init_mlp, kl_distr_init = None, None, None

    # Hook to insert delta
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init_attn, target_init_mlp

        if cur_layer == hparams.mlp_module_tmp.format(layer):
            if target_init_mlp is None:
                print("Recording initial value of v* in mlp")
                target_init_mlp = cur_out[0, lookup_idxs[0]].detach().clone()
            for i, idx in enumerate(lookup_idxs):
                cur_out[i, idx, :] += delta_mlp

        if cur_layer == hparams.attn_module_tmp.format(layer):
            if target_init_attn is None:
                print("Recording initial value of v* in attn")
                target_init_attn = cur_out[0, lookup_idxs[0]].detach().clone()
            for i, idx in enumerate(lookup_idxs):
                cur_out[i, idx, :] += delta_attn

        return cur_out

    opt = torch.optim.Adam([delta_mlp, delta_attn], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)
    nll_loss_factor = hparams.nll_loss_factor
    kl_factor = hparams.kl_factor

    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer),
                hparams.attn_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts):])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        full_repr = tr[hparams.layer_module_tmp.format(loss_layer)].output[0][: len(rewriting_prompts)]
        log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w + lm_b, dim=2)

        loss_nll_raw = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()
        max_probs = torch.max(log_probs, dim=2)[0]
        max_prob = torch.exp((max_probs * mask).sum(1) / target_ids.size(0)).mean().item()

        nll_loss_each = -(loss_nll_raw * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_factor * nll_loss_each.mean()
        kl_loss = kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta_mlp) / torch.norm(target_init_mlp) ** 2
            + torch.norm(delta_attn) / torch.norm(target_init_attn) ** 2
        )
        loss = nll_loss + kl_loss + weight_decay
        prob = torch.exp(-nll_loss_each).mean().item()

        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']['str']}] {prob}"
        )

        if loss < 5e-2:
            break
        if max_prob == prob:
            nll_loss_factor = 0.1 * hparams.nll_loss_factor
            if kl_loss <= 0.01:
                break
        else:
            nll_loss_factor = hparams.nll_loss_factor

        if it == hparams.v_num_grad_steps - 1:
            break

        loss.backward()
        opt.step()

        max_norm = hparams.clamp_norm_factor * target_init_mlp.norm()
        if delta_mlp.norm() > max_norm:
            with torch.no_grad():
                delta_mlp[...] = delta_mlp * max_norm / delta_mlp.norm()

        max_norm = hparams.clamp_norm_factor * target_init_attn.norm()
        if delta_attn.norm() > max_norm:
            with torch.no_grad():
                delta_attn[...] = delta_attn * max_norm / delta_attn.norm()

    target_attn = target_init_attn + delta_attn
    target_mlp = target_init_mlp + delta_mlp
    print(
        f"[ATTN]: Init norm {target_init_attn.norm()} | Delta norm {delta_attn.norm()} | Target norm {target_attn.norm()}",
        f"[MLP]: Init norm {target_init_mlp.norm()} | Delta norm {delta_mlp.norm()} | Target norm {target_mlp.norm()}",
    )

    return target_attn, target_mlp


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: aceHyperParams,
    layer: int,
    context_templates: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rank-1 update v* only on MLP path (legacy function, kept for compatibility with old calls).
    Adapted for models: hidden_size and single device.
    """
    device = next(model.parameters()).device

    lm_w = nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T
    ln_f = nethook.get_module(model, hparams.ln_f_module)
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size).to(device)

    print("Computing right vector (v)")

    target_ids = tok(request["target_new"]["str"], return_tensors="pt").to(device)["input_ids"][0]

    rewriting_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context_types in context_templates
        for context in context_types
    ]
    kl_prompts = ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to(device)

    rewriting_targets = torch.tensor(-100, device=device).repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids): ex_len] = target_ids

    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    embed_dim = getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd", None)
    if embed_dim is None:
        raise ValueError("Cannot find hidden size from model.config (expected 'hidden_size' or 'n_embd').")

    delta = torch.zeros((embed_dim,), requires_grad=True, device=device)
    target_init, kl_distr_init = None, None

    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        if cur_layer == hparams.mlp_module_tmp.format(layer):
            if target_init is None:
                print("Recording initial value of v*")
                target_init = cur_out[0, lookup_idxs[0]].detach().clone()
            for i, idx in enumerate(lookup_idxs):
                cur_out[i, idx, :] += delta
        return cur_out

    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)
    nll_loss_factor = hparams.nll_loss_factor

    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts):])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        full_repr = tr[hparams.layer_module_tmp.format(loss_layer)].output[0][: len(rewriting_prompts)]
        log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w + lm_b, dim=2)

        loss_nll_raw = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()
        max_probs = torch.max(log_probs, dim=2)[0]
        max_prob = torch.exp((max_probs * mask).sum(1) / target_ids.size(0)).mean().item()

        nll_loss_each = -(loss_nll_raw * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_factor * nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (torch.norm(delta) / torch.norm(target_init) ** 2)
        loss = nll_loss + kl_loss + weight_decay
        prob = torch.exp(-nll_loss_each).mean().item()

        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']['str']}] {prob}"
        )

        if loss < 5e-2:
            break
        if max_prob == prob:
            nll_loss_factor = 0.1 * hparams.nll_loss_factor
            if kl_loss <= 0.01:
                break
        else:
            nll_loss_factor = hparams.nll_loss_factor

        if it == hparams.v_num_grad_steps - 1:
            break

        loss.backward()
        opt.step()

        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta
    print(f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}")

    return target


def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, **context_info, **word_repr_args
        )
    elif fact_token_strategy == "last":
        raise Exception("This is definitely bugged, fix it.")
        context_info = dict(
            contexts=[
                tmp[i].format(words[i]) for i, tmp in enumerate(context_templates)
            ],
            idxs=[000000],
        )
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both", **context_info, **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return l_input.detach(), l_output.detach()

def get_modules_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_templates: List[str],
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the *inputs* of multiple layer modules.
    Used for parallel acquisition of input representations K (equation 19) for multiple 
    writable modules (typically attn and mlp).

    Return order is consistent with module_templates.
    """

    # Key fix: must use module_templates (plural) here, consistent with repr_tools interface
    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_templates=module_templates,
    )

    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_"):]
        # repr_tools interface: get_inputs_at_word_tokens(track="in", ..., module_templates=[...])
        # Returns a group of input representation tensors corresponding to module_templates
        l_inputs = repr_tools.get_inputs_at_word_tokens(
            track="in", subtoken=subtoken, **context_info, **word_repr_args
        )
        # Compatibility with old calls: legacy code expects two return values (attn, mlp)
        if not isinstance(l_inputs, (list, tuple)) or len(l_inputs) != len(module_templates):
            raise RuntimeError(
                f"Expected {len(module_templates)} input tensors, got {type(l_inputs)} with len={getattr(l_inputs, '.__len__', lambda: 'N/A')()}"
            )
        # Detach each one
        return tuple(t.detach() for t in l_inputs)
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret