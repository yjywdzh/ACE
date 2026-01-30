import json
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union, List, Dict
import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# from baselines.ft import FTHyperParams, apply_ft_to_model
# from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    get_tfidf_vectorizer,
)
from util.eval_utils.eval_utils_counterfact import compute_rewrite_quality_counterfact
from util.eval_utils.eval_utils_zsre import compute_rewrite_quality_zsre
from memit import MEMITHyperParams, apply_memit_to_model
from ace import aceHyperParams, apply_ace_to_model
# from rome import ROMEHyperParams, apply_rome_to_model
from util import nethook
from util.globals import *

from dsets.filterQA import FilteredQADataset

ALG_DICT = {
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "ace": (aceHyperParams, apply_ace_to_model),
    # "ROME": (ROMEHyperParams, apply_rome_to_model),
    # "FT": (FTHyperParams, apply_ft_to_model),
    # "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
    "filteredQA": (FilteredQADataset, compute_rewrite_quality_counterfact),
}


def load_model_and_tokenizer(model_name, model_path):
    """Load model and tokenizer, handling special requirements for different models"""
    
    if model_path:
        print(f"Instantiating model: {model_name} from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if "model_type_a" in model_name.lower() or "model_type_a" in model_path.lower():
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=None,
                trust_remote_code=True
            ).cuda()
        else:
            if "neox" in model_name:
                model = AutoModelForCausalLM.from_pretrained(model_path).half().cuda()
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
    else:
        print(f"Instantiating model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if "model_type_a" in model_name.lower():
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=None,
                trust_remote_code=True
            ).cuda()
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    dir_name: str,
    num_edits: int = 1,
    use_cache: bool = False,
    model_path: str = None,
    edit_layers: List[int] = None,
    use_cot: bool = True,
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory
    # Create new dir if not continuing from prev run OR prev run doesn't exist
    if (
        continue_from_run is None
        or not (run_dir := RESULTS_DIR / dir_name / continue_from_run).exists()
    ):
        continue_from_run = None
    if continue_from_run is None:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")

    # Get run hyperparameters
    params_path = (
        run_dir / "params.json"
        if continue_from_run is not None
        else HPARAMS_DIR / alg_name / hparams_fname
    )
    hparams = params_class.from_json(params_path)
    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing {alg_name} with parameters {hparams}")
    
    original_layers = hparams.layers.copy()
    
    if edit_layers:
        hparams.layers = edit_layers
        print(f"Using custom layers: {edit_layers}")
    
    if type(model_name) is str:
        model, tok = load_model_and_tokenizer(model_name, model_path)
    else:
        model, tok = model_name
        model_name = model.config._name_or_path
    
    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    if num_edits > 1:
        assert ds_name != "cf", f"{ds_name} does not support multiple edits"

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit)

    def ds_eval_with_cot(model, tok, record, snips=None, vec=None):
        """Evaluation function using Chain-of-Thought prompting"""
        cot_prompt = """
Question: {}
Thoughts: """
        
        if "compute_rewrite_quality_counterfact" in str(ds_eval_method):
            original_questions = {}
            if "questions" in record:
                original_questions["questions"] = record["questions"]
                record["questions"] = [cot_prompt.format(q) for q in record["questions"]]
            
            results = ds_eval_method(model, tok, record, snips, vec)
            
            if original_questions:
                record["questions"] = original_questions["questions"]
                
            return results
        else:
            return ds_eval_method(model, tok, record, snips, vec)
    
    eval_function = ds_eval_with_cot if use_cot else ds_eval_method

    # Get cache templates
    cache_template = None
    if use_cache:
        cache_template = (
            KV_DIR
            / f"{model_name.replace('/', '_')}_{alg_name}"
            / f"{ds_name}_layer_{{}}_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        print(f"Will load cache from {cache_template}")
    print(f"kvs cache template: {cache_template}")
    
    # Iterate through dataset
    for record in ds:
        case_result_template = str(run_dir / "{}_edits-case_{}.json")
        case_id = record["case_id"]
        
        already_finished = Path(case_result_template.format(num_edits, case_id)).exists()
        if already_finished:
            continue

        requested_rewrites = record["requested_rewrite"]
        if not isinstance(requested_rewrites, list):
            requested_rewrites = [requested_rewrites]
            
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["MEMIT", "ace", "ROME"]) else dict()
        
        if edit_layers:
            edit_requests = [{"case_id": case_id, **rewrite} for rewrite in requested_rewrites]
            
            start = time()
            model, weights_copy = apply_algo(
                model,
                tok,
                edit_requests,
                hparams,
                copy=False,
                return_orig_weights=True,
                **args_conserve_memory,
                **etc_args,
            )
        elif alg_name == "ace":
            for i, rewrite in enumerate(requested_rewrites):
                print(f"Processing case {case_id}, rewrite {i+1}/{len(requested_rewrites)}")
                
                edit_request = [{"case_id": case_id, **rewrite}]
                
                start = time()
                model, weights_copy = apply_algo(
                    model,
                    tok,
                    edit_request,
                    hparams,
                    copy=False,
                    return_orig_weights=True,
                    **args_conserve_memory,
                    **etc_args,
                )
        else:
            edit_requests = [{"case_id": case_id, **rewrite} for rewrite in requested_rewrites]
            
            start = time()
            model, weights_copy = apply_algo(
                model,
                tok,
                edit_requests,
                hparams,
                copy=False,
                return_orig_weights=True,
                **args_conserve_memory,
                **etc_args,
            )
            
        exec_time = time() - start
        print(f"Execution took {exec_time}")
        
        all_metrics = {
            "case_id": case_id,
            "num_edits": len(requested_rewrites),
            "requested_rewrites": requested_rewrites,
            "time": exec_time,
            "edit_layers": edit_layers,
            "use_cot": use_cot
        }
        
        print("Start evaluation")
        start = time()
        gen_test_vars = [snips, vec]
        
        post_eval = eval_function(
            model,
            tok,
            record,
            *(
                gen_test_vars
                if case_id % generation_test_interval == 0
                else [None, None]
            )
        )
        all_metrics["post"] = post_eval
        
        out_file = Path(case_result_template.format(num_edits, case_id))
        with open(out_file, "w") as f:
            json.dump(all_metrics, f, indent=1)
            
        with torch.no_grad():
            for k, v in weights_copy.items():
                nethook.get_parameter(model, k)[...] = v.to("cuda")
            
        print(f"Evaluation took {time() - start}")


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["MEMIT", "ace", "ROME"],
        default="MEMIT",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=False,
    )
    parser.add_argument(
        "--model_path",
        default=None
    )
    parser.add_argument(
        "--model_name",
        choices=["EleutherAI/gpt-neox-20b", "EleutherAI/gpt-j-6B", "ModelProvider/Model-8B"],
        default="EleutherAI/gpt-j-6B",
        help="Model to edit.",
        required=False,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="EleutherAI_gpt-j-6B.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=False,
    )
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "cf", "zsre", "filteredQA"],
        default="filteredQA",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), zsRE (zsre), or FilteredQA (filteredQA).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate dataset to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        default=True,
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--edit_layers",
        type=int,
        nargs="+",
        default=None,
        help="List of layers to edit (e.g., 24 31 for layers 24 and 31)",
    )
    parser.add_argument(
        "--use_cot",
        dest="use_cot",
        action="store_true",
        default=True,
        help="Use Chain-of-Thought prompting for evaluation",
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.conserve_memory,
        dir_name=args.alg_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        model_path=args.model_path,
        edit_layers=args.edit_layers,
        use_cot=args.use_cot,
    )