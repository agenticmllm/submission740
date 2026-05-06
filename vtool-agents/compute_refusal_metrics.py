import argparse
import numpy as np
import os
import sys
import pickle

from tqdm import tqdm


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, PROJECT_ROOT)

from red_team_utils import SafetyJudge, evaluate_arr

# Reuse model response parsers from compute_safety_metrics.py
from compute_safety_metrics import (
    parse_qwen3vl_response,
    parse_glm46v_response,
    parse_kimi_k25_response,
    parse_gpt5_response,
    parse_gemini3_response,
    parse_claude4_response,
    _refine_suffix,
)


API_MODEL_MAPPINGS = {
    "gpt4o": "gpt-4o",
    "gpt5": "gpt-5",
    "claude": "claude-opus-4",
}


# ========================================================================
# Helpers
# ========================================================================

def _is_refusal(judge_result) -> bool:
    return judge_result == 1


def _count_refusals(entries):
    return sum(1 for e in entries if _is_refusal(e.get("judge_result")))


# ========================================================================
# Dataset loading dispatch
# ========================================================================

def _load_entries(args):
    """Returns a list of entries for the given refusal dataset."""
    if args.dataset_name == "holisafe_refusal":
        from load_vl_overrefusal_dataset import load_holisafe_refusal
        entries = load_holisafe_refusal(no_pil_image=True)
        for entry in entries:
            entry["image_path"] = os.path.join(PROJECT_ROOT, entry["image_path"])
    elif args.dataset_name == "mssbench_refusal":
        from load_vl_overrefusal_dataset import load_mssbench_refusal
        entries = load_mssbench_refusal()
    elif args.dataset_name == "mossbench_refusal":
        from load_vl_overrefusal_dataset import load_mossbench_refusal
        entries = load_mossbench_refusal()
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")
    return entries


# ========================================================================
# Args
# ========================================================================

def arg_parse():
    parser = argparse.ArgumentParser(description="Compute refusal metrics (ARR) on over-refusal benchmarks")
    parser.add_argument("--dataset_name", type=str, required=True,
                        choices=["holisafe_refusal", "mssbench_refusal", "mossbench_refusal"])
    parser.add_argument("--model_output_path", type=str, default="./outputs")
    parser.add_argument("--eval_save_path", type=str, default="./eval_outputs")
    parser.add_argument("--save_every", type=int, default=30)

    parser.add_argument("--use_zoom_in", action="store_true")
    parser.add_argument("--use_code_interpreter", action="store_true")
    parser.add_argument("--use_tag", action="store_true")
    parser.add_argument("--use_ocr", action="store_true")
    parser.add_argument("--use_benign_ocr", action="store_true")

    parser.add_argument("--prompt_type", type=str, default="original_deep",
                        choices=["original", "simple", "original_deep", "no_tools", "no_tools_deep"])

    parser.add_argument("--model_name", type=str, required=True,
                        choices=["qwen3vl", "qwen35", "glm46v", "gpt", "gemini", "claude", "kimi_k25"])
    parser.add_argument("--disable_thinking", action="store_true")

    parser.add_argument("--compute_only_metrics", action="store_true")
    parser.add_argument("--base_url", type=str, default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--api_key", default=os.environ.get("API_KEY"), type=str)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--judge_model", type=str, default="gpt5")

    parser.add_argument("--refine_method", type=str, default=None,
                        choices=["mine", "adashield", "llamaguard", "taiji", "reminder"])
    parser.add_argument("--refine_prompt_type", type=str, default="original",
                        choices=["original", "safety_focus"])
    return parser.parse_args()


# ========================================================================
# Inference (single-split, vsl_bench-style)
# ========================================================================

def inference_on_refusal(args, parsed_id_2_result, res_file_name):

    ###########################################################
    # Load dataset
    ###########################################################
    print(f"Loading {args.dataset_name} dataset...")
    print()
    entries = _load_entries(args)
    print("--------------------------------")

    ###########################################################
    # Attach model responses (all benign → no gt_safe_label needed for MOSSBench prompt)
    ###########################################################
    eval_entries = []
    for entry in entries:
        sample_id = entry['sample_id']
        if sample_id not in parsed_id_2_result:
            continue
        entry['model_response'] = parsed_id_2_result[sample_id]['final_response']
        eval_entries.append(entry)

    print(f"Number of entries with model response: {len(eval_entries)}")
    print("--------------------------------")
    print()

    ###########################################################
    # Load temp results file if exists
    ###########################################################
    temp_res_file_name = res_file_name.replace(".pkl", "_temp.pkl")
    temp_res_file_path = os.path.join(args.eval_save_path, temp_res_file_name)

    sample_type = "all"

    if os.path.exists(temp_res_file_path):
        print(f"--- Loading temp results file: {temp_res_file_path}")
        with open(temp_res_file_path, "rb") as f:
            results_dict = pickle.load(f)
    else:
        print(f"--- Temp results file does not exist: {temp_res_file_path}")
        results_dict = {sample_type: {"finished_flag": False, "processed_entries": []}}

    ###########################################################
    # Load judge model (MOSSBench prompt, 0/1 output)
    ###########################################################
    print("Loading judge model...")
    print()
    judge = SafetyJudge(
        model_name=API_MODEL_MAPPINGS[args.judge_model],
        base_url=args.base_url,
        api_key=args.api_key,
        delay=args.delay,
        max_new_tokens=args.max_new_tokens,
        safety_prompt_type="mossbench",
    )

    ###########################################################
    # Evaluate ARR
    ###########################################################
    if results_dict[sample_type]["finished_flag"]:
        print(f"--- Results already finished: {temp_res_file_path}")
    else:
        print(f"Evaluating ARR for {args.dataset_name}...")
        print()
        results_dict = evaluate_arr(judge, eval_entries, results_dict, temp_res_file_path, sample_type, save_every=args.save_every)
        print("--------------------------------")
        print(f"Finished evaluating ARR for {args.dataset_name}")
        print(f"   ARR: ", results_dict[sample_type]["final_results"]["ARR"])
        with open(temp_res_file_path, "wb") as f:
            pickle.dump(results_dict, f)
            print(f"  --> Saved temp results file: {temp_res_file_path}")
            print()
        print("--------------------------------")
        print()

    ###########################################################
    # Save results
    ###########################################################
    print("Saving results...")
    print()
    save_path = os.path.join(args.eval_save_path, res_file_name)
    with open(save_path, "wb") as f:
        pickle.dump(results_dict, f)
    print(f"Saved results: {save_path}")
    print()
    print("--------------------------------")
    print()


# ========================================================================
# Metrics
# ========================================================================

def compute_refusal_metrics(args, parsed_id_2_result, res_file_name):

    # Statistics
    num_samples = len(parsed_id_2_result)
    num_samples_with_tool_use = sum(1 for s in parsed_id_2_result.values() if s['cnt_tool_use'] > 0)
    num_samples_with_multiple_tool_types = sum(1 for s in parsed_id_2_result.values() if s['num_tool_types'] > 1)
    num_samples_with_no_tool_use = num_samples - num_samples_with_tool_use
    print("------------------ Statistics -----------------------")
    print(f"Number of samples: {num_samples}")
    print(f"Number of samples with tool use: {num_samples_with_tool_use}")
    print(f"Number of samples with multiple tool types: {num_samples_with_multiple_tool_types}")
    print(f"Number of samples with no tool use: {num_samples_with_no_tool_use}")
    print()

    arr_results = pickle.load(open(os.path.join(args.eval_save_path, res_file_name), "rb"))

    sample_type = "all"
    fr = arr_results[sample_type]["final_results"]
    cnt = fr["counts"]

    # ---- Overall ARR ----
    print("----------------- ARR Results (all, lower is better) ---------------------")
    print(f"Total ARR: {fr['ARR']:.4f}   "
          f"(refusal={cnt['refusal']}, compliance={cnt['compliance']}, none={cnt['none']}, total={cnt['total']})")
    print("-------------------------------------------------------------------")
    print()

    # ---- ARR (tool-use only) ----
    print("----------------- ARR Results (tool-use only) ---------------------")
    tool_used_ids = {
        sid for sid, info in parsed_id_2_result.items()
        if info.get("cnt_tool_use", 0) > 0
    }

    proc = arr_results[sample_type]["processed_entries"]
    proc_tool = [e for e in proc if e.get("sample_id") in tool_used_ids]

    n = len(proc_tool)
    fail = _count_refusals(proc_tool)
    arr_tool = fail / n if n > 0 else float("nan")
    print(f"Total tool-only ARR: {arr_tool:.4f}   (refusal={fail}, total={n})")
    print("-------------------------------------------------------------------")
    print()

    # ---- ARR by number of tool calls ----
    print("----------------- ARR Results (by #tool calls) ---------------------")
    sid2k = {sid: info.get("cnt_tool_use", 0) for sid, info in parsed_id_2_result.items()}

    k2_total = {}
    k2_fail = {}
    for e in proc:
        sid = e.get("sample_id")
        if sid not in sid2k:
            continue
        k_raw = sid2k[sid]
        k = k_raw if k_raw < 5 else 5
        k2_total[k] = k2_total.get(k, 0) + 1
        if _is_refusal(e.get("judge_result")):
            k2_fail[k] = k2_fail.get(k, 0) + 1

    for k in sorted(k2_total.keys()):
        total_k = k2_total[k]
        fl = k2_fail.get(k, 0)
        label = f"k={k}" if k < 5 else "k>=5"
        print(f"{label}: ARR={fl/total_k:.4f}   (refusal={fl}, total={total_k})")
    print("-------------------------------------------------------------------")


# ========================================================================
# Main
# ========================================================================

def main(args):

    refine_suffix = _refine_suffix(args)

    ###########################################################
    # Load entries with responses
    ###########################################################
    print("Loading entries with responses...")
    print()
    if args.model_name == 'qwen3vl' or args.model_name == 'qwen35':
        file_name = f"{args.dataset_name}_id_2_{args.model_name}_tool_inference"
        if args.use_zoom_in: file_name += "_zoom_in"
        if args.use_code_interpreter: file_name += "_local_kernel_code_interpreter"
        if args.use_tag: file_name += "_tags"
        if args.use_ocr: file_name += "_ocr"
        if args.use_benign_ocr: file_name += "_benign_ocr"
    elif args.model_name == 'glm46v' or args.model_name == 'kimi_k25':
        if args.disable_thinking:
            file_name = f"{args.dataset_name}_id_2_{args.model_name}_nothink_agent_inference"
        else:
            file_name = f"{args.dataset_name}_id_2_{args.model_name}_agent_inference"
        if args.use_zoom_in: file_name += "_zoom_in"
        if args.use_tag: file_name += "_tags"
        if args.use_ocr: file_name += "_ocr"
        if args.use_benign_ocr: file_name += "_benign_ocr"
        if args.use_code_interpreter: file_name += "_code_interpreter"
    elif args.model_name == 'gpt' or args.model_name == 'gemini' or args.model_name == 'claude':
        file_name = f"{args.dataset_name}_id_2_openai_agent_inference_{args.model_name}"
        if args.use_zoom_in: file_name += "_zoom_in"
        if args.use_tag: file_name += "_tags"
        if args.use_ocr: file_name += "_ocr"
        if args.use_benign_ocr: file_name += "_benign_ocr"
        if args.use_code_interpreter: file_name += "_code_interpreter"
    else:
        raise ValueError(f"Invalid model name: {args.model_name}")

    file_name += f"_{args.prompt_type}"
    file_name += refine_suffix
    file_name += ".pkl"
    file_path = os.path.join(args.model_output_path, file_name)
    if os.path.exists(file_path):
        print(f"Loading result file: {file_path}")
        with open(file_path, "rb") as f:
            id_2_result = pickle.load(f)
    else:
        raise ValueError(f"Result file does not exist: {file_path}")

    # Result file name
    res_file_name = f"{args.dataset_name}_{args.model_name}"
    if args.disable_thinking: res_file_name += "_nothink"
    if args.use_zoom_in: res_file_name += "_zoom_in"
    if args.use_code_interpreter: res_file_name += "_code_interpreter"
    if args.use_tag: res_file_name += "_tags"
    if args.use_ocr: res_file_name += "_ocr"
    if args.use_benign_ocr: res_file_name += "_benign_ocr"
    res_file_name += f"_{args.prompt_type}"
    res_file_name += refine_suffix
    res_file_name += f"_judge_{args.judge_model}"
    res_file_name += "_refusal_eval_res.pkl"

    # Parse model responses
    if args.model_name == "qwen3vl" or args.model_name == "qwen35":
        parsed_id_2_result = parse_qwen3vl_response(id_2_result)
    elif args.model_name == "glm46v":
        parsed_id_2_result = parse_glm46v_response(id_2_result)
    elif args.model_name == "kimi_k25":
        parsed_id_2_result = parse_kimi_k25_response(id_2_result)
    elif args.model_name == "gpt":
        parsed_id_2_result = parse_gpt5_response(id_2_result)
    elif args.model_name == "gemini":
        parsed_id_2_result = parse_gemini3_response(id_2_result)
    elif args.model_name == "claude":
        parsed_id_2_result = parse_claude4_response(id_2_result)
    else:
        raise ValueError(f"Invalid model name: {args.model_name}")

    # Dispatch
    if not args.compute_only_metrics:
        inference_on_refusal(args, parsed_id_2_result, res_file_name)
    else:
        compute_refusal_metrics(args, parsed_id_2_result, res_file_name)


if __name__ == "__main__":
    args = arg_parse()
    if args.refine_method is not None:
        args.model_output_path = "./outputs_refined"
    main(args)