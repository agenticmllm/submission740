import argparse
import numpy as np
import random
import torch

from PIL import Image

import os
import re

import sys
from tqdm import tqdm
import pickle



PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, PROJECT_ROOT)

from load_vl_safety_dataset import load_holisafe, load_mm_safety_bench, load_vsl_bench
from red_team_utils import SafetyJudge, evaluate_asr, evaluate_rr

API_MODEL_MAPPINGS = {
    "gpt4o": "gpt-4o",
    "gpt5":  "gpt-5.2",
    "claude": "claude-opus-4-6",
}



def _is_vslbench_results(counts: dict) -> bool:
    return "unsafe" in counts


def _is_fail_lenient(judge_result) -> bool:
    return judge_result in ("No", "unsafe")


def _count_fails(entries):
    return sum(1 for e in entries if _is_fail_lenient(e.get("judge_result")))



def _refine_suffix(args) -> str:
    if args.refine_method is None:
        return ""
    if args.refine_method == "mine":
        return f"_refine_{args.refine_prompt_type}"
    return f"_refine_{args.refine_method}"


def arg_parse():
    parser = argparse.ArgumentParser(description="Compute safety metrics")
    parser.add_argument("--dataset_name", type=str, default="holisafe", help="Dataset name")
    parser.add_argument("--model_output_path", type=str, default="./outputs", help="Save path")
    parser.add_argument("--eval_save_path", type=str, default="./eval_outputs", help="Save path")
    parser.add_argument("--save_every", type=int, default=30, help="Save temp file every N steps")

    parser.add_argument("--use_zoom_in", action="store_true", help="Use zoom in tool")
    parser.add_argument("--use_code_interpreter", action="store_true", help="Use code interpreter tool")
    parser.add_argument("--use_tag", action="store_true", help="Use tag tool")
    parser.add_argument("--use_ocr", action="store_true", help="Use OCR tool")
    parser.add_argument("--use_benign_ocr", action="store_true", help="Use benign OCR tool")

    parser.add_argument("--prompt_type", type=str, default="original_deep", choices=["original", "original_deep", "no_tools", "no_tools_deep"], help="Prompt type")

    parser.add_argument("--model_name", type=str, required=True,
                        choices=["qwen3vl", "qwen35", "kimi_k25", "kimi_k26", "adareasoner",
                                 "gpt54", "gemini31", "gemini25", "claude46", "claude47", "glm5turbo"],
                        help="Tool-using model name (selects the response parser and the input pkl filename).")

    parser.add_argument("--disable_thinking", action="store_true", help="Disable Kimi-K2.5 thinking mode (default: thinking enabled)")

    parser.add_argument("--compute_only_metrics", action="store_true", help="Only compute metrics")
    parser.add_argument("--base_url", type=str,
                    default=os.environ.get("OPENAI_BASE_URL", ""),
                    help="Base URL for OpenAI-compatible API")
    parser.add_argument("--api_key", default=os.environ.get("API_KEY"), type=str)
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between API calls")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of new tokens to generate.")
    parser.add_argument("--judge_model", type=str, default="gpt5")
    parser.add_argument("--safety_prompt_type", type=str, default="vslbench", choices=["original", "holisafe", "vslbench", "mm_safety_bench"])

    parser.add_argument("--refine_method", type=str, default=None, choices=["mine", "adashield", "llamaguard", "taiji", "reminder"], help="Refinement method")
    parser.add_argument("--refine_prompt_type", type=str, default="original", choices=["original", "safety_focus"], help="Reflection prompt type used by --refine_method mine")

    parser.add_argument("--reinject_on_final", action="store_true",
                        help="For samples that used tools, re-inject original image+query "
                             "right before the final answer (tools disabled during reinject). "
                             "Measures context dilution.")

    parser.add_argument("--fixed_zoom_in", action="store_true", help="Use fixed zoom in tool instead of the dynamic zoom in tool.")
    return parser.parse_args()



def parse_qwen3vl_response(id_2_result):

    parsed_id_2_result = {}

    for sample_id, response in tqdm(id_2_result.items()):
        all_turn_responses = []
        tool_use_list = []
        if 'error' in response:
            continue
        elif len(response) == 1:
            final_response = response[0]['content']
            all_turn_responses.append(final_response)
        else:
            for turn_response in response:
                if turn_response['role'] == 'function':
                    tool_use_list.append(turn_response['name'])
                elif turn_response['role'] == 'assistant':
                    if turn_response['content'] == '':
                        continue
                    else:
                        all_turn_responses.append(turn_response['content'])
                else:
                    raise ValueError(f"Invalid role: {turn_response['role']}")
            if not all_turn_responses:
                continue
            final_response = all_turn_responses[-1]

        parsed_id_2_result[sample_id] = {
            'final_response': final_response,
            'all_turn_responses': all_turn_responses,
            'tool_use_list': tool_use_list,
            'cnt_tool_use': len(tool_use_list),
            'num_tool_types': len(set(tool_use_list)),
        }

    tool_use_stats = {}
    for entry in parsed_id_2_result.values():
        for tool_name in entry['tool_use_list']:
            if tool_name not in tool_use_stats:
                tool_use_stats[tool_name] = 0
            tool_use_stats[tool_name] += 1

    print("--------------------------------")
    print(f"Number of samples: {len(parsed_id_2_result)}")
    print(f"Number of samples with tool use: {sum(1 for sample in parsed_id_2_result.values() if sample['cnt_tool_use'] > 0)}")
    print(f"Number of samples with multiple tool types: {sum(1 for sample in parsed_id_2_result.values() if sample['num_tool_types'] > 1)}")
    print()
    print("Tool use statistics:")
    for tool_name, count in tool_use_stats.items():
        print(f"  {tool_name}: {count}")
    print()
    print("--------------------------------")

    return parsed_id_2_result


def parse_kimi_k25_response(id_2_result):

    parsed_id_2_result = {}

    for sample_id, response in tqdm(id_2_result.items()):
        all_turn_responses = []
        tool_use_list = []
        react_turns = 0

        if isinstance(response, dict) and 'error' in response:
            continue
        if not isinstance(response, list) or len(response) == 0:
            continue

        for turn in response:
            role = turn.get('role', '')

            if role in ('system', 'user'):
                continue
            elif role == 'tool':
                tool_use_list.append(turn.get('name', 'unknown_tool'))
            elif role == 'assistant':
                if 'tool_calls' in turn and turn['tool_calls']:
                    react_turns += 1

                content = turn.get('content', '')
                if not content or not content.strip():
                    continue
                all_turn_responses.append(content)

        if not all_turn_responses:
            continue

        final_response = all_turn_responses[-1]

        final_response = re.sub(r'<think>.*?</think>', '', final_response, flags=re.DOTALL)
        final_response = re.sub(r'</think>\s*', '', final_response)
        final_response = final_response.strip()

        parsed_id_2_result[sample_id] = {
            'final_response': final_response,
            'all_turn_responses': all_turn_responses,
            'tool_use_list': tool_use_list,
            'cnt_tool_use': len(tool_use_list),
            'num_tool_types': len(set(tool_use_list)),
            'cnt_react_turns': react_turns,
        }

    tool_use_stats = {}
    for entry in parsed_id_2_result.values():
        for tool_name in entry['tool_use_list']:
            if tool_name not in tool_use_stats:
                tool_use_stats[tool_name] = 0
            tool_use_stats[tool_name] += 1

    print("--------------------------------")
    print(f"Number of samples: {len(parsed_id_2_result)}")
    print(f"Number of samples with tool use: {sum(1 for s in parsed_id_2_result.values() if s['cnt_tool_use'] > 0)}")
    print(f"Number of samples with multiple tool types: {sum(1 for s in parsed_id_2_result.values() if s['num_tool_types'] > 1)}")
    print()
    print("Tool use statistics:")
    for tool_name, count in tool_use_stats.items():
        print(f"  {tool_name}: {count}")
    print()
    print("ReAct turn statistics (all samples):")
    react_counts = [s['cnt_react_turns'] for s in parsed_id_2_result.values()]
    if react_counts:
        print(f"  Avg ReAct turns: {np.mean(react_counts):.2f}")
        print(f"  Max ReAct turns: {max(react_counts)}")
    else:
        print("  No samples found")
    print()
    print("ReAct turn statistics (tool-use samples only):")
    tool_use_samples = [s for s in parsed_id_2_result.values() if s['cnt_tool_use'] > 0]
    if tool_use_samples:
        react_counts_tool = [s['cnt_react_turns'] for s in tool_use_samples]
        print(f"  Num tool-use samples: {len(tool_use_samples)}")
        print(f"  Avg ReAct turns: {np.mean(react_counts_tool):.2f}")
        print(f"  Max ReAct turns: {max(react_counts_tool)}")
        print(f"  Avg tool calls per ReAct turn: {np.mean([s['cnt_tool_use'] / s['cnt_react_turns'] for s in tool_use_samples if s['cnt_react_turns'] > 0]):.2f}")
    else:
        print("  No tool-use samples found")
    print("--------------------------------")

    return parsed_id_2_result


def parse_gpt5_response(id_2_result):
    return parse_gemini3_response(id_2_result)

def parse_gemini3_response(id_2_result):

    parsed_id_2_result = {}

    for sample_id, response in tqdm(id_2_result.items()):
        all_turn_responses = []
        tool_use_list = []
        react_turns = 0

        if isinstance(response, dict) and 'error' in response:
            continue
        if not isinstance(response, list) or len(response) == 0:
            continue

        for turn in response:
            role = turn.get('role', '')

            if role in ('system', 'user'):
                continue
            elif role == 'tool':
                tool_use_list.append(turn['name'])
            elif role == 'assistant':
                if 'tool_calls' in turn and turn['tool_calls']:
                    react_turns += 1

                content = turn.get('content', '')
                if not content or not content.strip():
                    continue
                all_turn_responses.append(content)

        if not all_turn_responses:
            continue
        final_response = all_turn_responses[-1]

        parsed_id_2_result[sample_id] = {
            'final_response': final_response,
            'all_turn_responses': all_turn_responses,
            'tool_use_list': tool_use_list,
            'cnt_tool_use': len(tool_use_list),
            'num_tool_types': len(set(tool_use_list)),
            'cnt_react_turns': react_turns,
        }

    tool_use_stats = {}
    for entry in parsed_id_2_result.values():
        for tool_name in entry['tool_use_list']:
            if tool_name not in tool_use_stats:
                tool_use_stats[tool_name] = 0
            tool_use_stats[tool_name] += 1

    print("--------------------------------")
    print(f"Number of samples: {len(parsed_id_2_result)}")
    print(f"Number of samples with tool use: {sum(1 for s in parsed_id_2_result.values() if s['cnt_tool_use'] > 0)}")
    print(f"Number of samples with multiple tool types: {sum(1 for s in parsed_id_2_result.values() if s['num_tool_types'] > 1)}")
    print()
    print("Tool use statistics:")
    for tool_name, count in tool_use_stats.items():
        print(f"  {tool_name}: {count}")
    print()
    print("ReAct turn statistics (all samples):")
    react_counts = [s['cnt_react_turns'] for s in parsed_id_2_result.values()]
    print(f"  Avg ReAct turns: {np.mean(react_counts):.2f}")
    print(f"  Max ReAct turns: {max(react_counts)}")
    print()
    print("ReAct turn statistics (tool-use samples only):")
    tool_use_samples = [s for s in parsed_id_2_result.values() if s['cnt_tool_use'] > 0]
    if tool_use_samples:
        react_counts_tool = [s['cnt_react_turns'] for s in tool_use_samples]
        print(f"  Num tool-use samples: {len(tool_use_samples)}")
        print(f"  Avg ReAct turns: {np.mean(react_counts_tool):.2f}")
        print(f"  Max ReAct turns: {max(react_counts_tool)}")
        print(f"  Avg tool calls per ReAct turn: {np.mean([s['cnt_tool_use'] / s['cnt_react_turns'] for s in tool_use_samples if s['cnt_react_turns'] > 0]):.2f}")
    else:
        print("  No tool-use samples found")
    print()
    print("--------------------------------")

    return parsed_id_2_result



def parse_claude4_response(id_2_result):

    parsed_id_2_result = {}

    for sample_id, response in tqdm(id_2_result.items()):
        all_turn_responses = []
        tool_use_list = []
        react_turns = 0

        if isinstance(response, dict) and 'error' in response:
            continue
        if not isinstance(response, list) or len(response) == 0:
            continue

        for turn in response:
            role = turn.get('role', '')

            if role in ('system', 'user'):
                continue
            elif role == 'tool':
                tool_use_list.append(turn['name'])
            elif role == 'assistant':
                if 'tool_calls' in turn and turn['tool_calls']:
                    react_turns += 1

                content = turn.get('content', '')
                if not content or not content.strip():
                    continue
                all_turn_responses.append(content)

        if not all_turn_responses:
            continue
        final_response = all_turn_responses[-1]

        parsed_id_2_result[sample_id] = {
            'final_response': final_response,
            'all_turn_responses': all_turn_responses,
            'tool_use_list': tool_use_list,
            'cnt_tool_use': len(tool_use_list),
            'num_tool_types': len(set(tool_use_list)),
            'cnt_react_turns': react_turns,
        }

    tool_use_stats = {}
    for entry in parsed_id_2_result.values():
        for tool_name in entry['tool_use_list']:
            if tool_name not in tool_use_stats:
                tool_use_stats[tool_name] = 0
            tool_use_stats[tool_name] += 1

    print("--------------------------------")
    print(f"Number of samples: {len(parsed_id_2_result)}")
    print(f"Number of samples with tool use: {sum(1 for s in parsed_id_2_result.values() if s['cnt_tool_use'] > 0)}")
    print(f"Number of samples with multiple tool types: {sum(1 for s in parsed_id_2_result.values() if s['num_tool_types'] > 1)}")
    print()
    print("Tool use statistics:")
    for tool_name, count in tool_use_stats.items():
        print(f"  {tool_name}: {count}")
    print()
    print("ReAct turn statistics (all samples):")
    react_counts = [s['cnt_react_turns'] for s in parsed_id_2_result.values()]
    print(f"  Avg ReAct turns: {np.mean(react_counts):.2f}")
    print(f"  Max ReAct turns: {max(react_counts)}")
    print()
    print("ReAct turn statistics (tool-use samples only):")
    tool_use_samples = [s for s in parsed_id_2_result.values() if s['cnt_tool_use'] > 0]
    if tool_use_samples:
        react_counts_tool = [s['cnt_react_turns'] for s in tool_use_samples]
        print(f"  Num tool-use samples: {len(tool_use_samples)}")
        print(f"  Avg ReAct turns: {np.mean(react_counts_tool):.2f}")
        print(f"  Max ReAct turns: {max(react_counts_tool)}")
        print(f"  Avg tool calls per ReAct turn: {np.mean([s['cnt_tool_use'] / s['cnt_react_turns'] for s in tool_use_samples if s['cnt_react_turns'] > 0]):.2f}")
    else:
        print("  No tool-use samples found")
    print("--------------------------------")

    return parsed_id_2_result




def inference_on_holisafe(args, parsed_id_2_result, res_file_name):

    print("Loading holisafe dataset...")
    print()
    entries = load_holisafe(no_pil_image=False)
    for entry in entries:
        entry['image_path'] = os.path.join(PROJECT_ROOT, entry['image_path'])
    entries = [entry for entry in entries if entry['sample_type'] != "SSS"]
    print("--------------------------------")

    print("Separating entries into different categories...")
    print()
    sss_entries, ssu_entries, usu_entries, suu_entries, uuu_entries = [], [], [], [], []
    for entry in entries:
        sample_id = entry['sample_id']
        if sample_id not in parsed_id_2_result:
            continue
        final_response = parsed_id_2_result[sample_id]['final_response']
        entry['model_response'] = final_response
        if entry["sample_type"] == "SSS":
            sss_entries.append(entry)
        elif entry["sample_type"] == "SSU":
            ssu_entries.append(entry)
        elif entry["sample_type"] == "USU":
            usu_entries.append(entry)
        elif entry["sample_type"] == "SUU":
            suu_entries.append(entry)
        elif entry["sample_type"] == "UUU":
            uuu_entries.append(entry)
    print("--------------------------------")
    print(f"Number of SSU entries: {len(ssu_entries)}")
    print(f"Number of USU entries: {len(usu_entries)}")
    print(f"Number of SUU entries: {len(suu_entries)}")
    print(f"Number of UUU entries: {len(uuu_entries)}")
    print("--------------------------------")
    print()

    temp_res_file_name = res_file_name.replace(".pkl", "_temp.pkl")
    temp_res_file_path = os.path.join(args.eval_save_path, temp_res_file_name)

    if os.path.exists(temp_res_file_path):
        print(f"--- Loading temp results file: {temp_res_file_path}")
        with open(temp_res_file_path, "rb") as f:
            results_dict = pickle.load(f)
    else:
        print(f"--- Temp results file does not exist: {temp_res_file_path}")
        results_dict = {"SSS": {"finished_flag": False, "processed_entries": []}, "SSU": {"finished_flag": False, "processed_entries": []}, "USU": {"finished_flag": False, "processed_entries": []}, "SUU": {"finished_flag": False, "processed_entries": []}, "UUU": {"finished_flag": False, "processed_entries": []}}
    
    
    print("Loading judge model...")
    print()
    judge = SafetyJudge(
        model_name=API_MODEL_MAPPINGS[args.judge_model],
        base_url=args.base_url,
        api_key=args.api_key,
        delay=args.delay,
        max_new_tokens=args.max_new_tokens,
        safety_prompt_type=args.safety_prompt_type,
    )
        
    print("Evaluating ASR for unsafe samples...")
    print()


    for split_name, split_entries in [("SSU", ssu_entries), ("USU", usu_entries), ("SUU", suu_entries), ("UUU", uuu_entries)]:
        if results_dict[split_name]["finished_flag"]:
            print(f"--- {split_name} results already finished: {temp_res_file_path}")
        else:
            print(f"--- Evaluating ASR for {split_name}...")
            print()
            results_dict = evaluate_asr(judge, split_entries, results_dict, temp_res_file_path, split_name, save_every=args.save_every)
            print("--------------------------------")
            print(f"Finished evaluating ASR for {split_name}")
            print(f"   ASR for {split_name}: ", results_dict[split_name]["final_results"]["ASR"])
            with open(temp_res_file_path, "wb") as f:
                pickle.dump(results_dict, f)
                print(f"  --> Saved temp results file: {temp_res_file_path}")
                print()
            print('--------------------------------')
            print()

    print("Saving results...")
    print()
    save_path = os.path.join(args.eval_save_path, res_file_name)
    with open(save_path, "wb") as f:
        pickle.dump(results_dict, f)
    print(f"Saved results: {save_path}")
    print()
    print("--------------------------------")
    print()




def inference_on_mm_safety_bench(args, parsed_id_2_result, res_file_name):

    print("Loading mm_safety_bench dataset...")
    print()
    entries = load_mm_safety_bench(no_pil_image=False)
    for entry in entries:
        entry['image_path'] = os.path.join(PROJECT_ROOT, entry['image_path'])
    print("--------------------------------")

    print("Separating entries into different categories...")
    print()
    cat_2_entries = {}
    for entry in entries:
        sample_id = entry['sample_id']
        if sample_id not in parsed_id_2_result:
            continue
        final_response = parsed_id_2_result[sample_id]['final_response']
        entry['model_response'] = final_response

        cat = entry['safety_cat']
        if cat not in cat_2_entries:
            cat_2_entries[cat] = []
        cat_2_entries[cat].append(entry)

    print("--------------------------------")
    for cat, cat_entries in cat_2_entries.items():
        print(f"  {cat}: {len(cat_entries)}")
    print("--------------------------------")
    print()

    temp_res_file_name = res_file_name.replace(".pkl", "_temp.pkl")
    temp_res_file_path = os.path.join(args.eval_save_path, temp_res_file_name)

    if os.path.exists(temp_res_file_path):
        print(f"--- Loading temp results file: {temp_res_file_path}")
        with open(temp_res_file_path, "rb") as f:
            results_dict = pickle.load(f)
    else:
        print(f"--- Temp results file does not exist: {temp_res_file_path}")
        results_dict = {cat: {"finished_flag": False, "processed_entries": []} for cat in cat_2_entries.keys()}

    print("Loading judge model...")
    print()
    judge = SafetyJudge(
        model_name=API_MODEL_MAPPINGS[args.judge_model],
        base_url=args.base_url,
        api_key=args.api_key,
        delay=args.delay,
        max_new_tokens=args.max_new_tokens,
        safety_prompt_type=args.safety_prompt_type,
    )

    print("Evaluating ASR for each category...")
    print()

    for cat, cat_entries in cat_2_entries.items():
        if results_dict[cat]["finished_flag"]:
            print(f"--- {cat} results already finished: {temp_res_file_path}")
            continue

        print(f"--- Evaluating ASR for {cat}...")
        print()
        results_dict = evaluate_asr(judge, cat_entries, results_dict, temp_res_file_path, cat, save_every=args.save_every)
        print("--------------------------------")
        print(f"Finished evaluating ASR for {cat}")
        print(f"   ASR for {cat}: ", results_dict[cat]["final_results"]["ASR"])
        with open(temp_res_file_path, "wb") as f:
            pickle.dump(results_dict, f)
            print(f"  --> Saved temp results file: {temp_res_file_path}")
            print()
        print("--------------------------------")
        print()

    print("Saving results...")
    print()
    save_path = os.path.join(args.eval_save_path, res_file_name)
    with open(save_path, "wb") as f:
        pickle.dump(results_dict, f)
    print(f"Saved results: {save_path}")
    print()
    print("--------------------------------")
    print()



def inference_on_vsl_bench(args, parsed_id_2_result, res_file_name):

    print("Loading vsl_bench dataset...")
    print()
    entries = load_vsl_bench(no_pil_image=False)
    for entry in entries:
        entry['image_path'] = os.path.join(PROJECT_ROOT, entry['image_path'])
    print("--------------------------------")

    eval_entries = []
    for entry in entries:
        sample_id = entry['sample_id']
        if sample_id not in parsed_id_2_result:
            continue
        entry['model_response'] = parsed_id_2_result[sample_id]['final_response']
        entry['gt_safe_label'] = 'unsafe'
        eval_entries.append(entry)

    print(f"Number of entries with model response: {len(eval_entries)}")
    print("--------------------------------")
    print()

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

    print("Loading judge model...")
    print()
    judge = SafetyJudge(
        model_name=API_MODEL_MAPPINGS[args.judge_model],
        base_url=args.base_url,
        api_key=args.api_key,
        delay=args.delay,
        max_new_tokens=args.max_new_tokens,
        safety_prompt_type=args.safety_prompt_type,
    )

    if results_dict[sample_type]["finished_flag"]:
        print(f"--- Results already finished: {temp_res_file_path}")
    else:
        print("Evaluating ASR for vsl_bench...")
        print()
        results_dict = evaluate_asr(judge, eval_entries, results_dict, temp_res_file_path, sample_type, save_every=args.save_every)
        print("--------------------------------")
        print(f"Finished evaluating ASR for vsl_bench")
        print(f"   ASR: ", results_dict[sample_type]["final_results"]["ASR"])
        with open(temp_res_file_path, "wb") as f:
            pickle.dump(results_dict, f)
            print(f"  --> Saved temp results file: {temp_res_file_path}")
            print()
        print("--------------------------------")
        print()

    print("Saving results...")
    print()
    save_path = os.path.join(args.eval_save_path, res_file_name)
    with open(save_path, "wb") as f:
        pickle.dump(results_dict, f)
    print(f"Saved results: {save_path}")
    print()
    print("--------------------------------")
    print()



def compute_safety_metrics_on_holisafe(args, parsed_id_2_result, res_file_name):
    
    num_samples = len(parsed_id_2_result)
    num_samples_with_tool_use = sum(1 for sample in parsed_id_2_result.values() if sample['cnt_tool_use'] > 0)
    num_samples_with_multiple_tool_types = sum(1 for sample in parsed_id_2_result.values() if sample['num_tool_types'] > 1)
    num_samples_with_no_tool_use = num_samples - num_samples_with_tool_use - num_samples_with_multiple_tool_types
    print("------------------ Statistics -----------------------")
    print(f"Number of samples: {num_samples}")
    print(f"Number of samples with tool use: {num_samples_with_tool_use}")
    print(f"Number of samples with multiple tool types: {num_samples_with_multiple_tool_types}")
    print(f"Number of samples with no tool use: {num_samples_with_no_tool_use}")
    print()
    
    asr_results = pickle.load(open(os.path.join(args.eval_save_path, res_file_name), "rb"))

    first_counts = asr_results["SSU"]["final_results"]["counts"]
    is_vslbench = _is_vslbench_results(first_counts)

    splits = ["SSU", "USU", "SUU", "UUU"]

    print("----------------- ASR Results (all) ---------------------")
    total_fail_lenient = 0
    total_n = 0

    for sp in splits:
        fr = asr_results[sp]["final_results"]
        cnt = fr["counts"]
        n = cnt["total"]
        total_n += n

        if is_vslbench:
            asr_l = fr.get("ASR_lenient", fr["ASR"])
            fail_l = cnt.get("unsafe", 0)
            total_fail_lenient += fail_l
            print(f"ASR for {sp}: {asr_l:.4f}   "
                  f"(unsafe={cnt.get('unsafe',0)}, warning={cnt.get('safe_with_warning',0)}, "
                  f"refusal={cnt.get('safe_with_refusal',0)}, total={n})")
        else:
            asr_val = fr["ASR"]
            no = cnt["no"]
            total_fail_lenient += no
            print(f"ASR for {sp}: {asr_val:.4f}   (no={no}, total={n})")

    total_asr = total_fail_lenient / total_n if total_n > 0 else float("nan")
    print()
    print(f"Total ASR: {total_asr:.4f}")
    print("-------------------------------------------------------------------")
    print()

    print("----------------- ASR Results (tool-use only) ---------------------")

    tool_used_ids = {
        sid for sid, info in parsed_id_2_result.items()
        if info.get("cnt_tool_use", 0) > 0
    }

    total_fail_l = 0
    total_n = 0

    for sp in splits:
        proc = asr_results[sp]["processed_entries"]
        proc_tool = [e for e in proc if e.get("sample_id") in tool_used_ids]

        n = len(proc_tool)
        fail_l = _count_fails(proc_tool)

        total_fail_l += fail_l
        total_n += n

        asr_l = fail_l / n if n > 0 else float("nan")
        print(f"{sp} tool-only ASR: {asr_l:.4f}   (fail={fail_l}, total={n})")

    total_tool_asr = total_fail_l / total_n if total_n > 0 else float("nan")
    print()
    print(f"Total tool-only ASR: {total_tool_asr:.4f}   (fail={total_fail_l}, total={total_n})")
    print("-------------------------------------------------------------------")
    print()

    print("----------------- ASR Results (by #tool calls) ---------------------")

    sid2k = {sid: info.get("cnt_tool_use", 0) for sid, info in parsed_id_2_result.items()}

    k2_total = {}
    k2_fail_l = {}

    for sp in splits:
        proc = asr_results[sp]["processed_entries"]
        for e in proc:
            sid = e.get("sample_id")
            if sid not in sid2k:
                continue

            k_raw = sid2k[sid]
            k = k_raw if k_raw < 5 else 5

            k2_total[k] = k2_total.get(k, 0) + 1
            if _is_fail_lenient(e.get("judge_result")):
                k2_fail_l[k] = k2_fail_l.get(k, 0) + 1

    for k in sorted(k2_total.keys()):
        total_k = k2_total[k]
        fl = k2_fail_l.get(k, 0)
        label = f"k={k}" if k < 5 else "k>=5"
        print(f"{label}: ASR={fl/total_k:.4f}   (fail={fl}, total={total_k})")

    print("-------------------------------------------------------------------")




def compute_safety_metrics_on_mm_safety_bench(args, parsed_id_2_result, res_file_name):

    num_samples = len(parsed_id_2_result)
    num_samples_with_tool_use = sum(1 for sample in parsed_id_2_result.values() if sample['cnt_tool_use'] > 0)
    num_samples_with_multiple_tool_types = sum(1 for sample in parsed_id_2_result.values() if sample['num_tool_types'] > 1)
    num_samples_with_no_tool_use = num_samples - num_samples_with_tool_use
    print("------------------ Statistics -----------------------")
    print(f"Number of samples: {num_samples}")
    print(f"Number of samples with tool use: {num_samples_with_tool_use}")
    print(f"Number of samples with multiple tool types: {num_samples_with_multiple_tool_types}")
    print(f"Number of samples with no tool use: {num_samples_with_no_tool_use}")
    print()

    asr_results = pickle.load(open(os.path.join(args.eval_save_path, res_file_name), "rb"))

    first_key = sorted(asr_results.keys())[0]
    first_counts = asr_results[first_key]["final_results"]["counts"]
    is_vslbench = _is_vslbench_results(first_counts)

    print("----------------- ASR Results (all) ---------------------")
    total_fail_l = 0
    total_n = 0
    for cat in sorted(asr_results.keys()):
        fr = asr_results[cat]["final_results"]
        cnt = fr["counts"]
        n = cnt["total"]
        total_n += n

        if is_vslbench:
            asr_l = fr.get("ASR_lenient", fr["ASR"])
            fl = cnt.get("unsafe", 0)
            total_fail_l += fl
            print(f"ASR for {cat}: {asr_l:.4f}   (unsafe={fl}, warning={cnt.get('safe_with_warning',0)}, total={n})")
        else:
            asr_val = fr["ASR"]
            no = cnt["no"]
            total_fail_l += no
            print(f"ASR for {cat}: {asr_val:.4f}   (no={no}, total={n})")

    total_asr = total_fail_l / total_n if total_n > 0 else float("nan")
    print()
    print(f"Total ASR: {total_asr:.4f}   (fail={total_fail_l}, total={total_n})")
    print("-------------------------------------------------------------------")
    print()

    print("----------------- ASR Results (tool-use only) ---------------------")
    tool_used_ids = {
        sid for sid, info in parsed_id_2_result.items()
        if info.get("cnt_tool_use", 0) > 0
    }

    total_fail_l = 0
    total_n = 0
    for cat in sorted(asr_results.keys()):
        proc = asr_results[cat]["processed_entries"]
        proc_tool = [e for e in proc if e.get("sample_id") in tool_used_ids]

        n = len(proc_tool)
        fl = _count_fails(proc_tool)
        total_fail_l += fl
        total_n += n

        asr_l = fl / n if n > 0 else float("nan")
        print(f"{cat} tool-only ASR: {asr_l:.4f}   (fail={fl}, total={n})")

    total_tool_asr = total_fail_l / total_n if total_n > 0 else float("nan")
    print()
    print(f"Total tool-only ASR: {total_tool_asr:.4f}   (fail={total_fail_l}, total={total_n})")
    print("-------------------------------------------------------------------")
    print()

    print("----------------- ASR Results (by #tool calls) ---------------------")
    sid2k = {sid: info.get("cnt_tool_use", 0) for sid, info in parsed_id_2_result.items()}

    k2_total = {}
    k2_fail_l = {}

    for cat in asr_results.keys():
        proc = asr_results[cat]["processed_entries"]
        for e in proc:
            sid = e.get("sample_id")
            if sid not in sid2k:
                continue
            k_raw = sid2k[sid]
            k = k_raw if k_raw < 5 else 5
            k2_total[k] = k2_total.get(k, 0) + 1
            if _is_fail_lenient(e.get("judge_result")):
                k2_fail_l[k] = k2_fail_l.get(k, 0) + 1

    for k in sorted(k2_total.keys()):
        total_k = k2_total[k]
        fl = k2_fail_l.get(k, 0)
        label = f"k={k}" if k < 5 else "k>=5"
        print(f"{label}: ASR={fl/total_k:.4f}   (fail={fl}, total={total_k})")

    print("-------------------------------------------------------------------")



def compute_safety_metrics_on_vsl_bench(args, parsed_id_2_result, res_file_name):

    num_samples = len(parsed_id_2_result)
    num_samples_with_tool_use = sum(1 for sample in parsed_id_2_result.values() if sample['cnt_tool_use'] > 0)
    num_samples_with_multiple_tool_types = sum(1 for sample in parsed_id_2_result.values() if sample['num_tool_types'] > 1)
    num_samples_with_no_tool_use = num_samples - num_samples_with_tool_use
    print("------------------ Statistics -----------------------")
    print(f"Number of samples: {num_samples}")
    print(f"Number of samples with tool use: {num_samples_with_tool_use}")
    print(f"Number of samples with multiple tool types: {num_samples_with_multiple_tool_types}")
    print(f"Number of samples with no tool use: {num_samples_with_no_tool_use}")
    print()

    asr_results = pickle.load(open(os.path.join(args.eval_save_path, res_file_name), "rb"))

    sample_type = "all"
    fr = asr_results[sample_type]["final_results"]
    cnt = fr["counts"]
    is_vslbench = _is_vslbench_results(cnt)

    print("----------------- ASR Results (all) ---------------------")
    if is_vslbench:
        asr_l = fr.get("ASR_lenient", fr["ASR"])
        print(f"Total ASR: {asr_l:.4f}   "
              f"(unsafe={cnt.get('unsafe',0)}, warning={cnt.get('safe_with_warning',0)}, "
              f"refusal={cnt.get('safe_with_refusal',0)}, total={cnt['total']})")
    else:
        asr = fr["ASR"]
        print(f"Total ASR: {asr:.4f}   (no={cnt['no']}, total={cnt['total']})")
    print("-------------------------------------------------------------------")
    print()

    print("----------------- ASR Results (tool-use only) ---------------------")
    tool_used_ids = {
        sid for sid, info in parsed_id_2_result.items()
        if info.get("cnt_tool_use", 0) > 0
    }

    proc = asr_results[sample_type]["processed_entries"]
    proc_tool = [e for e in proc if e.get("sample_id") in tool_used_ids]

    n = len(proc_tool)
    fl = _count_fails(proc_tool)
    asr_l = fl / n if n > 0 else float("nan")
    print(f"Total tool-only ASR: {asr_l:.4f}   (fail={fl}, total={n})")
    print("-------------------------------------------------------------------")
    print()

    print("----------------- ASR Results (by #tool calls) ---------------------")
    sid2k = {sid: info.get("cnt_tool_use", 0) for sid, info in parsed_id_2_result.items()}

    k2_total = {}
    k2_fail_l = {}

    for e in proc:
        sid = e.get("sample_id")
        if sid not in sid2k:
            continue
        k_raw = sid2k[sid]
        k = k_raw if k_raw < 5 else 5
        k2_total[k] = k2_total.get(k, 0) + 1
        if _is_fail_lenient(e.get("judge_result")):
            k2_fail_l[k] = k2_fail_l.get(k, 0) + 1

    for k in sorted(k2_total.keys()):
        total_k = k2_total[k]
        fl_k = k2_fail_l.get(k, 0)
        label = f"k={k}" if k < 5 else "k>=5"
        print(f"{label}: ASR={fl_k/total_k:.4f}   (fail={fl_k}, total={total_k})")

    print("-------------------------------------------------------------------")





def main(args):

    refine_suffix = _refine_suffix(args)

    print("Loading entries with responses...")
    print()
    if args.model_name == 'qwen3vl' or args.model_name == 'qwen35':
        file_name = f"{args.dataset_name}_id_2_{args.model_name}_tool_inference"
        if args.use_zoom_in: file_name += "_zoom_in"
        if args.use_code_interpreter: file_name += "_local_kernel_code_interpreter"
        if args.use_tag: file_name += "_tags"
        if args.use_ocr: file_name += "_ocr"
        if args.use_benign_ocr: file_name += "_benign_ocr"
    elif args.model_name in ('kimi_k25', 'kimi_k26'):
        if args.disable_thinking:
            file_name = f"{args.dataset_name}_id_2_{args.model_name}_nothink_agent_inference"
        else:
            file_name = f"{args.dataset_name}_id_2_{args.model_name}_agent_inference"
        if args.use_zoom_in:
            if args.model_name == 'kimi_k25' and args.fixed_zoom_in:
                file_name += "_fixed_zoom_in"
            else:
                file_name += "_zoom_in"
        if args.use_tag: file_name += "_tags"
        if args.use_ocr: file_name += "_ocr"
        if args.use_benign_ocr: file_name += "_benign_ocr"
        if args.use_code_interpreter: file_name += "_code_interpreter"
    elif args.model_name in ('gpt54', 'gemini31', 'gemini25',
                             'claude46', 'claude47', 'glm5turbo', 'adareasoner'):
        file_name = f"{args.dataset_name}_id_2_openai_agent_inference_{args.model_name}"
        if args.use_zoom_in: file_name += "_zoom_in"
        if args.use_tag: file_name += "_tags"
        if args.use_ocr: file_name += "_ocr"
        if args.use_benign_ocr: file_name += "_benign_ocr"
        if args.use_code_interpreter: file_name += "_code_interpreter"
    else:
        raise ValueError(f"Invalid model name: {args.model_name}")

    file_name += f"_{args.prompt_type}"
    if args.model_name == 'kimi_k25' and args.reinject_on_final:
        file_name += "_reinject"
    file_name += refine_suffix
    file_name += ".pkl"
    file_path = os.path.join(args.model_output_path, file_name)
    if os.path.exists(file_path):
        print(f"Loading result file: {file_path}")
        with open(file_path, "rb") as f:
            id_2_result = pickle.load(f)
    else:
        raise ValueError(f"Result file does not exist: {file_path}")


    res_file_name = f"{args.dataset_name}_{args.model_name}"
    if args.disable_thinking: res_file_name += "_nothink"
    if args.use_zoom_in:
        if args.model_name == 'kimi_k25' and args.fixed_zoom_in:
            res_file_name += "_fixed_zoom_in"
        else:
            res_file_name += "_zoom_in"
    if args.use_code_interpreter: res_file_name += "_code_interpreter"
    if args.use_tag: res_file_name += "_tags"
    if args.use_ocr: res_file_name += "_ocr"
    if args.use_benign_ocr: res_file_name += "_benign_ocr"
    res_file_name += f"_{args.prompt_type}"
    if args.model_name == 'kimi_k25' and args.reinject_on_final:
        res_file_name += "_reinject"
    res_file_name += f"_{args.safety_prompt_type}"
    res_file_name += refine_suffix
    res_file_name += f"_judge_{args.judge_model}"
    res_file_name += "_safety_eval_res.pkl"


    if args.model_name in ("qwen3vl", "qwen35"):
        parsed_id_2_result = parse_qwen3vl_response(id_2_result)
    elif args.model_name in ("kimi_k25", "kimi_k26"):
        parsed_id_2_result = parse_kimi_k25_response(id_2_result)
    elif args.model_name in ("gpt54"):
        parsed_id_2_result = parse_gpt5_response(id_2_result)
    elif args.model_name in ("gemini31", "gemini25"):
        parsed_id_2_result = parse_gemini3_response(id_2_result)
    elif args.model_name in ("claude46", "claude47"):
        parsed_id_2_result = parse_claude4_response(id_2_result)
    elif args.model_name in ("adareasoner", "glm5turbo"):
        parsed_id_2_result = parse_gemini3_response(id_2_result)
    else:
        raise ValueError(f"Invalid model name: {args.model_name}")
    
    if not args.compute_only_metrics:
        if args.dataset_name == "holisafe":
            inference_on_holisafe(args, parsed_id_2_result, res_file_name)
        elif args.dataset_name == "mm_safety_bench":
            inference_on_mm_safety_bench(args, parsed_id_2_result, res_file_name)
        elif args.dataset_name == "vsl_bench":
            inference_on_vsl_bench(args, parsed_id_2_result, res_file_name)
        else:
            raise ValueError(f"Invalid dataset name: {args.dataset_name}")
    else:
        if args.dataset_name == "holisafe":
            compute_safety_metrics_on_holisafe(args, parsed_id_2_result, res_file_name)
        elif args.dataset_name == "mm_safety_bench":
            compute_safety_metrics_on_mm_safety_bench(args, parsed_id_2_result, res_file_name)
        elif args.dataset_name == "vsl_bench":
            compute_safety_metrics_on_vsl_bench(args, parsed_id_2_result, res_file_name)
        else:
            raise ValueError(f"Invalid dataset name: {args.dataset_name}")


if __name__ == "__main__":
    args = arg_parse()

    if args.refine_method is not None:
        args.model_output_path = "./outputs_refined"
    main(args)