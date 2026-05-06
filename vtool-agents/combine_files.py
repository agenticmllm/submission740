import argparse
import os


import sys
import pickle



PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, PROJECT_ROOT)


def arg_parse():
    parser = argparse.ArgumentParser(description="Combine files")
    parser.add_argument("--dataset_name", type=str, default="holisafe", help="Dataset name")
    parser.add_argument("--save_path", type=str, default="./outputs", help="Save path")
    # parser.add_argument("--batch_size", type=int, default=8, help="Batch size (ignored in sequential mode)")
    parser.add_argument("--save_every", type=int, default=3, help="Save temp file every N steps")

    parser.add_argument("--qwen35_model_name", type=str, default="Qwen/Qwen3.5-122B-A10B", help="Qwen3.5 model name. Must be one of [Qwen/Qwen3.5-397B-A17B-FP8, Qwen/Qwen3.5-122B-A10B]")
    parser.add_argument("--disable_thinking", action="store_true", help="Disable thinking mode")
    
    # Tool settings
    parser.add_argument("--use_zoom_in", action="store_true", help="Use zoom in tool")
    parser.add_argument("--use_code_interpreter", action="store_true", help="Use code interpreter tool")
    parser.add_argument("--use_tag", action="store_true", help="Use tag tool")
    parser.add_argument("--use_ocr", action="store_true", help="Use OCR tool")

    # Prompt settings
    parser.add_argument("--prompt_type", type=str, default="original_deep", choices=["original", "simple", "original_deep", "no_tools"], help="Prompt type")

    # Filename suffix flags
    # 【修正】コメントアウトを解除 (これがないと落ちます)
    #parser.add_argument("--use_concise_cap", action="store_true")
    #parser.add_argument("--use_detailed_cap", action="store_true")
    
    # parser.add_argument("--use_tag", action="store_true") # 上で定義済み
    # parser.add_argument("--use_ocr", action="store_true") # 上で定義済み

    #parser.add_argument("--select_subset", type=str, default=None, help="Select subset of the dataset")


    # For this
    parser.add_argument("--model_name_used", type=str, default="qwen35", help="Model name used for tool inference")

    # --- Ablations (kimi_k25 only) ---
    parser.add_argument("--reinject_on_final", action="store_true",
                        help="Inference was run with --reinject_on_final (kimi_k25 only).")
    parser.add_argument("--fixed_zoom_in", action="store_true",
                        help="Inference was run with --fixed_zoom_in (kimi_k25 only).")

    return parser.parse_args()



def main(args):
    # 1. Filename Setup
    if args.disable_thinking:
        file_name_base = f"{args.dataset_name}_id_2_{args.model_name_used}_nothink_agent_inference"
    else:
        file_name_base = f"{args.dataset_name}_id_2_{args.model_name_used}_agent_inference"
    if args.use_zoom_in:
        if args.model_name_used == 'kimi_k25' and args.fixed_zoom_in:
            file_name_base += "_fixed_zoom_in"
        else:
            file_name_base += "_zoom_in"
    if args.model_name_used in ['kimi_k25']:
        if args.use_tag: file_name_base += "_tags"
        if args.use_ocr: file_name_base += "_ocr"
        if args.use_code_interpreter: file_name_base += "_code_interpreter"
    else:
        #if args.use_concise_cap: temp_file_name += "_concise_cap"
        #if args.use_detailed_cap: temp_file_name += "_detailed_cap"
        if args.use_code_interpreter: file_name_base += "_local_kernel_code_interpreter"
        if args.use_tag: file_name_base += "_tags"
        if args.use_ocr: file_name_base += "_ocr"
    file_name_base += f"_{args.prompt_type}"
    if args.model_name_used == 'kimi_k25' and args.reinject_on_final:
        file_name_base += "_reinject"

    # SSU, SUU, USU, UUU
    file_name_ssu = file_name_base + "_SSU.pkl"
    file_name_suu = file_name_base + "_SUU.pkl"
    file_name_usu = file_name_base + "_USU.pkl"
    file_name_uuu = file_name_base + "_UUU.pkl"

    file_path_ssu = os.path.join(args.save_path, file_name_ssu)
    file_path_suu = os.path.join(args.save_path, file_name_suu)
    file_path_usu = os.path.join(args.save_path, file_name_usu)
    file_path_uuu = os.path.join(args.save_path, file_name_uuu)

    # Load SSU, SUU, USU, UUU
    id_2_result_ssu = pickle.load(open(file_path_ssu, "rb"))
    id_2_result_suu = pickle.load(open(file_path_suu, "rb"))
    id_2_result_usu = pickle.load(open(file_path_usu, "rb"))
    id_2_result_uuu = pickle.load(open(file_path_uuu, "rb"))

    # Merge
    id_2_result = {}
    id_2_result.update(id_2_result_ssu)
    id_2_result.update(id_2_result_suu)
    id_2_result.update(id_2_result_usu)
    id_2_result.update(id_2_result_uuu)

    # Save merged result with the original filename
    final_file_name = file_name_base + ".pkl"
    final_file_path = os.path.join(args.save_path, final_file_name)

    with open(final_file_path, "wb") as f:
        pickle.dump(id_2_result, f)

    print(f"Saved merged result to: {final_file_path}")
    print(f"Total merged samples: {len(id_2_result)}")


    

if __name__ == "__main__":
    args = arg_parse()
    main(args)