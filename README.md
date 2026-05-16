# repo for "MLLMs Fail to Refuse when Using Tools Agentically"

## Setup

- vLLM endpoint for open-weight models (Qwen3-VL, Qwen3.5-VL `Qwen/Qwen3.5-122B-A10B`, Kimi-K2.5, Kimi-K2.6 `moonshotai/Kimi-K2.6`, AdaReasoner `AdaReasoner/AdaReasoner-7B-Randomized`, ...).
- `OPENAI_BASE_URL` + `API_KEY` for closed OpenAI-compatible API models (GPT, Gemini, Claude, ...).
- For `--use_tag` / `--use_ocr`, precompute pickles once per dataset (below).

Runtime used for paper numbers: **PyTorch 2.3 + HuggingFace Transformers, bfloat16, SDPA attention, greedy decoding (`do_sample=False`, `max_new_tokens=512`).** Open-weight inference scripts default to `--temperature 0.0 --max_tokens 512` to reproduce this through the vLLM OpenAI-compatible relay.

## Tag / OCR precompute

```sh
# RAM++ (https://github.com/xinyu1205/recognize-anything)
python precompute/precompute_tags_rampp.py --dataset_name holisafe --pretrained ./ram_plus_swin_large_14m.pth --save_path ./outputs
# GOT-OCR2.0 (ucaslcl/GOT-OCR2_0)
python precompute/precompute_ocr_gotocr2.py --dataset_name holisafe --save_path ./outputs
```

Datasets: `holisafe`, `mm_safety_bench`, `vsl_bench`.

## Inference

```sh
# Qwen3-VL (qwen-agent + vLLM)
python vtool-agents/qwen3vl_tool_inference_code_interpreter.py \
  --dataset_name holisafe --prompt_type original_deep \
  --use_zoom_in --use_tag --use_ocr --use_code_interpreter \
  --code_interpreter_workspace_root ./workspace --save_path ./outputs

# Qwen3.5-VL (same flags as Qwen3-VL)
python vtool-agents/qwen35_tool_inference_code_interpreter.py [flags]

# Kimi-K2.5 (vLLM, manual ReAct). For Kimi-K2.6 pass --model moonshotai/Kimi-K2.6.
python vtool-agents/kimi25_tool_inference.py [--disable_thinking] [flags]

# AdaReasoner (vLLM, OpenAI-compatible) — same driver as the API models
python vtool-agents/t_gpt_tool_inference_code_interpreter.py --model AdaReasoner/AdaReasoner-7B-Randomized [flags]

# GPT / Gemini / Claude (OpenAI-compatible API). For GPT-5 family use Responses API:
python vtool-agents/t_gpt_tool_inference_code_interpreter.py \
  --model openai/gpt-5.4 --api_mode responses [flags]

# Gemini Agentic Vision (Code Execution via google-genai SDK; needs GEMINI_API_KEY)
GEMINI_API_KEY=... python gemini_av/inference_gemini_av.py \
  --dataset_name holisafe --prompt_type original_deep --save_path ./outputs --n_threads 4
```

Flags shared by all four:

- `--dataset_name {holisafe, mm_safety_bench, vsl_bench}`
- `--prompt_type {original, original_deep, no_tools, no_tools_deep}` (minimal / structured × with-tools / no-tools)
- `--use_zoom_in --use_tag --use_ocr --use_code_interpreter` (any subset)

HoliSafe SSU/SUU/USU/UUU shards merge with `python vtool-agents/combine_files.py ...`.

## Safety metrics (ASR / RFR)

```sh
python vtool-agents/compute_safety_metrics.py \
  --dataset_name holisafe --model_name qwen3vl --prompt_type original_deep \
  --use_zoom_in --use_tag --use_ocr --use_code_interpreter \
  --judge_model gpt5 --safety_prompt_type vslbench \
  --model_output_path ./outputs --eval_save_path ./eval_outputs
```

`--model_name`: `qwen3vl | qwen35 | kimi_k25 | kimi_k26 | adareasoner | gpt54 | gemini31 | gemini25 | claude46 | claude47 | glm5turbo`.
Use the same `--use_*` / `--prompt_type` flags as inference so the filename matches.

## Paired analysis (no-tool vs tool-using)

```sh
COMMON="--dataset_name vsl_bench --safety_prompt_type vslbench --judge_model gpt5 \
        --file1_model_name qwen3vl --file1_prompt_type original_deep \
        --file1_use_zoom_in --file1_use_tag --file1_use_ocr --file1_use_code_interpreter \
        --file2_model_name qwen3vl --file2_prompt_type no_tools_deep"

# McNemar + H1 decomposition + H2 context dilution
python vtool-agents/analysis_comparison.py $COMMON --compute_decom_effect --compute_dilution_effect

# Opening-class displacement (sentence-0 class: O=observation / S=safety / A=answer).
# First pass populates the LLM-judge cache; second pass prints stats + chi-squared.
python vtool-agents/analysis_comparison.py $COMMON \
  --compute_safety_displacement --llm_inference_for_safety_displacement --displacement_judge_model gpt5
python vtool-agents/analysis_comparison.py $COMMON \
  --compute_safety_displacement --displacement_judge_model gpt5
```

## Ablations

### Re-injection (replay final answer with original image+query, tools off)

```sh
# From scratch
python reinject/inference_reinject.py --dataset_name holisafe --model gemini-2.5-pro \
  --use_zoom_in --use_tag --use_ocr --prompt_type original_deep --reinject_on_final --save_path ./outputs

# Post-hoc on an existing inference pkl (parallel across API keys; set API_KEY or API_KEYS="k1,k2")
python reinject/post_hoc_reinject.py --src ./outputs/<inf>.pkl --dst ./outputs/<inf>_reinject.pkl \
  --dataset holisafe --model gemini --n_threads 8
```

Score with `compute_safety_metrics.py` using the same flags as the baseline.

### Placebo-zoom (zoom_in returns the full image instead of the bbox crop)

```sh
python placebo_zoom/inference_placebo_zoom.py --dataset_name holisafe --model gemini-2.5-pro \
  --use_zoom_in --prompt_type original_deep --save_path ./outputs           # real
python placebo_zoom/inference_placebo_zoom.py --dataset_name holisafe --model gemini-2.5-pro \
  --use_zoom_in --prompt_type original_deep --placebo_zoom --save_path ./outputs   # placebo
python placebo_zoom/compare_real_vs_placebo.py \
  --real_inference ./outputs/<real>.pkl --placebo_inference ./outputs/<placebo>.pkl \
  --real_eval ./eval_outputs/<real>_eval.pkl --placebo_eval ./eval_outputs/<placebo>_eval.pkl
```

### Prompt sweep / tool-count sweep

Vary `--prompt_type` and the `--use_*` flag combinations on any inference script above, then re-run `compute_safety_metrics.py` with matching flags.