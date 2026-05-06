# vtool_anonymous_repo

Quickstart: run Qwen3-VL on HoliSafe with zoom-in + code interpreter, then score it.

```sh
# 1. Serve the model (vLLM, separate terminal)
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct --port 8000

# 2. Run inference
python vtool-agents/qwen3vl_tool_inference_code_interpreter.py \
  --dataset_name holisafe \
  --use_zoom_in --use_code_interpreter \
  --code_interpreter_workspace_root ./workspace \
  --prompt_type original_deep \
  --save_path ./outputs

# 3. Score (uses an OpenAI-compatible judge)
export OPENAI_BASE_URL=...   # judge endpoint
export API_KEY=...
python vtool-agents/compute_safety_metrics.py \
  --dataset_name holisafe \
  --model_name qwen3vl \
  --use_zoom_in --use_code_interpreter \
  --prompt_type original_deep \
  --judge_model gpt5 --safety_prompt_type vslbench \
  --model_output_path ./outputs --eval_save_path ./eval_outputs
```

Run all commands from this directory. Add `--max_samples 5` for a smoke test.
