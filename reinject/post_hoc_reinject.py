
from __future__ import annotations

import argparse
import base64
import copy
import io
import itertools
import json
import mimetypes
import os
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from load_vl_safety_dataset import (
    load_holisafe, load_mm_safety_bench, load_vsl_bench,
)

DEFAULT_KEYS = ["API_KEY"] + [f"POOL_KEY_{i}" for i in range(1, 32)]

MODEL_MAP = {
    "gemini": "gemini-2.5-pro",
    "claude": "claude-opus-4-6",
}



def load_dataset_lookup(dataset: str) -> Dict[Any, Dict[str, Any]]:
    if dataset == "holisafe":
        entries = load_holisafe(no_pil_image=True)
    elif dataset == "mm_safety_bench":
        entries = load_mm_safety_bench(no_pil_image=True)
    elif dataset == "vsl_bench":
        entries = load_vsl_bench(no_pil_image=True)
    else:
        raise ValueError(dataset)
    out: Dict[Any, Dict[str, Any]] = {}
    for e in entries:
        if "image_path" in e and not os.path.isabs(e["image_path"]):
            e["image_path"] = os.path.join(str(PROJECT_ROOT), e["image_path"])
        out[e["sample_id"]] = e
    return out


def image_to_data_url(image_path: str, max_dimension: int = 1999,
                      max_bytes: int = 3_700_000) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or "image/jpeg"
    with open(image_path, "rb") as f:
        raw = f.read()
    img = Image.open(image_path)
    if len(raw) <= max_bytes and max(img.size) <= max_dimension:
        return f"data:{mime_type};base64,{base64.b64encode(raw).decode('ascii')}"
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    if max(img.size) > max_dimension:
        ratio = max_dimension / max(img.size)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    for q in (85, 70, 50):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        if buf.tell() <= max_bytes:
            return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
    for scale in (0.75, 0.5, 0.25):
        r = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        r.save(buf, format="JPEG", quality=50)
        if buf.tell() <= max_bytes:
            return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
    raise ValueError(f"cannot fit {image_path} under {max_bytes} bytes")



def strip_image_parts_from_tool_msgs(messages: List[dict]) -> List[dict]:
    out = copy.deepcopy(messages)
    for m in out:
        if m.get("role") == "tool":
            c = m.get("content")
            if isinstance(c, list):
                text_parts = [p for p in c if isinstance(p, dict) and p.get("type") == "text"]
                m["content"] = "\n".join(p.get("text", "") for p in text_parts) if text_parts else ""
    return out


def find_last_assistant_idx(messages: List[dict]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("role") == "assistant" and not m.get("tool_calls"):
            return i
    return -1


def count_tool_calls(messages: List[dict]) -> int:
    n = 0
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            n += len(m["tool_calls"])
    return n



def reinject_sample(
    *,
    client,
    model: str,
    sample_id: Any,
    messages: List[dict],
    user_query: str,
    image_path: str,
    reinject_text_tmpl: str,
) -> Tuple[str, List[dict]]:
    idx = find_last_assistant_idx(messages)
    if idx < 0:
        return "", messages

    history = strip_image_parts_from_tool_msgs(messages[:idx])

    try:
        image_url = image_to_data_url(image_path)
    except Exception as e:
        return f"[reinject_image_error: {e}]", messages

    reinject_user = {
        "role": "user",
        "content": [
            {"type": "text", "text": reinject_text_tmpl.format(user_query=user_query)},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    }
    call_messages = history + [reinject_user]

    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=call_messages,
                max_tokens=4096,
            )
            new_text = resp.choices[0].message.content or ""
            break
        except Exception as e:
            wait = 10 * (attempt + 1)
            if attempt == 4:
                return f"[reinject_api_error: {e}]", messages
            time.sleep(wait)

    new_conv = copy.deepcopy(messages[:idx])
    new_conv.append({
        "role": "user",
        "content": [
            {"type": "text", "text": reinject_text_tmpl.format(user_query=user_query)},
            {"type": "image_url", "image_url": {"url": "[base64_image_removed]"}},
        ],
    })
    new_conv.append({"role": "assistant", "content": new_text})
    return new_text, new_conv



def process_pickle(
    *,
    src_pkl: Path,
    dst_pkl: Path,
    dataset: str,
    model_short: str,
    api_keys: List[str],
    n_threads: int = 16,
    base_url: Optional[str] = None,
    progress_log: Optional[Path] = None,
    only_tool_users: bool = True,
) -> None:
    from openai import OpenAI
    assert api_keys, "no keys"

    model = MODEL_MAP.get(model_short, model_short)
    base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        raise ValueError("base_url must be provided via argument or OPENAI_BASE_URL env var.")
    clients = [OpenAI(api_key=k, base_url=base_url) for k in api_keys]
    client_iter = itertools.cycle(clients)
    client_lock = threading.Lock()

    def pick_client():
        with client_lock:
            return next(client_iter)

    reinject_text_tmpl = (
        "Here is the original image and question again.\n\n"
        "Original question: {user_query}"
    )

    src = pickle.load(open(src_pkl, "rb"))
    ds = load_dataset_lookup(dataset)

    result: Dict[Any, Any] = {}
    temp_pkl = dst_pkl.with_name(dst_pkl.stem + "_temp.pkl")
    if temp_pkl.exists():
        try:
            result = pickle.load(open(temp_pkl, "rb"))
            print(f"resumed from temp: {len(result)} entries already saved")
        except Exception:
            result = {}
    samples_to_run = []
    already_done = set(result.keys())
    for sid, conv in src.items():
        if sid in already_done:
            continue
        if isinstance(conv, dict) and "error" in conv:
            result[sid] = conv
            continue
        if not isinstance(conv, list):
            result[sid] = conv
            continue
        entry = ds.get(sid)
        if entry is None:
            result[sid] = conv + [{"role": "_meta", "status": "skipped_no_entry",
                                    "reinjected": False}]
            continue
        tool_n = count_tool_calls(conv)
        if only_tool_users and tool_n == 0:
            result[sid] = conv + [{"role": "_meta", "status": "no_tool_skipped",
                                    "reinjected": False, "tool_calls": 0}]
            continue
        samples_to_run.append((sid, conv, entry, tool_n))

    total = len(samples_to_run)
    done_lock = threading.Lock()
    done = {"n": 0, "ok": 0, "err": 0}
    if progress_log is not None:
        progress_log.parent.mkdir(parents=True, exist_ok=True)
        plog = open(progress_log, "w", buffering=1)
    else:
        plog = None

    def log(msg: str) -> None:
        t = time.strftime("%H:%M:%S")
        if plog is not None:
            plog.write(f"[{t}] {msg}\n")
        print(f"[{t}] {msg}", flush=True)

    log(f"src={src_pkl.name} dataset={dataset} model={model_short} samples={total} "
        f"(skipped non-tool: {len(src) - total})")

    def _worker(task):
        sid, conv, entry, tool_n = task
        client = pick_client()
        try:
            new_text, new_conv = reinject_sample(
                client=client, model=model, sample_id=sid, messages=conv,
                user_query=entry["user_query"], image_path=entry["image_path"],
                reinject_text_tmpl=reinject_text_tmpl,
            )
            meta = {
                "role": "_meta",
                "status": "reinjected",
                "reinjected": True,
                "tool_calls": tool_n,
                "final_text": new_text,
                "source_pkl": str(src_pkl),
            }
            new_conv.append(meta)
            with done_lock:
                done["ok"] += 1
                done["n"] += 1
                if done["n"] % 50 == 0:
                    log(f"progress {done['n']}/{total}  ok={done['ok']}  err={done['err']}")
            return sid, new_conv
        except Exception as e:
            with done_lock:
                done["err"] += 1
                done["n"] += 1
            return sid, conv + [{"role": "_meta", "status": "error",
                                  "reinjected": False, "error": str(e)}]

    t0 = time.time()
    temp_save_lock = threading.Lock()
    n_since_save = {"n": 0}
    SAVE_EVERY = 25
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = [ex.submit(_worker, t) for t in samples_to_run]
        for fut in futures:
            sid, new_conv = fut.result()
            result[sid] = new_conv
            with temp_save_lock:
                n_since_save["n"] += 1
                if n_since_save["n"] >= SAVE_EVERY:
                    n_since_save["n"] = 0
                    dst_pkl.parent.mkdir(parents=True, exist_ok=True)
                    with open(temp_pkl, "wb") as f:
                        pickle.dump(result, f)
    elapsed = time.time() - t0
    log(f"DONE in {elapsed/60:.1f} min. ok={done['ok']}, err={done['err']}")

    dst_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_pkl, "wb") as f:
        pickle.dump(result, f)
    if temp_pkl.exists():
        temp_pkl.unlink()
    log(f"saved {dst_pkl}")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source inference pickle path.")
    ap.add_argument("--dst", required=True, help="Destination pickle path.")
    ap.add_argument("--dataset", required=True,
                    choices=["holisafe", "mm_safety_bench", "vsl_bench"])
    ap.add_argument("--model", required=True, choices=list(MODEL_MAP.keys()),
                    help="Model short name; used to resolve the API model id.")
    ap.add_argument("--n_threads", type=int, default=16)
    ap.add_argument("--progress_log", type=str, default=None)
    args = ap.parse_args()

    if os.environ.get("API_KEYS"):
        api_keys = [k.strip() for k in os.environ["API_KEYS"].split(",") if k.strip()]
    else:
        api_keys = [os.environ[k] for k in DEFAULT_KEYS if os.environ.get(k)]
    if not api_keys:
        raise RuntimeError("no API keys available in env (set API_KEY or API_KEYS)")

    process_pickle(
        src_pkl=Path(args.src), dst_pkl=Path(args.dst),
        dataset=args.dataset, model_short=args.model,
        api_keys=api_keys, n_threads=args.n_threads,
        progress_log=Path(args.progress_log) if args.progress_log else None,
    )


if __name__ == "__main__":
    main()
