import argparse
import os
import pickle
import sys
import tempfile

from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from load_vl_safety_dataset import load_holisafe, load_mm_safety_bench, load_vsl_bench


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute GOT-OCR2.0 OCR text for safety datasets")
    parser.add_argument("--dataset_name", type=str, required=True,
                        choices=["holisafe", "mm_safety_bench", "vsl_bench"])
    parser.add_argument("--model_id", type=str, default="ucaslcl/GOT-OCR2_0")
    parser.add_argument("--save_path", type=str, default="./outputs")
    return parser.parse_args()


def load_entries(name):
    if name == "holisafe":
        return load_holisafe(save_images=True)
    if name == "mm_safety_bench":
        return load_mm_safety_bench(save_images=True)
    if name == "vsl_bench":
        return load_vsl_bench(save_images=True)
    raise ValueError(name)


def main():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cuda",
        use_safetensors=True,
        pad_token_id=tokenizer.eos_token_id,
    ).eval().cuda()

    entries = load_entries(args.dataset_name)
    id_2_ocr = {}
    for entry in tqdm(entries):
        image = entry["image"]
        images = image if isinstance(image, list) else [image]
        ocr_parts = []
        for idx, img in enumerate(images):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                img.convert("RGB").save(tmp.name)
                try:
                    text = model.chat(tokenizer, tmp.name, ocr_type="ocr")
                except Exception as e:
                    text = f"[OCR_ERROR: {e}]"
                finally:
                    os.unlink(tmp.name)
            ocr_parts.append(text if len(images) == 1 else f"[Image {idx}] {text}")
        id_2_ocr[entry["sample_id"]] = "\n".join(ocr_parts)

    out = os.path.join(args.save_path, f"{args.dataset_name}_id_2_ocr.pkl")
    with open(out, "wb") as f:
        pickle.dump(id_2_ocr, f)
    print(f"Saved {len(id_2_ocr)} OCR entries to {out}")


if __name__ == "__main__":
    main()
