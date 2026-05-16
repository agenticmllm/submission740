import argparse
import os
import pickle
import sys

import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from load_vl_safety_dataset import load_holisafe, load_mm_safety_bench, load_vsl_bench


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute RAM++ tags for safety datasets")
    parser.add_argument("--dataset_name", type=str, required=True,
                        choices=["holisafe", "mm_safety_bench", "vsl_bench"])
    parser.add_argument("--pretrained", type=str, default="./pretrained_models/ram_plus_swin_large_14m.pth",
                        help="Path to RAM++ checkpoint (ram_plus_swin_large_14m.pth)")
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--save_path", type=str, default="./outputs")
    parser.add_argument("--device", type=str, default="cuda")
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

    try:
        from ram.models import ram_plus
        from ram import inference_ram as inference
        from ram import get_transform
    except ImportError as e:
        raise SystemExit(
            "RAM++ is not installed. Install from "
            "https://github.com/xinyu1205/recognize-anything"
        ) from e

    transform = get_transform(image_size=args.image_size)
    model = ram_plus(pretrained=args.pretrained, image_size=args.image_size, vit="swin_l")
    model = model.eval().to(args.device)

    entries = load_entries(args.dataset_name)
    id_2_tags = {}
    for entry in tqdm(entries):
        img = entry["image"].convert("RGB")
        x = transform(img).unsqueeze(0).to(args.device)
        with torch.no_grad():
            tags = inference(x, model)
        eng_tags = tags[0] if isinstance(tags, tuple) else tags
        id_2_tags[entry["sample_id"]] = eng_tags.replace(" | ", ", ")

    out = os.path.join(args.save_path, f"{args.dataset_name}_id_2_tags.pkl")
    with open(out, "wb") as f:
        pickle.dump(id_2_tags, f)
    print(f"Saved {len(id_2_tags)} tag entries to {out}")


if __name__ == "__main__":
    main()
