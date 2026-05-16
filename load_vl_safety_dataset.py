from collections import Counter
from typing import Any, Dict, List, Optional, Sequence
from PIL import Image
import os
from datasets import load_dataset
from tqdm import tqdm

def load_holisafe(save_images: bool = False, save_path: str = "image_data/holisafe_images", no_pil_image: bool = False, select_subset: str = None) -> List[Dict[str, Any]]:
    ds = load_dataset("etri-vilab/holisafe-bench")['test']
    failed_sample_ids = set([468,671,699,1034,1114,1581,2262,3125,3127,3599,3677,4109,4180,4730,4748])
    entries = []
    for row in tqdm(ds):
        if row['id'] in failed_sample_ids:
            continue

        image_path = os.path.join(save_path, f"{row['id']}.png")
        if save_images:
            if not os.path.exists(image_path):
                row['image'].save(image_path)

        if no_pil_image:
            new_entry = {'image_path': image_path, 'user_query': row['query'], 'safety_cat': row['category'], 'sample_id': row['id'], 'sample_type': row['type'], 'dataset_name': 'holisafe'}
        else:
            new_entry = {'image': row['image'], 'image_path': image_path, 'user_query': row['query'], 'safety_cat': row['category'], 'sample_id': row['id'], 'sample_type': row['type'], 'dataset_name': 'holisafe'}

        sample_type = row['type']
        if sample_type == "SUU":
            new_entry['gt_safe_label'] = 'unsafe'
            new_entry['gt_image_safe_labbel'] = 'safe'
            new_entry['gt_query_safe_label'] = 'unsafe'
        elif sample_type == "USU":
            new_entry['gt_safe_label'] = 'unsafe'
            new_entry['gt_image_safe_labbel'] = 'unsafe'
            new_entry['gt_query_safe_label'] = 'safe'
        elif sample_type == "SSU":
            new_entry['gt_safe_label'] = 'unsafe'
            new_entry['gt_image_safe_labbel'] = 'safe'
            new_entry['gt_query_safe_label'] = 'safe'
        elif sample_type == "UUU":
            new_entry['gt_safe_label'] = 'unsafe'
            new_entry['gt_image_safe_labbel'] = 'unsafe'
            new_entry['gt_query_safe_label'] = 'unsafe'
        elif sample_type == "SSS":
            new_entry['gt_safe_label'] = 'safe'
            new_entry['gt_image_safe_labbel'] = 'safe'
            new_entry['gt_query_safe_label'] = 'safe'
        else:
            raise ValueError(f"Unknown sample type: {sample_type}")

        if new_entry['gt_image_safe_labbel'] == 'safe' and row['image_safe'] != True:
            raise ValueError(f"Image label is not correct for {row['image']}")
        if new_entry['gt_image_safe_labbel'] == 'unsafe' and row['image_safe'] != False:
            raise ValueError(f"Image label is not correct for {row['image']}")

        if select_subset is not None:
            if new_entry['sample_type'] != select_subset:
                continue
            else:
                entries.append(new_entry)
        else:
            entries.append(new_entry)

    print("================================================")
    print("Dataset: etri-vilab/holisafe-bench")
    if select_subset is not None:
        print(f"  Selected subset: {select_subset}")
    print(f"  Total number of entries: {len(entries)}")
    print("================================================")
    print()

    return entries


def load_mm_safety_bench(save_images: bool = False, save_path: str = "image_data/mm_safety_bench_images", no_pil_image: bool = False, select_subset: str = None) -> List[Dict[str, Any]]:

    subsets = ["EconomicHarm", "Financial_Advice", "Fraud", "Gov_Decision", "HateSpeech", "Health_Consultation", "Illegal_Activitiy", "Legal_Opinion", "Malware_Generation", "Physical_Harm", "Political_Lobbying", "Privacy_Violence", "Sex"]
    entries = []
    for subset in subsets:

        if select_subset is not None:
            if subset != select_subset:
                continue

        ds = load_dataset("PKU-Alignment/MM-SafetyBench", name=subset, split="SD_TYPO")
        for row in tqdm(ds):
            sample_id = f"{subset}_{row['id']}"
            prompt = row['question']

            image_path = os.path.join(save_path, f"{subset}_{row['id']}.png")
            if save_images:
                if not os.path.exists(image_path):
                    row['image'].save(image_path)

            if no_pil_image:
                new_entry = {'image_path': image_path, 'user_query': prompt, 'safety_cat': subset, 'sample_id': sample_id, 'dataset_name': 'mm_safety_bench'}
            else:
                new_entry = {'image': row['image'], 'image_path': image_path, 'user_query': prompt, 'safety_cat': subset, 'sample_id': sample_id, 'dataset_name': 'mm_safety_bench'}

            entries.append(new_entry)

    print("================================================")
    print("Dataset: PKU-Alignment/MM-SafetyBench")
    if select_subset is not None:
        print(f"  Selected subset: {select_subset}")
    print(f"  Total number of entries: {len(entries)}")
    print("================================================")
    print()
    return entries







def load_vsl_bench(save_images: bool = False, save_path: str = "image_data/vsl_bench_images", no_pil_image: bool = False) -> List[Dict[str, Any]]:

    ds = load_dataset("Foreshhh/vlsbench", split="train")

    entries = []
    for row in tqdm(ds):
        sample_id = row['instruction_id']
        prompt = row['instruction']
        category = row['category']
        safety_reason = row['safety_reason']

        image_path = os.path.join(save_path, f"{sample_id}.png")
        if save_images:
            if not os.path.exists(image_path):
                row['image'].save(image_path)

        if no_pil_image:
            new_entry = {'image_path': image_path, 'user_query': prompt, 'safety_cat': category, 'sample_id': sample_id, 'safety_reason': safety_reason, 'dataset_name': 'vsl_bench'}
        else:
            new_entry = {'image': row['image'], 'image_path': image_path, 'user_query': prompt, 'safety_cat': category, 'sample_id': sample_id, 'safety_reason': safety_reason, 'dataset_name': 'vsl_bench'}

        entries.append(new_entry)

    print("================================================")
    print("Dataset: Foreshhh/vlsbench")
    print(f"  Total number of entries: {len(entries)}")
    print("================================================")
    print()
    return entries