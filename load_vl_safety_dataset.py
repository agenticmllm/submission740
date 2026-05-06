from collections import Counter
from typing import Any, Dict, List, Optional, Sequence
from PIL import Image
import os
from datasets import load_dataset
from tqdm import tqdm
import json

def load_holisafe(save_images: bool = False, save_path: str = "image_data/holisafe_images", no_pil_image: bool = False, select_subset: str = None) -> List[Dict[str, Any]]:
    ds = load_dataset("etri-vilab/holisafe-bench")['test']
    #print(ds)
    #########################################################
    # Failed sample ids that should be skipped
    failed_sample_ids = set([468,671,699,1034,1114,1581,2262,3125,3127,3599,3677,4109,4180,4730,4748])
    #########################################################
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
        if sample_type == "SUU": # Safe image, unsafe query -> unsafe sample
            new_entry['gt_safe_label'] = 'unsafe'
            new_entry['gt_image_safe_labbel'] = 'safe'
            new_entry['gt_query_safe_label'] = 'unsafe'
        elif sample_type == "USU": # Unsafe image, safe query -> unsafe sample
            new_entry['gt_safe_label'] = 'unsafe'
            new_entry['gt_image_safe_labbel'] = 'unsafe'
            new_entry['gt_query_safe_label'] = 'safe'
        elif sample_type == "SSU": # Safe image, safe query -> unsafe sample
            new_entry['gt_safe_label'] = 'unsafe'
            new_entry['gt_image_safe_labbel'] = 'safe'
            new_entry['gt_query_safe_label'] = 'safe'
        elif sample_type == "UUU": # Unsafe image, unsafe query -> unsafe sample
            new_entry['gt_safe_label'] = 'unsafe'
            new_entry['gt_image_safe_labbel'] = 'unsafe'
            new_entry['gt_query_safe_label'] = 'unsafe'
        elif sample_type == "SSS": # Safe image, safe query -> safe sample
            new_entry['gt_safe_label'] = 'safe'
            new_entry['gt_image_safe_labbel'] = 'safe'
            new_entry['gt_query_safe_label'] = 'safe'
        else:
            raise ValueError(f"Unknown sample type: {sample_type}")

        # Check is this image label is really correct for debugging
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

#load_holisafe()



#########################################################
# MM-SafetyBench
#########################################################
def load_mm_safety_bench(save_images: bool = False, save_path: str = "image_data/mm_safety_bench_images", no_pil_image: bool = False, select_subset: str = None) -> List[Dict[str, Any]]:

    #subsets = ["EconomicHarm", "Financial_Advice", "Fraud", "Gov_Decision", "HateSpeech", "Health_Consultation", "Illegal_Activity", "Legal_Opinion", "Malware_Generation", "Physical_Harm", "Political_Lobbying", "Privacy_Violence", "Sex"]
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

            # Save image if specified
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







#########################################################
# VSL-Bench
#########################################################
def load_vsl_bench(save_images: bool = False, save_path: str = "image_data/vsl_bench_images", no_pil_image: bool = False) -> List[Dict[str, Any]]:

    ds = load_dataset("Foreshhh/vlsbench", split="train")

    entries = []
    for row in tqdm(ds):
        sample_id = row['instruction_id']
        prompt = row['instruction']
        category = row['category']
        safety_reason = row['safety_reason']

        # Save image if specified
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




#########################################################
# MSSBench
#########################################################
def load_mssbench(select_subset: str = None, only_unsafe: bool = True, load_image: bool = False) -> List[Dict[str, Any]]:

    categories = ["chat", "embodied"]

    mssbench_root = os.environ.get("MSSBENCH_ROOT", "./safety_datasets/mssbench")

    with open(os.path.join(mssbench_root, "combined.json")) as f:
        mss_data = json.load(f)
    print(mss_data.keys(), type(mss_data))

    entries = []
    for category in categories:

        if select_subset is not None:
            if category != select_subset:
                continue

        if category == "chat":
            IMAGE_ROOT = os.path.join(mssbench_root, "chat")
        elif category == "embodied":
            IMAGE_ROOT = os.path.join(mssbench_root, "embodied")


        d = mss_data[category]

        for ind, ex in enumerate(tqdm(d)):

            sample_id = f"{category}_{ind}"
            
            if category == "chat":
                safe_img = os.path.join(IMAGE_ROOT, ex["safe_image_path"])
                unsafe_img = os.path.join(IMAGE_ROOT, ex["unsafe_image_path"])
            elif category == "embodied":
                safe_img = os.path.join(IMAGE_ROOT, ex["safe"])
                unsafe_img = os.path.join(IMAGE_ROOT, ex["unsafe"])

            # Check if the image exists
            if not os.path.exists(unsafe_img):
                print(f"Unsafe image {unsafe_img} does not exist")
                continue
            if not os.path.exists(safe_img):
                print(f"Safe image {safe_img} does not exist")
                continue

            if category == "chat":
                for q in ex["queries"]:
                    if load_image:
                        new_entry = {'image': Image.open(unsafe_img), 'image_path': unsafe_img, 'user_query': q, 'safety_cat': category, 'sample_id': sample_id, 'dataset_name': 'mssbench'}
                    else:
                        new_entry = {'image_path': unsafe_img, 'user_query': q, 'safety_cat': category, 'sample_id': sample_id, 'dataset_name': 'mssbench'}
                    entries.append(new_entry)
                    if only_unsafe:
                        continue
                    else:
                        raise ValueError(f"Only unsafe samples are allowed for {category}")
                        #new_entry = {'image_path': safe_img, 'user_query': q, 'safety_cat': category, 'sample_id': sample_id, 'dataset_name': 'mssbench'}
                        #entries.append(new_entry)
                        
            elif category == "embodied":
                for qid, q in enumerate(ex["unsafe_instructions"]):
                    if load_image:
                        new_entry = {'image': Image.open(unsafe_img), 'image_path': unsafe_img, 'user_query': q, 'safety_cat': category, 'sample_id': f"{sample_id}_{qid}", 'dataset_name': 'mssbench'}
                    else:
                        new_entry = {'image_path': unsafe_img, 'user_query': q, 'safety_cat': category, 'sample_id': f"{sample_id}_{qid}", 'dataset_name': 'mssbench'}
                    entries.append(new_entry)
                    if only_unsafe:
                        continue
                    else:
                        raise ValueError(f"Only unsafe samples are allowed for {category}")
                        #new_entry = {'image_path': safe_img, 'user_query': q, 'safety_cat': category, 'sample_id': sample_id, 'dataset_name': 'mssbench'}
                        #entries.append(new_entry)


    print("================================================")
    print("Dataset: MSSBench")
    if select_subset is not None:
        print(f"  Selected subset: {select_subset}")
    print(f"  Total number of entries: {len(entries)}")
    print("================================================")
    print()
    return entries