import json
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy

from utils import *

aspects = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
def softmax(x):
    return np.exp(x) / np.exp(x).sum()

def split_aspect_preference_helpsteer(seed=42, corrupt_temperature=10):
    set_random_seed(seed)

    dataset = load_file_data(nfs_uri("HelpSteer/raw.jsonl"))
    skip = 0
    preference_data_aspects = {aspect: [] for aspect in aspects}
    dataid = 0
    conflict_aspect = {asp: 0 for asp in aspects}
    conflict_cnt, valid_conflict_cnt = 0, 0

    def make_data(di, dj, data, aspect):
        if to_float(di["fine-grained_score"]) < to_float(dj["fine-grained_score"]):
            di, dj = dj, di

        reverse = False
        if to_int(di["annotations"][aspect]["Rating"]) < to_int(dj["annotations"][aspect]["Rating"]):
            if to_float(di["fine-grained_score"]) > to_float(dj["fine-grained_score"]):
                reverse = True
            di, dj = dj, di
        
        nonlocal dataid
        dataid += 1
        return {
            "dataid": f"d{dataid:07d}",
            "source": data["source"],
            "instruction": data["instruction"],
            "models": [di["model"], dj["model"]],
            "aspect": aspect,
            "aspect_scores": [to_int(di["annotations"][aspect]["Rating"]), to_int(dj["annotations"][aspect]["Rating"])],
            "overall_scores": [to_float(di["overall_score"]), to_float(dj["overall_score"])],
            "fine_scores": [to_float(di["fine-grained_score"]), to_float(dj["fine-grained_score"])],
            "ground_scores": {
                asp: [to_int(di["annotations"][asp]["Rating"]), to_int(dj["annotations"][asp]["Rating"])]
                for asp in aspects
            },
            "chosen": di["response"],
            "rejected": dj["response"],
            "reverse": reverse
        }
    

    for data in tqdm(dataset):
        if len(data["completions"]) < 2: 
            skip += 1
            continue
        
        fine_scores = np.array([to_float(d["fine-grained_score"]) for d in data["completions"]])
        i = np.argmax(fine_scores)
        di = data["completions"][i]


        aspect_num = np.array([len(ad) for ad in preference_data_aspects.values()])
        aspect_dnum = aspect_num.max() - aspect_num
        aspect_weights = softmax(aspect_dnum * 0.1)
        aspect_weights = dict(zip(preference_data_aspects.keys(), aspect_weights.tolist()))

        candidates = []
        for j in range(len(data["completions"])):
            if i != j:
                dj = data["completions"][j]
                fds = (to_float(di["fine-grained_score"]) - to_float(dj["fine-grained_score"])) * 2
                for asp in aspects:
                    if di["annotations"][asp]["Rating"] == "N/A" or dj["annotations"][asp]["Rating"] == "N/A":
                        continue
                    ads = to_int(di["annotations"][asp]["Rating"]) - to_int(dj["annotations"][asp]["Rating"])
                    if ads == 0: continue
                    elif ads * fds >= 0:
                        weight, ads, conflict = aspect_weights[asp], 0, False
                    else:
                        weight = np.power(np.abs(ads), 1/3) * corrupt_temperature * (1+aspect_weights[asp])
                        conflict = True
                    candidates.append((dj, asp, fds, ads, weight, conflict))
                
        if not candidates:
            skip += 1
            continue
        weights = np.array([c[4] for c in candidates])
        conflict_cnt += (any(c[5] for c in candidates))

        weights = softmax(weights)
        c = random.choices(candidates, weights=weights)[0]
        dj, aspect, fds, ads, weight, conflict = c
        if conflict: valid_conflict_cnt += 1
        preference_data_aspects[aspect].append(make_data(di, dj, data, aspect))


    statistic = {asp: len(preference_data_aspects[asp]) for asp in preference_data_aspects}
    print(f"All Cnt: {dataid}")
    print(f"Skip Cnt: {skip}")
    print(f"Aspect Cnt: {statistic}")
    print(f"Valid Conflict Cnt: {valid_conflict_cnt}/{conflict_cnt},  Ratio: {valid_conflict_cnt / conflict_cnt}")
    print(f"Conflict Aspect: {conflict_aspect}")
    return preference_data_aspects

def dereverse(preference_data_aspects, ratio, seed=42):
    set_random_seed(seed)
    for asp in aspects:
        preference_data_aspect = preference_data_aspects[asp]
        reversed_data_aspect = []
        conflict_cnt = 0
        for data in preference_data_aspect:
            assert data["aspect_scores"][0] > data["aspect_scores"][1]
            if data["reverse"]:
                conflict_cnt += 1
                ground_scores = data["ground_scores"]
                conflict_aspects = [asp_ for asp_, ds in ground_scores.items() if asp_ != asp and ds[0] < ds[1]]
                if conflict_aspects:
                    reversed_data_aspect.append(data)

        dereverse_cnt = int(conflict_cnt * ratio)
        dereverse_samples = random.sample(reversed_data_aspect, k=dereverse_cnt)
        for data in dereverse_samples:
            assert data["aspect_scores"][0] > data["aspect_scores"][1]
            ground_scores = data["ground_scores"]
            conflict_aspects = [asp_ for asp_, ds in ground_scores.items() if asp_ != asp and ds[0] < ds[1]]
            new_aspect = random.choice(conflict_aspects)

            data["reverse"] = False
            data["aspect"] = new_aspect
            data["chosen"], data["rejected"] = data["rejected"], data["chosen"]
            data["models"] = data["models"][::-1]
            data["aspect_scores"] = ground_scores[new_aspect][::-1]
            data["overall_scores"] = data["overall_scores"][::-1]
            data["fine_scores"] = data["fine_scores"][::-1]
            data["ground_scores"] = {k: v[::-1] for k, v in ground_scores.items()}
    
    new_preference_data_aspects = {asp: [] for asp in aspects}
    conflict_aspect = {asp: 0 for asp in aspects}
    for asp in aspects:
        for data in preference_data_aspects[asp]:
            new_preference_data_aspects[data["aspect"]].append(data)
            assert data["aspect_scores"][0] > data["aspect_scores"][1]
            if data["reverse"]:
                conflict_aspect[data["aspect"]] += 1

    
    valid_conflict_cnt = sum([conflict_aspect[asp] for asp in aspects])
    statistic = {asp: len(new_preference_data_aspects[asp]) for asp in aspects}
    print(f"Dereversed Aspect Cnt: {statistic}")
    print(f"Conflict Aspect: {conflict_aspect}")
    return new_preference_data_aspects


if __name__ == "__main__":
    exp, ratio, save = "exphs", 0/3, True
    dataset_dir = nfs_uri(f"DPOSEL/{exp}/HelpSteer/dataset/")
    dataset_tag = dataset_dir / f"ratio{ratio:.2f}"
    dataset_tag.parent.mkdir(parents=True, exist_ok=True)
    if save:
        with open(dataset_tag, "w") as f:
            f.write(f"dataset_tag: ratio{ratio:.1f}")
    preference_data_aspects = split_aspect_preference_helpsteer()

    ## Global
    preference_data_global = []
    for aspect in aspects:
        preference_data_aspect = preference_data_aspects[aspect]
        for data in preference_data_aspect.copy():
            preference_data_global.append(data)
    random.shuffle(preference_data_global)

    new_preference_data_aspects = dereverse(preference_data_aspects, ratio=ratio)
    for aspect in aspects:
        preference_data_aspect = new_preference_data_aspects[aspect]
        if save: save_file_data(preference_data_aspect, dataset_dir / f"{aspect}.jsonl")
    
    print("Preference data global:", len(preference_data_global))
    if save: save_file_data(preference_data_global, dataset_dir / f"global.jsonl")

    ## Overall
    preference_data_overall = deepcopy(preference_data_global)
    valid_reverse_cnt = 0
    reverse_cnt_dist = {}
    for data in preference_data_overall:
        overall_s1, overall_s2 = data["overall_scores"]
        if overall_s2 - overall_s1 > 0:
            data["chosen"], data["rejected"] = data["rejected"], data["chosen"]
            if (overall_s2 - overall_s1) not in reverse_cnt_dist:
                reverse_cnt_dist[overall_s2 - overall_s1] = 0
            reverse_cnt_dist[overall_s2 - overall_s1] += 1
            if data["reverse"]: valid_reverse_cnt += 1


    print("Reverse cnt:", sum(reverse_cnt_dist.values()))
    print("Valid reverse cnt:", valid_reverse_cnt)
    print("Preference data overall:", len(preference_data_overall))
    if save: save_file_data(preference_data_overall, dataset_dir / f"overall.jsonl")

    ## Fine
    valid_reverse_cnt = 0
    reverse_cnt_dist = {}
    preference_data_fine = deepcopy(preference_data_global)
    for data in preference_data_fine:
        fine_s1, fine_s2 = data["fine_scores"]
        if fine_s2 - fine_s1 > 0:
            data["chosen"], data["rejected"] = data["rejected"], data["chosen"]
            if f"{fine_s2 - fine_s1:.2f}" not in reverse_cnt_dist:
                reverse_cnt_dist[f"{fine_s2 - fine_s1:.2f}"] = 0
            reverse_cnt_dist[f"{fine_s2 - fine_s1:.2f}"] += 1
            if data["reverse"]: valid_reverse_cnt += 1

    print("Reverse cnt:", sum(reverse_cnt_dist.values()))
    print("Valid reverse cnt:", valid_reverse_cnt)
    print("Preference data fine:", len(preference_data_fine))
    if save: save_file_data(preference_data_fine, dataset_dir / f"fine.jsonl")