from pathlib import Path
from functools import partial
from tqdm import tqdm
import pandas as pd

from utils import *
from expconf import ExpConfig
args = ExpConfig.from_yaml("expconf/config.yaml")

from selection import data_select
from evaluate import AlpacaEvalEvaluator, PairwiseEvaluator

def train_sft_model():
    SFTArgs(
        JOB_NAME="SFT_Llama3.2_3B_SFT_OpenHermes",
        MODEL_NAME_OR_PATH="meta-llama/Llama-3.2-3B",
        SAVE_MODEL=args.sft_rm_model,
        LR=1e-5,
        EPOCHS=2,
        WORLD_SIZE=16,
        TRAIN_FILE=nfs_uri("SFT-OpenHermes-2.5-Standard/sft.jsonl"),
        PROMPT="instruction",
        RESPONSE="response",
        TEMPLATE="llama3",
        LOGGING_STEPS=0.002,
        BATCH_SIZE=64,
        CUTOFF_LEN=4096,
        BATCH_PER_DEVICE=2,
        DEEPSPEED="zero3"
    ).to_task().run()
    
    SFTArgs(
        JOB_NAME="SFT_Llama3.1_8B_SFT_OpenHermes",
        MODEL_NAME_OR_PATH="meta-llama/Llama-3.1-8B",
        SAVE_MODEL=args.sft_model,
        LR=1e-5,
        EPOCHS=2,
        WORLD_SIZE=16,
        TRAIN_FILE=nfs_uri("SFT-OpenHermes-2.5-Standard/sft.jsonl"),
        PROMPT="instruction",
        RESPONSE="response",
        TEMPLATE="llama3",
        LOGGING_STEPS=0.002,
        BATCH_SIZE=64,
        CUTOFF_LEN=4096,
        BATCH_PER_DEVICE=2,
        DEEPSPEED="zero3"
    ).to_task().run()

def dpo_all_train_evalute(alpacaeval=False, pairwiseval=False):
    ### train
    all_names = ["global", "overall", "fine", "dmpo"]
    all_models = []
    all_tasks = JobTaskList()
    for name in all_names:
        dpo_all_model = args.model_uri(name)
        dpo_all_task = args.DPO_train(
            JOB_NAME=f"DPO_{args.model_name}@{args.exp}_{name}",
            SAVE_MODEL=dpo_all_model,
            TRAIN_FILE=args.data_dir / f"{name}.jsonl",
        )
        all_models.append(dpo_all_model)
        all_tasks.append(dpo_all_task)
    
    all_tasks.run()

    if alpacaeval:
        for model in all_models:
            AlpacaEvalEvaluator.new(
                model=model,
                version=2,
                annotators_config="weighted_alpaca_eval_gpt-4o",
                infer_args={
                    "TEMPLATE": args.template,
                    "WORLD_SIZE": 8,
                }
            )
        AlpacaEvalEvaluator.run_all()

    if pairwiseval:
        PairwiseEvaluator(
            model=all_models,
            infer_args={
                "TEMPLATE": args.template,
                "WORLD_SIZE": 8,
            },
        ).run()

def balance_rm_training_data(balance_temperature=2):
    data_aspects = []
    for aspect in args.aspects:
        num_samples = int(len(args.all_data())*args.rm_data_ratio / len(args.aspects))
        data_aspect = args.all_data(aspect)
        rm_data_aspect_pos = []
        rm_data_aspect_neg = []
        for d in tqdm(data_aspect, desc=f"Generate RM train data"):
            aspect_instruction = args.aspects_reward_template[aspect].format(instruction=d["instruction"])
            data = {
                "dataid": d["dataid"],
                "instruction": aspect_instruction,
                "chosen": d["chosen"],
                "rejected": d["rejected"],
            }
            if len(d["chosen"]) >= len(d["rejected"]):
                rm_data_aspect_pos.append(data)
            else:
                rm_data_aspect_neg.append(data)

        balanced_ratio = np.array([len(rm_data_aspect_pos), len(rm_data_aspect_neg)]) / (len(rm_data_aspect_pos) + len(rm_data_aspect_neg))
        balanced_ratio = np.exp(balanced_ratio / balance_temperature)
        balanced_ratio = balanced_ratio / balanced_ratio.sum()

        rm_data_aspect = random.sample(rm_data_aspect_pos, k=int(num_samples*balanced_ratio[0])) + random.sample(rm_data_aspect_neg, k=int(num_samples*balanced_ratio[1]))
        data_aspects += rm_data_aspect
    return data_aspects

def rm_training():
    rm_data_aspects = balance_rm_training_data()
    rm_data_aspects_tmp_path = Tempfile(f"RM_{args.exp}_aspects.jsonl")
    print(f"RM data aspects length: {len(rm_data_aspects)}")
    print(rm_data_aspects[0]["instruction"])
    save_file_data(rm_data_aspects, rm_data_aspects_tmp_path)
    rm_aspects_task = args.RM_train(
        JOB_NAME=f"RM_{args.exp}_aspects",
        SAVE_MODEL=args.rm_model_uri("aspects"),
        TRAIN_FILE=rm_data_aspects_tmp_path,
        LENGTH_PENALTY=args.rm_length_penalty
    )
    rm_aspects_task.run()

def rm_training2():
    rm_train_tasklist = JobTaskList()
    for aspect in ["global"]:
        rm_data_aspect = []
        data_aspect = args.all_data(aspect)
        for d in tqdm(data_aspect, desc=f"Generate RM train data {aspect}"):
            if aspect in args.aspects:
                aspect_instruction = args.aspects_reward_template[aspect].format(instruction=d["instruction"])
            else:
                aspect_instruction = d["instruction"]
            rm_data_aspect.append({
                "dataid": d["dataid"],
                "instruction": aspect_instruction,
                "chosen": d["chosen"],
                "rejected": d["rejected"],
            })
        set_random_seed(args.seed)
        rm_data_aspect = random.sample(rm_data_aspect, k=int(len(rm_data_aspect)*args.rm_data_ratio))
        rm_data_aspect_tmp_path = Tempfile(f"RM_{args.exp}_{aspect}.jsonl")
        print(f"RM data {aspect} length: {len(rm_data_aspect)}")
        save_file_data(rm_data_aspect, rm_data_aspect_tmp_path)
        rm_aspect_task = args.RM_train(
            JOB_NAME=f"RM_{args.exp}_{aspect}",
            SAVE_MODEL=args.rm_model_uri(aspect),
            TRAIN_FILE=rm_data_aspect_tmp_path,
            LENGTH_PENALTY=args.rm_length_penalty,
        )
        rm_train_tasklist.append(rm_aspect_task)
    rm_train_tasklist.run()

def rm_inference(infer_aspects=args.aspects):
    rm_infer_tasks = JobTaskList()
    for aspect in infer_aspects + ["global"]:
        if aspect in args.aspects:
            rm_model = args.rm_model_uri("aspects")
        else:
            rm_model = args.rm_model_uri(aspect)
        
        inference_data = []
        for d in tqdm(args.all_data(), desc=f"Generate RM infer data"):
            for prefer in ["chosen", "rejected"]:
                if aspect in args.aspects:
                    aspect_instruction = args.aspects_reward_template[aspect].format(instruction=d["instruction"])
                else:
                    aspect_instruction = d["instruction"]
                inference_data.append({
                    "dataid": f"{d['dataid']}_{prefer}",
                    "instruction": aspect_instruction,
                    "response": d[prefer],
                    "aspect": d["aspect"],
                    "rm": f"RM_{aspect}"
                })
        rm_infer_temp_path = Tempfile(f"RM_{aspect}_infer.jsonl")
        save_file_data(inference_data, rm_infer_temp_path)
        rm_output_path = args.rm_output_paths[aspect]
        print(f"Inference data length for RM_{aspect}: {len(inference_data)}")

        infer_kwargs = {
            "WORLD_SIZE": 8,
            "MODEL_NAME_OR_PATH": rm_model,
            "INFER_FILE": rm_infer_temp_path,
            "OUTPUT_FILE": rm_output_path,
            "PROMPT": "instruction",
            "RESPONSE": "response",
            "TEMPLATE": "llama3",
            "BATCH_SIZE": 8,
            "CUTOFF_LEN": 4096,
        }
        infer_task = InferArgs(
            JOB_NAME=f"RM_{args.exp}_{aspect}_infer",
            INFER_TYPE="RewardScores",
            **infer_kwargs
        ).to_task()
        rm_infer_tasks.append(infer_task)
    rm_infer_tasks.run()

def dpo_training():
    DPO_tasks = JobTaskList()
    for sconf in args.select_configs:
        smodel = sconf.model
        DPO_tasks.append(
            args.DPO_train(
                JOB_NAME=f"DPO_{args.model_name}@{args.exp}_{sconf.name}",
                SAVE_MODEL=smodel,
                TRAIN_FILE=sconf.data_path,
            )
        )
    DPO_tasks.run()

def evaluate_alpacaeval():
    for sconf in args.select_configs:
        smodel = sconf.model
        AlpacaEvalEvaluator.new(
            model=smodel,
            version=2,
            annotators_config="weighted_alpaca_eval_gpt-4o",
            infer_args={
                "TEMPLATE": args.template,
                "WORLD_SIZE": 8,
            }
        )
    AlpacaEvalEvaluator.run_all()

def evaluate_pairwise():
    smodels = []
    for sconf in args.select_configs:
        smodel = sconf.model
        smodels.append(smodel)
    with file_lock(PairwiseEvaluator.LOCK_FILE_PATH, "PairwiseEval"):
        PairwiseEvaluator(
            model=smodels,
            infer_args={
                "TEMPLATE": args.template,
                "WORLD_SIZE": 8,
            },
        ).run()


if __name__ == "__main__":
    train_sft_model()
    dpo_all_train_evalute(alpacaeval=True, pairwiseval=True)
    rm_training()
    rm_inference()
    data_select()
    dpo_training()
    evaluate_alpacaeval()
    evaluate_pairwise()