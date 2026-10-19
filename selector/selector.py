import torch
import random
from tqdm import tqdm
import pandas as pd

from utils import *
from expconf import ExpConfig
from scipy.optimize import linprog

args = ExpConfig.from_yaml()

logger = get_logger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Selector:
    def __init__(self, df: pd.DataFrame=None, normlize=True, **kwargs):
        self.df = df or self.data_prepare(normlize=normlize, **kwargs)
        self.num_records = self.df.shape[0]
    
    def data_prepare(self, normlize=True):
        all_preference_data = {}
        for d in tqdm(args.all_data()):
            dataid, aspect = d["dataid"], d["aspect"]
            ground_scores = d["ground_scores"]
            overall_scores = d["overall_scores"]

            k = 1 if d["aspect_scores"][0] >= d["aspect_scores"][1] else -1
            assert k == 1
            ground_dict = {
                f"GD_{asp}": (ground_scores[asp][0] - ground_scores[asp][1])
                for asp in args.aspects
            }
            ground_dict = ground_dict | {"GD_overall": overall_scores[0] - overall_scores[1]}
            all_preference_data[dataid] = d | ground_dict

        for aspect in args.aspects + ["global"]:
            if not args.rm_output_paths[aspect].exists(): continue
            rm_output_data = load_file_data(args.rm_output_paths[aspect])
            for dc, dr in tqdm(zip(rm_output_data[::2], rm_output_data[1::2])):
                assert dc["dataid"].split("_")[0] == dr["dataid"].split("_")[0] and dc["aspect"] == dr["aspect"]
                dataid = dc["dataid"].split("_")[0]
                if isinstance(dc["predict"], list) and len(dc["predict"]) == 2:
                    chosen_score, rejected_score = dc["predict"][0], dr["predict"][0]
                    length_delta = dc["predict"][1] - dr["predict"][1]

                    length_logits = length_delta * args.rm_length_penalty

                    rm_score_delta = chosen_score - rejected_score - length_logits
                    if "DTLENS" not in all_preference_data[dataid]:
                        all_preference_data[dataid]["DTLENS"] = length_delta
                    all_preference_data[dataid][f"RMD_{aspect}_raw"] = chosen_score - rejected_score
                else:
                    rm_score_delta = dc["predict"] - dr["predict"]
                all_preference_data[dataid][f"RMD_{aspect}"] = rm_score_delta
        
        df = pd.DataFrame(all_preference_data.values()).set_index("dataid")
        if normlize:
            for aspect in args.aspects:
                RMD = df.loc[df["aspect"] != aspect, f"RMD_{aspect}"]
                quantile_val = RMD.abs().quantile(0.98)
                df[f"RMD_{aspect}"] = (df[f"RMD_{aspect}"] / quantile_val).clip(-1, 1)
                df[f"RMD_{aspect}_PDT"] = (RMD / quantile_val).clip(-1, 1)
        else:
            for aspect in args.aspects:
                RMD = df.loc[df["aspect"] != aspect, f"RMD_{aspect}"]
                df[f"RMD_{aspect}_PDT"] = RMD

        df["PDT"] = - df[[f"RMD_{aspect}_PDT" for aspect in args.aspects]].mean(axis=1)
        df.drop(columns=[f"RMD_{aspect}_PDT" for aspect in args.aspects], axis=1, inplace=True)
        return df

    def select(self, strategy: str="alldata", budget: float=0.1, random_state: int=42, include_columns=[], **kwargs):
        set_random_seed(random_state)
        budget = budget if budget < 1 else budget / 100
        num_target = int(self.num_records * budget)
        strategy_func = getattr(self, f"_strategy_{strategy}", self._strategy_no_impl)
        coreset_ids = strategy_func(num_target, **kwargs)
        records = self.df[["instruction", "chosen", "rejected"]+include_columns].iloc[coreset_ids].reset_index("dataid").to_dict(orient="records")
        return records

    def _strategy_no_impl(self, num_target: int, **kwargs):
        logger.info(f"Selection strategy is not implemented.")
        raise RuntimeError()

    def _strategy_alldata(self, num_target: int, **kwargs):
        all_ids = list(range(self.num_records))
        return all_ids

    def _strategy_ours(self, num_target: int, **kwargs):
        LENS = self.df["chosen"].apply(lambda x: len(x))
        top_indices = self.df["PDT"].nsmallest(num_target).index.to_list()
        print(f"{LENS.loc[top_indices].mean()} ({self.df.loc[top_indices, 'PDT'].mean()}, {self.df.loc[top_indices, 'PDT'].min()}, {self.df.loc[top_indices, 'PDT'].max()})")
        top_indices = self.df.index.get_indexer(top_indices)
        return top_indices

    def _strategy_random(self, num_target: int, **kwargs):
        all_ids = list(range(self.num_records))
        return random.sample(all_ids, num_target)

    def _strategy_high(self, num_target: int, **kwargs):
        LENS = self.df["chosen"].apply(lambda x: len(x))
        top_indices = self.df["PDT"].nlargest(num_target).index.to_list()
        print(f"{LENS.loc[top_indices].mean()} ({self.df.loc[top_indices, 'PDT'].mean()}, {self.df.loc[top_indices, 'PDT'].min()}, {self.df.loc[top_indices, 'PDT'].max()})")
        top_indices = self.df.index.get_indexer(top_indices)
        return top_indices

    def _strategy_mid(self, num_target: int, **kwargs):
        LENS = self.df["chosen"].apply(lambda x: len(x))
        num_exclusive = (self.num_records - num_target) // 2
        exclusive_indices = self.df["PDT"].nlargest(num_exclusive).index.to_list() + self.df["PDT"].nsmallest(num_exclusive).index.to_list()
        top_indices = list(set(self.df.index) - set(exclusive_indices))
        print(f"{LENS.loc[top_indices].mean()} ({self.df.loc[top_indices, 'PDT'].mean()}, {self.df.loc[top_indices, 'PDT'].min()}, {self.df.loc[top_indices, 'PDT'].max()})")
        top_indices = self.df.index.get_indexer(top_indices)
        return top_indices

    def _strategy_raf(self, num_target: int, **kwargs):
        LENS = self.df["chosen"].apply(lambda x: len(x))
        top_indices = self.df["RMD_global"].nlargest(num_target).index.to_list()
        print(f"{LENS.loc[top_indices].mean()} ({self.df.loc[top_indices, 'RMD_global'].mean()}, {self.df.loc[top_indices, 'RMD_global'].min()}, {self.df.loc[top_indices, 'RMD_global'].max()})")
        top_indices = self.df.index.get_indexer(top_indices)
        return top_indices

    def _strategy_ab2(self, num_target: int, **kwargs):
        if "PDT_raw" not in self.df.columns:
            df = self.df.copy()
            for aspect in args.aspects:
                RMD = df.loc[df["aspect"] != aspect, f"RMD_{aspect}_raw"]
                df[f"RMD_{aspect}_PDT_raw"] = RMD
            self.df["PDT_raw"] = - df[[f"RMD_{aspect}_PDT_raw" for aspect in args.aspects]].mean(axis=1)
        LENS = self.df["chosen"].apply(lambda x: len(x))
        top_indices = self.df["PDT_raw"].nsmallest(num_target).index.to_list()
        print(f"{LENS.loc[top_indices].mean()} ({self.df.loc[top_indices, 'PDT_raw'].mean()}, {self.df.loc[top_indices, 'PDT_raw'].min()}, {self.df.loc[top_indices, 'PDT_raw'].max()})")
        top_indices = self.df.index.get_indexer(top_indices)
        return top_indices
        
