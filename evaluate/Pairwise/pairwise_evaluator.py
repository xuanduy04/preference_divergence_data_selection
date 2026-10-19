import os
import sys
from pathlib import Path
import pandas as pd

from utils import *
from expconf import ExpConfig
from .openai import openai_completions

args = ExpConfig.from_yaml()
CURRENT_DIR = Path(__file__).parent.absolute()
logger = get_logger(__name__)

system_prompt = "You are a helpful and precise assistant for checking the quality of the answer."
eval_instruction = (
    "We would like to request your feedback on the performance of two AI assistants "
    "in response to the user question displayed above.\n"
    "Please rate the helpfulness, relevance, accuracy, level of details of their responses. "
    "Each assistant receives an overall score on a scale of 1 to 10, where a higher score "
    "indicates better overall performance. Please first output a single line containing "
    "only two values indicating the scores for Assistant 1 and 2, respectively. The two "
    "scores are separated by a space. In the subsequent line, please provide a comprehensive "
    "explanation of your evaluation, avoiding any potential length bias and ensuring that the order "
    "in which the responses were presented does not affect your judgment.\n"
)

default_pweval_prompt_template = (
    # system_prompt + "\n"
    "[Question]\n"
    "{instruction}\n"
    "[The Start of Assistant 1's Answer]\n"
    "{out1}\n"
    "[The End of Assistant 1's Answer]\n"
    "[The Start of Assistant 2's Answer]\n"
    "{out2}\n"
    "[The End of Assistant 2's Answer]\n\n"
) + eval_instruction
    
class PairwiseEvaluator:
    LOCK_FILE_PATH = CURRENT_DIR / "pairwiseval.lock"
    OUTPUTS_DIR = nfs_uri("evaluate/pairwise")
    RESULTS_DIR = CURRENT_DIR / "results"
    TEST_COLUMNS = ["WizardLM", "Sinstruct", "Vicuna", "Koala", "Lima"] + ["Avg"]
    TEST_DATASET = nfs_uri("evaluate/pairwise/all_test.json")
    BMODEL = args.model_uri("overall")
    BMODEL.sver = "exp30_overall"
    JUDGE_MODEL = "qwen_max"
    JUDGE_MODEL_DECODING_KWARGS = {
        "max_tokens": 6
    }
    CACHE_PATH  = CURRENT_DIR / "cache" / JUDGE_MODEL / "annotations_seed0_configs.json"
    PWEVAL_SYSTEM_PROMPT = system_prompt
    PWEVAL_PROMPT_TEMPLATE = default_pweval_prompt_template
    DEFAULT_INFER_ARGS = {
        "WORLD_SIZE": 8,
        "PROMPT": "instruction",
    }

    def __init__(
        self, 
        model,
        bmodel = None,
        infer_args={},
        override=False
    ):
        if not isinstance(model, list):
            model = [model]
        self.emodels = model
        self.bmodel = bmodel or self.BMODEL
        self.infer_args = self.DEFAULT_INFER_ARGS | infer_args
        self.evaluator = f"PairEval"
        self.override = override

        self.load_result_df()
        self.load_pweval_cache()
        self.check_tests()
    
    def check_tests(self):
        for emodel in self.emodels.copy():
            if emodel.alias in self.result_df.index:
                logger.info(f"{emodel.alias} already evaluated, skip ...")
                self.emodels.remove(emodel)

    def load_result_df(self):
        expected_columns = list(self.TEST_COLUMNS)

        self.result_df_path = self.RESULTS_DIR / self.JUDGE_MODEL / f"evaluate_vs[{self.bmodel.sver}].csv"

        if self.result_df_path.exists():
            self.result_df = pd.read_csv(self.result_df_path, index_col=0)
        else:
            self.result_df = pd.DataFrame(columns=expected_columns)
    
    def load_pweval_cache(self):
        self.CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if self.CACHE_PATH.exists():
            with open(self.CACHE_PATH, "r") as f:
                self.pweval_cache = json.load(f)
        else:
            self.pweval_cache = {}
    
    def save_pweval_cache(self):
        with open(self.CACHE_PATH, "w") as f:
            json.dump(self.pweval_cache, f)
    
    def add_results(self, emodel, results):
        assert emodel.alias not in self.result_df.index
        self.result_df.loc[emodel.alias] = results
        self.result_df.sort_index(inplace=True)
        self.save_result_df()

    def save_result_df(self):
        self.result_df_path.parent.mkdir(parents=True, exist_ok=True)
        self.result_df.to_csv(self.result_df_path)

    def infer_output_path(self, model):
        return self.OUTPUTS_DIR / model.alias / f"infer_output.jsonl"
    
    def pweval_data_path(self, model):
        return self.OUTPUTS_DIR / model.alias / f"pweval@{self.bmodel.sver}.jsonl"

    def pweval_output_path(self, model):
        return self.OUTPUTS_DIR / model.alias / f"pweval_output@{self.bmodel.sver}@{self.JUDGE_MODEL}.jsonl"
    
    def path_exists(self, path):
        if path.exists():
            if not self.override:
                logger.info(f"{path} already exists, skip ...")
                return True
            else:
                path.unlink()
                return False

    def infer_test_outputs(self):
        infer_tasks = JobTaskList()
        for model in self.emodels + [self.bmodel]:
            infer_output_path = self.infer_output_path(model)
            if self.path_exists(infer_output_path): continue
            infer_task = InferArgs(
                JOB_NAME=f"[{self.evaluator}_infer]-{model.alias}",
                MODEL_NAME_OR_PATH=model,
                INFER_FILE=self.TEST_DATASET,
                OUTPUT_FILE=infer_output_path,
                **self.infer_args
            ).to_task()
            infer_tasks.append(infer_task)
        logger.info(f"Infer test outputs for [{len(infer_tasks)}] models ...")
        infer_tasks.run()
    
    def generate_pweval_data(self):
        for emodel in self.emodels:
            pweval_data_path = self.pweval_data_path(emodel)
            if self.path_exists(pweval_data_path): continue

            emodel_infer_output_path = self.infer_output_path(emodel)
            bmodel_infer_output_path = self.infer_output_path(self.bmodel)
            emodel_infer_output = load_file_data(emodel_infer_output_path)
            bmodel_infer_output = load_file_data(bmodel_infer_output_path)

            pweval_data = []
            for eout, bout in zip(emodel_infer_output, bmodel_infer_output):
                assert (instruction := eout["instruction"]) == bout["instruction"]
                eval_prompt = self.PWEVAL_PROMPT_TEMPLATE.format(
                    instruction=instruction, 
                    out1=eout['predict'],
                    out2=bout['predict'],
                    generator=emodel.alias
                )
                pweval_data.append({
                    "testset": eout["testset"],
                    "instruction": eval_prompt,
                    "swap": False
                })

                eval_prompt = self.PWEVAL_PROMPT_TEMPLATE.format(
                    instruction=instruction, 
                    out1=bout['predict'],
                    out2=eout['predict'],
                    generator=emodel.alias
                )
                pweval_data.append({
                    "testset": eout["testset"],
                    "instruction": eval_prompt,
                    "swap": True
                })
            save_file_data(pweval_data, pweval_data_path)

    def infer_pweval_outputs(self):
        for emodel in self.emodels:
            pweval_data_path = self.pweval_data_path(emodel)
            assert pweval_data_path.exists()
            pweval_output_path = self.pweval_output_path(emodel)
            if not self.path_exists(pweval_output_path):
                pweval_data = load_file_data(pweval_data_path)
                pweval_batch_size = 128
                for batch_pweval_data in tqdm([
                    pweval_data[i:i+pweval_batch_size] 
                    for i in range(0, len(pweval_data), pweval_batch_size)
                ]):
                    request_pwds = []
                    for pwd in batch_pweval_data:
                        if pwd["instruction"] in self.pweval_cache:
                            pwd["predict"] = self.pweval_cache[pwd["instruction"]]
                        else:
                            request_pwds.append(pwd)

                    request_prompts = [
                        [
                            {"role": "system", "content": self.PWEVAL_SYSTEM_PROMPT},
                            {"role": "user", "content": pwd["instruction"]}
                        ]
                        for pwd in request_pwds
                    ]
                    if not request_prompts:
                        continue

                    completions = openai_completions(
                        request_prompts,
                        model_name=self.JUDGE_MODEL,
                        **self.JUDGE_MODEL_DECODING_KWARGS
                    )["completions"]

                    for pwd, comp in zip(request_pwds, completions):
                        pwd["predict"] = comp
                        self.pweval_cache[pwd["instruction"]] = comp
                    
                    self.save_pweval_cache()

                assert all("predict" in pwd for pwd in pweval_data)
                save_file_data(pweval_data, pweval_output_path)
            pweval_output = load_file_data(pweval_output_path)
            self.calc_winning_scores(emodel, pweval_output)

    
    def calc_winning_scores(self, emodel, pweval_output):
        results = {}
        for test in self.TEST_COLUMNS:
            win, tie, lose, error = 0, 0, 0, 0
            for pwd in pweval_output:
                if test == "Avg" or pwd["testset"] == test:
                    pweval_scores = pwd["predict"].split("\n")[0].split(" ")
                    try:
                        score0, score1 = int(pweval_scores[0]), int(pweval_scores[1])
                    except:
                        error += 1
                        continue
                    if pwd["swap"]: score0, score1 = score1, score0
                    if score0 > score1: win += 1
                    elif score0 == score1: tie += 1
                    else: lose += 1
            total = win + tie + lose + error
            win_score = (win - lose) / (total) + 1
            results[test] = win_score
        self.add_results(emodel, results)


    def run(self):
        self.infer_test_outputs()
        self.generate_pweval_data()
        self.infer_pweval_outputs()