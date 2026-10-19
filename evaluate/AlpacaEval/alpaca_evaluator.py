import os
import sys
from pathlib import Path

from utils import *

CURRENT_DIR = Path(__file__).parent.absolute()

logger = get_logger(__name__)
    
class AlpacaEvalEvaluator:
    LOCK_FILE_PATH = CURRENT_DIR / "alpacaeval.lock"
    OUTPUTS_DIR = nfs_uri("evaluate/alpacaeval")
    INFER_FILE = nfs_uri("evaluate/alpacaeval/alpaca_eval.json", user="data")
    DEFAULT_INFER_ARGS = {
        "WORLD_SIZE": 8,
        "INFER_FILE": INFER_FILE,
        "PROMPT": "content",
    }
    EVALUATORS = []

    def __init__(
        self, 
        model=None, 
        model_outputs=None, 
        annotators_config="weighted_alpaca_eval_gpt-4o",
        version=1,
        delay_infer=False,
        infer_args={},
    ):
        self.evaluator = f"AlpacaEval{version}"
        self.eval_model = model
        self.annotators_config = annotators_config
        self.delay_infer = delay_infer
        self.infer_args = self.DEFAULT_INFER_ARGS | infer_args


        if model_outputs is None:
            if self.eval_model is None:
                raise RuntimeError("Please provide a model to evaluate")
            self.infer_outputs()
        else:
            self.model_outputs = Path(model_outputs)

    
    def infer_outputs(self):

        def post_process():
            outputs_data = load_file_data(outputs)
            outputs_data = self.to_format(outputs_data, generator=generator)
            save_file_data(outputs_data, self.model_outputs)

        generator = self.eval_model if isinstance(self.eval_model, str) else self.eval_model.alias
        self.model_outputs = self.OUTPUTS_DIR / generator / "output.json"

        if self.model_outputs.exists():
            logger.info(f"{generator} alpacaeval outputs already exists: {self.model_outputs}")
            return

        job_name = f"[{self.evaluator}]-{generator}"
        outputs = Tempfile(f"{job_name}.jsonl")
        self.infertask = InferArgs(
            JOB_NAME=job_name,
            MODEL_NAME_OR_PATH=self.eval_model,
            OUTPUT_FILE=outputs,
            CUSTOM_POST_PROCESS=post_process,
            **self.infer_args
        ).to_task()
        
        if not self.delay_infer:
            self.infertask.run()

    def to_format(self, data, generator="test"):
        format_data = []
        for d in data:
            format_data.append({
                "instruction": d["content"],
                "output": d["predict"],
                "generator": generator,
                "dataset": "alpaca_eval",
                "datasplit":"eval"
            })
        return format_data

    def run(self, **kwargs):
        assert self.model_outputs.is_relative_to(self.OUTPUTS_DIR)
        assert self.model_outputs.exists()
        
        if self.evaluator == "AlpacaEval1":
            os.environ["IS_ALPACA_EVAL_2"] = "False"
        elif self.evaluator == "AlpacaEval2":
            os.environ["IS_ALPACA_EVAL_2"] = "True"
        else:            
            raise RuntimeError("Unsupported alpaca_eval version")

        logger.info(f"AlpacaEval Start ...")

        CUR_DIR = Path(__file__).parent
        if str(CUR_DIR) not in sys.path:
            sys.path.append(str(CUR_DIR))

        from .alpaca_eval.main import evaluate
        with file_lock(self.LOCK_FILE_PATH, "AlpacaEval"):
            evaluate(
                model_outputs=self.model_outputs,
                annotators_config=self.annotators_config,
                **kwargs
            )

        if str(CUR_DIR) in sys.path:
            sys.path.remove(str(CUR_DIR))

    @classmethod
    def new(cls, *args, **kwargs):
        assert "delay_infer" not in kwargs
        cls.EVALUATORS.append(cls(delay_infer=True, *args, **kwargs))
    
    @classmethod
    def run_all(cls, **kwargs):
        all_infer_tasks = [evaluator.infertask for evaluator in cls.EVALUATORS if hasattr(evaluator, "infertask")]
        JobTaskList(all_infer_tasks).run()
        for evaluator in cls.EVALUATORS:
            evaluator.run(**kwargs)
        cls.EVALUATORS = []
