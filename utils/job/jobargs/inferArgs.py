from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Optional, Callable

from ..jobArgs import LlamaFactoryArgs
from utils import *

logger = get_logger(__name__)

@dataclass
class InferArgs(LlamaFactoryArgs):
    ENTRY: str = f"{LlamaFactoryArgs.TARGET_DIR}/src/generate.py"
    WORLD_SIZE: int = 1
    INFER_TYPE: Optional[str] = field(
        default="Sequence",
        metadata={"help": "Inference type, default is Sequence.", "choices": ["Sequence", "Embedding", "RewardScores", "Perplexity", "OnPolicyWeight"]},
    )
    
    BATCH_SIZE: int = 8
    TEMPERATURE: float = 1.0
    TOP_P: float = 0.8
    TOP_K: int = 100
    SEED: int = 42
    CUTOFF_LEN: int = 2048
    MAX_NEW_TOKENS: int = 2048
    
    # Required
    MODEL_NAME_OR_PATH: str = None
    INFER_FILE: str = None
    OUTPUT_FILE: str = None
    PROMPT: str = None
    RESPONSE: str = None
    INPUT_COLUMNS: str = None
    TEMPLATE: str = None
    
    CUSTOM_POST_PROCESS: Optional[Callable] = None
    
    def __str__(self):
        params = [
            f"--model_name_or_path={self.MODEL_NAME_OR_PATH}",
            f"--batch_size={self.BATCH_SIZE}",
            f"--template={self.TEMPLATE}",
            f"--load_from=file",
            f"--do_sample",
            f"--temperature={self.TEMPERATURE}",
            f"--top_p={self.TOP_P}",
            f"--top_k={self.TOP_K}",
            f"--seed={self.SEED}",
            f"--cutoff_len={self.CUTOFF_LEN}",
            f"--max_new_tokens={self.MAX_NEW_TOKENS}",
            f"--infer_mode=default",
            f"--inputs={self.INFER_FILE}",
            f"--outputs={self.OUTPUT_FILE}",
            f"--prompt_column={self.PROMPT}",
            f"--response_column={self.RESPONSE}" if self.RESPONSE else "",
            f"--input_columns={self.INPUT_COLUMNS}" if self.INPUT_COLUMNS else "",
            f"--infer_type={self.INFER_TYPE}"
        ]
        params = self.repr_args(params)
        return super().__str__(params)

    def __post_init__(self):
        super().__post_init__()
        name, ext = os.path.splitext(self.OUTPUT_FILE)
        if self.INFER_TYPE == "Embedding":
            if ext != ".pkl":
                if ext:
                    raise ValueError(f"Output file must end with .pkl, but given {self.OUTPUT_FILE}")
                self.OUTPUT_FILE += ".pkl"
        elif not ext:
            self.OUTPUT_FILE += ".jsonl"
        # else:
            # if ext != ".jsonl" and ext != ".json":
                # if ext:
                    # raise ValueError(f"Output file must end with .jsonl, but given {self.OUTPUT_FILE}")
                
    
    def post_process(self):
        if self.WORLD_SIZE == 1:
            return 
        output_data = []
        for output_file in [
            Path(f"{self.OUTPUT_FILE}_{i}")
            for i in range(self.WORLD_SIZE)
        ]:
            if output_file.exists():
                output_data.extend(load_file_data(output_file))
                output_file.unlink()
        save_file_data(output_data, self.OUTPUT_FILE)

        if self.CUSTOM_POST_PROCESS is not None:
            self.CUSTOM_POST_PROCESS()