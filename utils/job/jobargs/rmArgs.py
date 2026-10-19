from dataclasses import dataclass, field
from typing import Union

from ..jobArgs import LlamaFactoryArgs

@dataclass
class RMArgs(LlamaFactoryArgs):
    ENTRY: str = f"{LlamaFactoryArgs.TARGET_DIR}/src/train_bash.py"
    WORLD_SIZE: int = 8
    
    MAX_STEPS: int = -1
    EPOCHS: int = 2
    LR: float  = 5e-6
    WARMUP_RATIO: float = 0.05
    CUTOFF_LEN: int = 4096
    BATCH_SIZE: int = 64
    BATCH_PER_DEVICE: int = 2
    LOGGING_STEPS: float = 0.05
    LR_SCHEDULER: str = "cosine"
    SAVE_STEPS: Union[int, float] = 0.99999
    VAL_SIZE: float = 0.05
    DEEPSPEED: str = "zero3"
    
    
    # Required
    MODEL_NAME_OR_PATH: str = None
    TRAIN_FILE: str = None
    PROMPT: str = None
    QUERY: str = None
    CHOSEN: str = None
    REJECTED: str = None
    TEMPLATE: str = None
    SAVE_MODEL: str = None

    LENGTH_PENALTY: float = 0
    
    def __post_init__(self):
        super().__post_init__()
        self.GRADACCU_STEPS: int = self.BATCH_SIZE//(self.BATCH_PER_DEVICE * self.WORLD_SIZE)

    def __str__(self):
        params = [
            f"--stage=rm",
            f"--model_name_or_path={self.MODEL_NAME_OR_PATH}",
            f"--do_train",
            f"--file_name={self.TRAIN_FILE}",
            f"--ranking",
            f"--prompt={self.PROMPT}",
            f"--query={self.QUERY}" if self.QUERY else "",
            f"--chosen={self.CHOSEN}",
            f"--rejected={self.REJECTED}",
            f"--template={self.TEMPLATE}",
            f"--finetuning_type=full",
            f"--output_dir=local/tmp/ckpt_save_path/",
            f"--overwrite_cache",
            f"--per_device_train_batch_size={self.BATCH_PER_DEVICE}",
            f"--gradient_accumulation_steps={self.GRADACCU_STEPS}",
            f"--per_device_eval_batch_size={self.BATCH_PER_DEVICE}",
            f"--lr_scheduler_type={self.LR_SCHEDULER}",
            f"--logging_steps={self.LOGGING_STEPS}",
            f"--save_strategy=steps",
            f"--save_steps={self.SAVE_STEPS}",
            f"--num_train_epochs={self.EPOCHS}",
            f"--learning_rate={self.LR}",
            f"--warmup_ratio={self.WARMUP_RATIO}",
            f"--cutoff_len={self.CUTOFF_LEN}",
            f"--preprocessing_num_workers=8",
            f"--dataloader_num_workers=4",
            f"--plot_loss",
            f"--deepspeed={self.TARGET_DIR}/scripts/ds_{self.DEEPSPEED}.json",
            f"--rm_length_penalty={self.LENGTH_PENALTY}" if self.LENGTH_PENALTY > 0 else "",
            f"--bf16",
        ]
        if self.VAL_SIZE:
            params.extend([
                f"--do_eval",
                f"--val_size={self.VAL_SIZE}",
                f"--eval_strategy=steps",
                f"--eval_steps={self.LOGGING_STEPS*4}"
            ])
        params = self.repr_args(params)
        return super().__str__(params)