from dataclasses import dataclass, field
from typing import Union

from ..jobArgs import LlamaFactoryArgs

@dataclass
class SFTArgs(LlamaFactoryArgs):
    ENTRY: str = f"{LlamaFactoryArgs.TARGET_DIR}/src/train_bash.py"
    WORLD_SIZE: int = 8
    
    MAX_STEPS: int = -1
    EPOCHS: int = 3
    LR: float  = 2e-5
    CUTOFF_LEN: int = 4096
    WARMUP_RATIO: float = 0.03
    BATCH_SIZE: int = 128
    BATCH_PER_DEVICE: int = 8
    LOGGING_STEPS: float = 0.01
    LR_SCHEDULER: str = "cosine"
    SAVE_STEPS: Union[int, float] = 0.99999
    DEEPSPEED: str = "zero3"
    
    # Required
    MODEL_NAME_OR_PATH: str = None
    TRAIN_FILE: str = None
    PROMPT: str = None
    QUERY: str = None
    RESPONSE: str = None
    MESSAGES: str = None
    ROLE_TAG: str = "role"
    CONTENT_TAG: str = "content"
    USER_TAG: str = "user"
    ASSISTANT_TAG: str = "assistant"
    TEMPLATE: str = None
    SAVE_MODEL: str = None

    def __post_init__(self):
        super().__post_init__()
        self.GRADACCU_STEPS: int = self.BATCH_SIZE//(self.BATCH_PER_DEVICE * self.WORLD_SIZE)
        assert (bool(self.PROMPT) and bool(self.RESPONSE)) + bool(self.MESSAGES) == 1
        assert (self.MAX_STEPS > 0) + (self.EPOCHS > 0) == 1

    def __str__(self):
        params = [
            f"--stage=sft",
            f"--model_name_or_path={self.MODEL_NAME_OR_PATH}",
            f"--do_train",
            f"--file_name={self.TRAIN_FILE}",

            f"--prompt={self.PROMPT}" if (self.PROMPT and self.RESPONSE) else "",
            f"--query={self.QUERY}" if (self.PROMPT and self.RESPONSE) and self.QUERY else "",
            f"--response={self.RESPONSE}" if (self.PROMPT and self.RESPONSE) else "",

            f"--messages={self.MESSAGES}" if self.MESSAGES else "",
            f"--role_tag={self.ROLE_TAG}" if self.MESSAGES else "",
            f"--content_tag={self.CONTENT_TAG}" if self.MESSAGES else "",
            f"--user_tag={self.USER_TAG}" if self.MESSAGES else "",
            f"--assistant_tag={self.ASSISTANT_TAG}" if self.MESSAGES else "",

            f"--template={self.TEMPLATE}",
            f"--finetuning_type=full",
            f"--output_dir=local/tmp/ckpt_save_path/",
            f"--overwrite_cache",
            f"--per_device_train_batch_size={self.BATCH_PER_DEVICE}",
            f"--gradient_accumulation_steps={self.GRADACCU_STEPS}",
            f"--lr_scheduler_type={self.LR_SCHEDULER}",
            f"--logging_steps={self.LOGGING_STEPS}",
            f"--save_strategy=steps",
            f"--save_steps={self.SAVE_STEPS}",
            f"--num_train_epochs={self.EPOCHS}" if self.MAX_STEPS <= 0 else "",
            f"--max_steps={self.MAX_STEPS}" if self.MAX_STEPS > 0 else "",
            f"--learning_rate={self.LR}",
            f"--warmup_ratio={self.WARMUP_RATIO}",
            f"--cutoff_len={self.CUTOFF_LEN}",
            f"--preprocessing_num_workers=8",
            f"--dataloader_num_workers=4",
            f"--plot_loss",
            f"--deepspeed={self.TARGET_DIR}/scripts/ds_{self.DEEPSPEED}.json",
            f"--bf16",
        ]
        params = self.repr_args(params)
        return super().__str__(params)