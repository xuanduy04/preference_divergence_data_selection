import itertools
from pathlib import Path
from functools import partial
from dataclasses import dataclass, field, fields
from transformers import HfArgumentParser
from typing import Any, Dict, List, Optional, Union, get_args, get_origin


from utils import *
from .baseconfig import BaseConfig

logger = get_logger(__name__)

default_config_path = "expconf/config.yaml"

@dataclass
class SelectConfig(BaseConfig):
    name: str
    strategy: str
    budget: Union[float, int]
    data_path: Path
    model: model_uri
    kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)

@dataclass
class ExpConfig(BaseConfig):
    exp: str = field(
        metadata={"help": "The name of the experiment."}
    )
    seed: int = field(
        default=42, 
        metadata={"help": "The seed of the experiment."}
    )
    dataset_name: Optional[str] = field(
        default="UltraFeedback", 
        metadata={"help": "The name of dataset used in the experiment."}
    )
    data_dir: Optional[Path] = field(
        default=None, 
        metadata={"help": "The directory of multi-aspect datasets."}
    )
    exp_dir: Optional[Path] = field(
        default=None, 
        metadata={"help": "The directory of the experiment."}
    )

    aspects_dict: Optional[Dict[str, str]] = field(
        default_factory=lambda: [],
        metadata={"help": "The preference aspects with description in the dataset."}
    )

    aspects_reward_template: Optional[Dict[str, str]] = field(
        default_factory=lambda: {},
        metadata={"help": "The preference aspects with reward template."}
    )

    model_name: Optional[str] = field(
        default="Llama3.1_8B", 
        metadata={"help": "The name of the model series."}
    )

    sft_model: Optional[str] = field(
        default=None, 
        metadata={"help": "The uri of the SFT model."}
    )

    rm_model_name: Optional[str] = field(
        default="Llama3.2-3B", 
        metadata={"help": "The name of the proxy model series."}
    )

    sft_rm_model: Optional[str] = field(
        default=None
    )

    rm_data_ratio: Optional[float] = field(
        default=0.3, 
        metadata={"help": "The ratio of RM training data."}
    )

    rm_length_penalty: Optional[float] = field(
        default=0.001, 
        metadata={"help": "The length penalty of the RM model."}
    )

    template: Optional[str] = field(
        default="llama3", 
        metadata={"help": "The template of the prompt."}
    )

    rm_template: Optional[str] = field(
        default="llama3", 
        metadata={"help": "The template of the prompt of rm model."}
    )

    deepspeed: Optional[str] = field(
        default="zero3", 
        metadata={"help": "The deepspeed config."}
    )
    
    select_onlocal: Optional[bool] = field(
        default=False
    )

    select_override: Optional[bool] = field(
        default=False, 
        metadata={"help": "Whether to override previous selection."}
    )

    select_configs: List[SelectConfig] = field(
        default_factory=list,
        metadata={"help": "The selection config to be used."}
    )

    exp_out_dir: Optional[Path] = field(
        default=None, 
        metadata={"help": "The directory of the experiment output."}
    )
    exp_sel_dir: Optional[Path] = field(
        default=None, 
        metadata={"help": "The directory of the experiment selection output."}
    )
    rm_output_paths: Optional[Dict[str, Path]] = field(
        default=None, 
        metadata={"help": "The paths of each aspect's RM inference output."}
    )

    def getvalue(self, name, default=None):
        if getattr(self, name, None) is None:
            return default
        return getattr(self, name)

    def type_inspect(self):
        for k, v in self.__dict__.items():
            if isinstance(v, Path):
                setattr(self, k, Path(v))

    def __post_init__(self):
        super().__post_init__(exclusive=["select_configs"])

        self.data_dir = nfs_uri(self.data_dir)
        self.exp_dir = self.getvalue("exp_dir", nfs_uri(f"DPOSEL/{self.exp}/{self.dataset_name}"))
        self.exp_out_dir = self.getvalue("exp_out_dir", self.exp_dir / "output")
        self.exp_sel_dir = self.getvalue("exp_sel_dir", self.exp_dir / "select")
        self.exp_out_dir.mkdir(parents=True, exist_ok=True)
        self.exp_sel_dir.mkdir(parents=True, exist_ok=True)

        assert self.aspects_dict, "aspects_dict must be specified"
        self.aspects = list(self.aspects_dict.keys())
        if not self.aspects_reward_template:
            for asp, persona in self.aspects_dict.items():
                self.aspects_reward_template[asp] = (
                    f"{persona}\n"
                    f"Now, answer the following instruction:\n"
                    f"{{instruction}}"
                )
        
        self.rm_output_paths = self.getvalue("rm_output_paths", 
            {
                aspect: self.exp_out_dir / f"RM_{aspect}_infer_output.jsonl" 
                for aspect in self.aspects
            }
        ) | {"global": self.exp_out_dir / "RM_global_infer_output.jsonl"}
        
        self.model_uri = lambda sname: model_uri(
            model=self.model_name, 
            version=f"DPO_{self.dataset_name}", 
            sver=f"{self.exp}_{sname}"
        )
        if self.sft_model is None:
            self.sft_model = model_uri(model=self.model_name, version="SFT_OpenHermes")
        
        self.rm_model_uri = lambda sname: model_uri(
            model=self.rm_model_name, 
            version=f"RM_{self.dataset_name}",
            sver=f"{self.exp}_{sname}"
        )
        if self.sft_rm_model is None:
            self.sft_rm_model = model_uri(model=self.rm_model_name, version="SFT_OpenHermes")

        dpo_train_kwargs = {
            "WORLD_SIZE": 16,
            "MODEL_NAME_OR_PATH": self.sft_model,
            "LR": 1e-6,
            "EPOCHS": 1,
            "PROMPT": "instruction",
            "CHOSEN": "chosen",
            "REJECTED": "rejected",
            "TEMPLATE": self.template,
            "LOGGING_STEPS": 0.005,
            "CUTOFF_LEN": 4096,
            "DEEPSPEED": self.deepspeed,
            "BATCH_SIZE": 32
        }
        self.DPO_train = DPOArgs.partial(**dpo_train_kwargs)

        rm_train_kwargs = {
            "WORLD_SIZE": 16,
            "MODEL_NAME_OR_PATH": self.sft_rm_model,
            "LR": 2e-5,
            "EPOCHS": 1,
            "PROMPT": "instruction",
            "CHOSEN": "chosen",
            "REJECTED": "rejected",
            "TEMPLATE": self.rm_template,
            "LOGGING_STEPS": 0.005,
            "CUTOFF_LEN": 4096,
            "DEEPSPEED": self.deepspeed,
            "BATCH_SIZE": 32
        }
        self.RM_train = RMArgs.partial(VAL_SIZE=0.05, **rm_train_kwargs)

        name_func = lambda strategy, budget: f"{strategy}_{(int(budget * 100) if budget < 1 else budget):02d}"
        
        valid_select_configs = []
        for sconf in self.select_configs:
            strategy, budget = sconf["strategy"], sconf["budget"]
            sconf["name"] = f"{name_func(strategy, budget)}"
            sconf["data_path"] = self.exp_sel_dir / f"{sconf['name']}.jsonl"
            sconf["budget"] = budget if budget < 1 else budget / 100
            sconf["model"] = self.model_uri(sconf['name'])
            sconf = SelectConfig(**sconf)
            valid_select_configs.append(sconf)
        self.select_configs = valid_select_configs
        
    @classmethod
    def from_yaml(cls, config_path=default_config_path):
        if not hasattr(cls, "expconf"):
            parser = HfArgumentParser((cls,))
            cls.expconf = parser.parse_yaml_file(yaml_file=config_path)[0]
            logger.info(f"\n{cls.expconf}")
        return cls.expconf

    @classmethod
    def from_cli(cls):
        if not hasattr(cls, "expconf"):
            parser = HfArgumentParser((cls,))
            cls.expconf = parser.parse_args_into_dataclasses()[0]
            logger.info(f"\n{cls.expconf}")
        return cls.expconf

    def all_data(self, aspect="global"):
        if not hasattr(self, "_all_data"):
            self._all_data = {
                asp: load_file_data(self.data_dir / f"{asp}.jsonl")
                for asp in self.aspects + ["global", "overall", "fine"]
            }
        return self._all_data[aspect]
        