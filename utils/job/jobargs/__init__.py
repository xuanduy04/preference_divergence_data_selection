from .sftArgs import SFTArgs
from .inferArgs import InferArgs
from .rmArgs import RMArgs
from .dpoArgs import DPOArgs

job_args_map = {
    "sft": SFTArgs,
    "infer": InferArgs,
    "rm": RMArgs,
    "dpo": DPOArgs,
}