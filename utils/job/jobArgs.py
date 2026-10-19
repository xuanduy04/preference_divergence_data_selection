import os
import json
import shutil
import inspect
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import contextmanager

from utils.tools import get_logger

JOB_DIR = os.path.relpath(os.path.dirname(__file__))

logger = get_logger(__name__)

class ArgInitError(Exception):
    pass

class JobArgsPartial:
    def __init__(self, argsclass, to_task=True, **kwargs):
        self.argsclass = argsclass
        self.to_task = to_task
        self.kwargs = kwargs
    
    def __call__(self, **kwargs):
        kwargs = self.kwargs | kwargs
        args = self.argsclass(**kwargs)
        if self.to_task:
            return args.to_task()
        else:
            return args

@dataclass
class JobArgs:
    JOB_NAME: str

    JOB_CONFIG_PATH: str = os.path.join(JOB_DIR, "config.json")
    QUEUE: str = None
    
    ENTRY: str = None
    SAVE_MODEL: str = None
    WORLD_SIZE: int = 1
    
    SOURCE_DIR = None
    TARGET_DIR = JOB_DIR
    
    def __post_init__(self):
        assert os.path.exists(self.JOB_CONFIG_PATH)
        with open(self.JOB_CONFIG_PATH, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        
        
    def repr_args(self, args):
        args = [arg for arg in args if arg]
        for arg in args:
            if "None" in arg:
                raise ArgInitError(f"Uninitlized arg: {arg}")
        return " ".join(args)

    def __str__(self, params=None, custom_job_args=""):
        args = [
            f"--entry={self.ENTRY}",
            f"--params=\"{params}\"",
            f"--queue={self.QUEUE}",
            f"--worker_count={self.WORLD_SIZE}",
            f"--cluster_file={self.TARGET_DIR}/cluster.json",
            f"--job_name={self.JOB_NAME}",
            f"--save_model={self.SAVE_MODEL}" if self.SAVE_MODEL else "",
            f"--env=\"NCCL_NVLS_ENABLE=0\""

            f"{custom_job_args}"
        ]

        return self.repr_args(args)
    
    def to_task(self):
        from .jobTask import JobTask
        return JobTask(args=self)
    
    @contextmanager
    def source_framework(self):
        try:
            logger.info(f"Copying {self.SOURCE_DIR} to ./{self.TARGET_DIR}")
            shutil.copytree(self.SOURCE_DIR, self.TARGET_DIR, dirs_exist_ok=True)
            shutil.copyfile(os.path.join(self.TARGET_DIR, "requirements.txt"), "./requirements.txt")
            yield self
        except Exception as e:
            logger.info(f"source_framework failed: {e}")
        finally:
            logger.info(f"Removing ./{self.TARGET_DIR}")
            if os.path.exists(self.TARGET_DIR):
                shutil.rmtree(self.TARGET_DIR)
            if os.path.exists("./requirements.txt"):
                os.remove("./requirements.txt")
                
    def pre_process(self):
        pass
    
    def post_process(self):
        pass

    @classmethod
    def partial(cls, to_task=True, **kwargs):
        return JobArgsPartial(cls, to_task=to_task, **kwargs)

@dataclass
class CustomArgs(JobArgs):
    SOURCE_DIR: str = os.path.expanduser("~/codebase/Framework/Custom")
    TARGET_DIR: str = "Custom"
    
    WORLD_SIZE: int = 1

    ENTRY: str = None
    PARAMS: dict = field(default_factory=dict)
    
    def __str__(self):
        params = [
            f"--{key}={value}"
            for key, value in self.PARAMS.items()
        ]
        params = self.repr_args(params)
        return super().__str__(params)

    def at_job(self):
        return Path(self.TARGET_DIR).is_dir()

    def to_task(self, call_func=None):
        if self.at_job():
            return type("", (), {"run": call_func})
        else:
            return super().to_task()

    @classmethod
    def wrapper(cls, onlocal=False, **kwargs):
        def decorator(func):
            file_path_obj = Path(inspect.getfile(func)).resolve()
            kwargs["ENTRY"] = file_path_obj.relative_to(Path.cwd())
            def run():
                func() if onlocal else cls(**kwargs).to_task(func).run()
            return run
        return decorator


    
@dataclass
class LlamaFactoryArgs(JobArgs):
    SOURCE_DIR = os.path.expanduser("~/codebase/Framework/LlamaFactory")
    TARGET_DIR = "LlamaFactory"
    
    def __str__(self, *args, **kwargs):
        return super().__str__(*args, **kwargs)
    
