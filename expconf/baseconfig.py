from pathlib import Path
from functools import partial
from dataclasses import dataclass, field, fields
from transformers import HfArgumentParser
from typing import Any, Dict, List, Optional, Union, get_args, get_origin


from utils import *

@dataclass
class BaseConfig:
    CONVERTERS = {
        Path: Path,
        int: int,
        float: float,
        str: str,
        bool: bool,
    }

    def _convert_value(self, value: Any, target_type: Any) -> Any:
        if value is None:
            return None

        origin = get_origin(target_type)
        if origin is Union:
            args = get_args(target_type)
            non_none_type = next((t for t in args if t is not type(None)), None)
            if non_none_type:
                return self._convert_value(value, non_none_type)
            else:
                return value

        if origin in (list, List) and isinstance(value, list):
            item_type = get_args(target_type)[0]
            return [self._convert_value(item, item_type) for item in value]

        if origin in (dict, Dict) and isinstance(value, dict):
            key_type, val_type = get_args(target_type)
            return {
                self._convert_value(k, key_type): self._convert_value(v, val_type)
                for k, v in value.items()
            }
        
        if origin is None and target_type in self.CONVERTERS:
            if isinstance(value, target_type):
                return value
            
            try:
                converter = self.CONVERTERS[target_type]
                return converter(value)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Cannot convert '{value}' (type: {type(value).__name__}) "
                    f"to type: {target_type.__name__}. Error: {e}"
                )

        return value

    def __post_init__(self, exclusive: List[str] = []):
        for f in fields(self):
            if f.name in exclusive: continue
            value = getattr(self, f.name)
            converted_value = self._convert_value(value, f.type)
            setattr(self, f.name, converted_value)

    def str_cutoff(self, value,  MAX_STR_LEN = 87):
        value_str = str(value)
        value_str = value_str.replace("/nfs", "/...")
        if len(value_str) > MAX_STR_LEN:
            value_str = value_str[:MAX_STR_LEN - 3] + '...'
        if isinstance(value, str):
            value_str = f"\"{value_str}\""
        return value_str

    def __str__(self, level: int = 0) -> str:
        class_name = self.__class__.__name__
        parts = []
        indent = " " * ((level+1) * 4)
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, (list, tuple)):
                value_str = ["["] + [
                        f"{indent}{' '*4}{item.__str__(level+2)}" if isinstance(item, BaseConfig) else f"{indent}{' '*4}{self.str_cutoff(item)}"
                        for item in value
                    ]
                if len(value_str) == 1:
                    value_str = "".join(value_str + ["]"])
                else:
                    value_str = "\n".join(value_str + [f"{indent}]"])
            elif isinstance(value, dict):
                value_str = ["{"] + [
                        f"{indent}{' '*4}{k}: {v.__str__(level+2)}" if isinstance(v, BaseConfig) else f"{indent}{' '*4}{k}: {self.str_cutoff(v)}"
                        for k, v in value.items()
                    ]
                if len(value_str) == 1:
                    value_str = "".join(value_str + ["}"])
                else:
                    value_str = "\n".join(value_str + [f"{indent}}}"])
            else:
                value_str = f"{item.__str__(level+1)}" if isinstance(value, BaseConfig) else f"{self.str_cutoff(value)}"
            
            
            parts.append(f"{indent}{f.name}: {value_str}")

        body = '\n'.join(parts)
        if not body:
             return f"{class_name}()"
        return f"{class_name} (\n{body}\n{indent[:-4]})"