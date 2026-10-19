import contextlib
import copy
import glob
import itertools
import logging
import os
import pathlib
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import datasets
import numpy as np
import numpy.typing as npt
import pandas as pd
import pkg_resources
import tqdm
import yaml

from . import constants

# don't load from utils to avoid unnecessary dependencies
AnyPath = Union[str, os.PathLike, pathlib.Path]
AnyData = Union[Sequence[dict[str, Any]], pd.DataFrame, datasets.Dataset]
DUMMY_EXAMPLE = dict(instruction="1+1=", output_1="2", input="", output_2="3")


class Timer:
    """Timer context manager"""

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start = time.time()
        return self

    def __exit__(self, *args):
        """Stop the context manager timer"""
        self.end = time.time()
        self.duration = self.end - self.start

    def __str__(self):
        return f"{self.duration:.1f} seconds"


@contextlib.contextmanager
def silent():
    """Context manager to remove all outputs and warnings."""
    import IPython

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), DisableLogger(), IPython.utils.io.capture_output():
        yield


def get_all_clients(
    client_config_path: AnyPath,
    model_name: str,
    default_client_class: str,
    get_backwards_compatible_configs: Callable,
    backward_compatibility_kwargs: dict = {},
    **kwargs,
) -> list:
    """Returns a list of different kwargs to pass to the client, each element corresponds to one possible client.
    For more information see `client_configs/README.md`.
    """

    client_config_path = Path(client_config_path)
    if client_config_path.is_file():
        with open(client_config_path) as f:
            all_client_configs = yaml.safe_load(f)

        client_configs = []

        if model_name in all_client_configs:
            if "default" in all_client_configs[model_name]:
                assert "default" in all_client_configs, "default client was asked for but not found"
                client_configs = client_configs + all_client_configs["default"]
                # remove "default" from the list of configs for this model
                all_client_configs[model_name] = [
                    config for config in all_client_configs[model_name] if config != "default"
                ]

            client_configs = client_configs + all_client_configs[model_name]

        else:
            assert (
                "default" in all_client_configs
            ), f"default client config is required as there are no model specific configs for {model_name}"
            client_configs = all_client_configs["default"]

    else:
        # backward compatibility
        logging.warning(
            f"{client_config_path} wasn't found. We are using environment variables to construct the client configs."
            "This is the old and non-recommended way of doing it. Please see `client_configs/README.md` for the "
            "recommended way of specifying client configs."
        )
        client_configs = get_backwards_compatible_configs(**backward_compatibility_kwargs)

    all_clients = []
    for config in client_configs:
        client_class = config.pop("client_class", default_client_class)
        ClientClass = import_class(client_class)
        all_clients.append(ClientClass(**config, **kwargs))

    return all_clients

def prompt_to_chatml(prompt: str, start_token: str = "<|im_start|>", end_token: str = "<|im_end|>"):
    r"""Convert a text prompt to ChatML formal

    Examples
    --------
    >>> prompt = (
    ... "<|im_start|>system\n"
    ... "You are a helpful assistant.\n<|im_end|>\n"
    ... "<|im_start|>system name=example_user\nKnock knock.\n<|im_end|>\n<|im_start|>system name=example_assistant\n"
    ... "Who's there?\n<|im_end|>\n<|im_start|>user\nOrange.\n<|im_end|>"
    ... )
    >>> print(prompt)
    <|im_start|>system
    You are a helpful assistant.
    <|im_end|>
    <|im_start|>system name=example_user
    Knock knock.
    <|im_end|>
    <|im_start|>system name=example_assistant
    Who's there?
    <|im_end|>
    <|im_start|>user
    Orange.
    <|im_end|>
    >>> prompt_to_chatml(prompt)
    [{'content': 'You are a helpful assistant.', 'role': 'system'},
      {'content': 'Knock knock.', 'role': 'system', 'name': 'example_user'},
      {'content': "Who's there?", 'role': 'system', 'name': 'example_assistant'},
      {'content': 'Orange.', 'role': 'user'}]
    """
    prompt = prompt.strip()
    assert prompt.startswith(start_token)
    assert prompt.endswith(end_token)

    message = []
    for p in prompt.split("<|im_start|>")[1:]:
        newline_splitted = p.split("\n", 1)
        role = newline_splitted[0].strip()
        content = newline_splitted[1].split(end_token, 1)[0].strip()

        if role.startswith("system") and role != "system":
            # based on https://github.com/openai/openai-cookbook/blob/main/examples
            # /How_to_format_inputs_to_ChatGPT_models.ipynb
            # and https://github.com/openai/openai-python/blob/main/chatml.md it seems that system can specify a
            # dictionary of other args
            other_params = _string_to_dict(role.split("system", 1)[-1])
            role = "system"
        else:
            other_params = dict()

        message.append(dict(content=content, role=role, **other_params))

    return message