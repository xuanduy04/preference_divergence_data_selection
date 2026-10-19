import ast
import getpass
import os
from functools import partial
from pathlib import Path

CURRENT_DIR = Path(__file__).parent.absolute()
BASE_DIR = Path(__file__).parents[2].absolute()

### API specific ###
API_MAX_CONCURRENCY = int(os.environ.get("API_MAX_CONCURRENCY", 10))

OPENAI_MAX_CONCURRENCY = int(os.environ.get("OPENAI_MAX_CONCURRENCY", 5))
OPENAI_CLIENT_CONFIG_PATH = os.environ.get("OPENAI_CLIENT_CONFIG_PATH", BASE_DIR / "client_configs/openai_configs.yaml")
# the following is for backward compatibility, the recommended way is to use OPENAI_CLIENT_CONFIG_PATH
OPENAI_API_KEYS = os.environ.get("OPENAI_API_KEYS", os.environ.get("OPENAI_API_KEY", None))
if isinstance(OPENAI_API_KEYS, str):
    OPENAI_API_KEYS = OPENAI_API_KEYS.split(",")
OPENAI_ORGANIZATION_IDS = os.environ.get("OPENAI_ORGANIZATION_IDS", None)
if isinstance(OPENAI_ORGANIZATION_IDS, str):
    OPENAI_ORGANIZATION_IDS = OPENAI_ORGANIZATION_IDS.split(",")