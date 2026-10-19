import os
import json
import time
import logging
import argparse
import random
import shutil
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import yaml
import pickle
from tqdm import tqdm
from contextlib import contextmanager
from pytimeparse import parse as time_parse

__ALL__ = [
    "model_uri",
    "nfs_uri",
    "load_file_data", 
    "save_file_data", 
    "set_random_seed", 
    "decorated_print",
    "Tempfile",
    "copy_file_or_dir",
    "get_logger",
    "chunk_list",
    "to_float",
    "to_int",
    "cache_file_with_progress",
    "file_lock",
]

try:
    USER_NAME = os.environ["USER_NAME"]
except:
    raise RuntimeError("Set USER_NAME environment variable in utils/__init__.py")

class model_uri:
    def __init__(self, uri=None, model="test", version="test", sver=""):
        self.open_model = None
        if uri is not None:
            if uri.startswith("model://"):
                prefix, version = uri.split("/version=")
                model = prefix.split(".", 2)[-1]
            else:
                self.open_model = uri
                
        if "@" in version:
            ver, sver = version.split("@")
        else:
            ver = version

        self.model = model
        self.ver = ver
        self.sver = sver

    @property
    def version(self):
        if self.open_model: return ""
        if self.sver:
            return f"{self.ver}@{self.sver}"
        else:
            return self.ver

    def __str__(self):
        if self.open_model:
            return self.open_model
        return f"model://{USER_NAME}.{self.model}/version={self.version}"
    
    def __repr__(self):
        return f"model_uri({self.__str__()})"
    
    @property
    def alias(self):
        if self.open_model:
            return self.open_model.split("/")[-1]
        return f"{self.model}_{self.version}"


def nfs_uri(path, user=USER_NAME):
    return Path(f"/nfs/{user}/{path}")

def get_logger(
    name, 
    level=logging.INFO, 
    log_to_console=True,
    log_to_file=False, 
    log_format=None):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        return logger

    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s > %(message)s'

    # console handler
    if log_to_console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter(log_format, datefmt='%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    if log_to_file:
        log_path = Path(f"log/{name}.log")
        log_path.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

logger = get_logger(__name__)

def load_file_data(data_path):
    def load_json_data(data_path):
        try:
            with open(data_path, "r", encoding='utf-8') as file:
                data = json.load(file)
        except:
            with open(data_path, "r", encoding='utf-8') as file:
                data = [json.loads(line) for line in file]
        return data

    def load_pickle_data(data_path):
        data = []
        with open(data_path, 'rb') as f:
            while True:
                try:
                    data_item = pickle.load(f)
                    if isinstance(data_item, list):
                        data.extend(data_item)
                    else:
                        data.append(data_item)
                except EOFError:
                    break
        return data

    data_path = cache_file_with_progress(data_path)
    logger.info(f"Loading data from {data_path} ...")
    ext = os.path.splitext(data_path)[1]
    if "json" in ext:
        return load_json_data(data_path)
    elif "pkl" in ext:
        return load_pickle_data(data_path)
    else:
        raise ValueError(f"Unsupported file type for loading: {data_path}")

def save_file_data(data, data_path, **kwargs):
    def save_json_data(data, data_path, compact=False):
        if not data: return
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        if compact:
            dump_kwargs = {
                "indent": None,
                "ensure_ascii": False,
                "separators": (",", ":")
            }
        else:
            dump_kwargs = {
                "indent": 4,
                "ensure_ascii": False,
            }
        with open(data_path, "w", encoding='utf-8') as file:
            json.dump(data, file, **dump_kwargs)
            
    def save_pickle_data(data, data_path):
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)

    logger.info(f"Saving data to {data_path} ...")
    ext = os.path.splitext(data_path)[1]
    if "json" in ext:
        save_json_data(data, data_path, **kwargs)
    elif "pkl" in ext:
        save_pickle_data(data, data_path)
    else:
        raise ValueError(f"Unsupported file type for saving: {data_path}")

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def decorated_print(title, message):
    try:
        terminal_size = os.get_terminal_size()
        width = terminal_size.columns
    except OSError:
        width = 50

    decoration_line = '*' * width

    title = f" [ {title} ] "
    title_lwidth = (width - len(title)) // 2
    title_line = "*" * title_lwidth + title + "*" * (width - title_lwidth - len(title))
    print(title_line)
    print(message)
    print(decoration_line)

def copy_file_or_dir(src, dst):
    if not os.path.exists(src):
        raise FileNotFoundError(f"Source path does not exist: {src}")

    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
        logger.info(f"Folder copied: {src} -> {dst}")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        logger.info(f"File copied: {src} -> {dst}")

def chunk_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

class Tempfile:
    logger = logger

    def __init__(self, filename):
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        name, ext = os.path.splitext(filename)
        self.tempname = "".join([name, f"@{timestamp}", ext]).strip()
        self.filepath = nfs_uri(f"tmp/{self.tempname}")

    def __str__(self):
        return str(self.filepath)

    def __repr__(self):
        return str(self)

    def __fspath__(self): 
        return str(self.filepath)

    def __del__(self):
        self.delete()

    def exists(self):
        return self.filepath.exists()

    def delete(self):
        if self.filepath.exists():
            self.filepath.unlink()
            self.logger.info(f"Tempfile has been deleted: {self.tempname}")

def to_float(s):
    percent = 1
    if isinstance(s, str) and s.endswith("%"):
        s = s[:-1]
        percent = 0.01
    try:
        return float(s) * percent
    except ValueError:
        return False

def to_int(s):
    try:
        return int(s)
    except ValueError:
        return 0

def cache_file_with_progress(src_path, chunk_size=1024 * 1024):
    src_path = Path(src_path).absolute()
    nfs_dir = Path("/nfs")
    tmp_dir = Path("/tmp")

    if not src_path.is_relative_to(nfs_dir):
        return src_path
        
    dst_path = tmp_dir / src_path.relative_to(nfs_dir)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if dst_path.exists():
        src_stat = src_path.stat()
        dst_stat = dst_path.stat()
        if src_stat.st_size == dst_stat.st_size and src_stat.st_mtime == dst_stat.st_mtime:
            # logger.info(f"Cache file already exists for: {src_path}")
            return dst_path

    # logger.info(f"Caching file: {src_path} ...")
    total_size = os.path.getsize(src_path)

    class TqdmUpTo(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    class TqdmFileReader:
        def __init__(self, fd, pbar_instance):
            self.fd = fd
            self.pbar = pbar_instance
        def read(self, size):
            data = self.fd.read(size)
            if data:
                self.pbar.update(len(data))
            return data
        def __getattr__(self, attr):
            return getattr(self.fd, attr)

    try:
        with TqdmUpTo(
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            total=total_size,
            desc=f"[Caching File]",
            leave=False
        ) as pbar:
            with open(src_path, 'rb') as fsrc:
                with open(dst_path, 'wb') as fdst:
                    wrapped_fsrc = TqdmFileReader(fsrc, pbar)
                    shutil.copyfileobj(wrapped_fsrc, fdst, length=chunk_size)
                    
                    pbar.n = total_size
                    pbar.refresh()
        shutil.copystat(src_path, dst_path)

    except Exception as e:
        logger.error(f"Encountered an error while copying file: {e}")
        if dst_path.exists():
            dst_path.unlink()
        raise e

    return dst_path

@contextmanager
def file_lock(lock_file_path, name=None):
    try:
        while True:
            if not os.path.exists(lock_file_path):
                break
            wait_time = random.uniform(3, 7)
            print(f">> {name} lock file exists. Waiting for {wait_time:.2f} seconds...", end="\r", flush=True)
            time.sleep(wait_time)

        with open(lock_file_path, 'w') as lock_file:
            lock_file.write("Locked")

        yield

    finally:
        if os.path.exists(lock_file_path):
            os.remove(lock_file_path)
            print(f">> {name} lock file removed: {lock_file_path}")