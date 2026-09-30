import warnings
import gzip
import json
import logging

import torch


def silence_logs():
    warnings.filterwarnings("ignore")
    for name in ("pytorch_lightning", "lightning.pytorch", "lightning", "darts",
                 "pytorch_lightning.utilities.rank_zero",
                 "pytorch_lightning.accelerators.cuda"):
        logging.getLogger(name).setLevel(logging.ERROR)


def load_gz_json(path):
    with gzip.open(path, mode="r") as f:
        return json.loads(f.read())

def get_trainer_kwargs():
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        torch.set_float32_matmul_precision("high")
        gpu = {"accelerator": "gpu", "devices": [0]}
    else:
        gpu = {"accelerator": "cpu"}
    gpu.update({"enable_progress_bar": False, "logger": False,
                "enable_model_summary": False})
    return gpu, has_gpu

def load_lyapunov(path):
    """Parse 'Name: value' lines. First occurrence wins (file has duplicates)."""
    d = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip()
            if k in d:
                continue
            try:
                d[k] = float(v.strip())
            except ValueError:
                continue
    return d
