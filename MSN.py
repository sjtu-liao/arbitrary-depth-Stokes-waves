#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Public MSN training, evaluation, and inference script.

The script implements the three-stage multi-stage multi-scale neural network
described in the manuscript:
  1. Stage-1 multi-scale subnetworks for coefficient slices and K.
  2. A global residual refiner.
  3. Slice-wise final residual correction networks.

Input coefficient files are expected to follow:
  r0_<r0>_r_<r>_stp_<stp>_Iter_<...>_O_<...>_c0_<...>_a.txt

Each file should contain r Fourier coefficients followed by the Bernoulli
constant K. Header/comment lines starting with "#" are ignored.
"""

import argparse
import csv
import glob
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional.
    def tqdm(x, **kwargs):
        return x


MAX_OUTPUT = 50001
DEFAULT_SEED = 3407
EPS = 1e-12

SLICES = {
    "s1": (0, 10),
    "s2": (10, 100),
    "s3": (100, 500),
    "s4": (500, 2000),
    "s5": (2000, 10000),
    "s6": (10000, 25000),
    "s7": (25000, 50000),
    "B": (50000, 50001),
}

STAGE1_SLICES = {
    "Scale1_DNN": (0, 10),
    "Scale2_DNN": (10, 100),
    "Scale3_DNN": (100, 500),
    "Scale4_DNN": (500, 2000),
    "Scale5_DNN": (2000, 10000),
    "Scale6_DNN": (10000, 25000),
    "Scale7_DNN": (25000, 50000),
    "B_MLP": (50000, 50001),
}

FILENAME_RE = re.compile(
    r"r0_([0-9.]+)_r_(\d+)_stp_([0-9.eE+-]+)_Iter_(\d+)_O_(\d+)_c0_([0-9.eE+-]+)_a\.txt$"
)


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def ymax(x):
    """Maximum steepness proxy used to normalize the steepness input."""
    return (
        -0.511131251816334 * x**9
        + 2.09419123420497 * x**8
        - 3.40683252028638 * x**7
        + 2.98100380861159 * x**6
        - 1.86955315518757 * x**5
        + 0.985545826406913 * x**4
        - 0.0903307668736722 * x**3
        - 0.323863844860537 * x**2
        + 0.000113402087328375 * x
        + 0.141080131501302
    )


def parse_filename(path: str) -> Optional[Tuple[float, int, float, str]]:
    match = FILENAME_RE.search(os.path.basename(path))
    if not match:
        return None
    r0_str = match.group(1)
    return float(r0_str), int(match.group(2)), float(match.group(3)), r0_str


def read_coefficients(path: str) -> np.ndarray:
    values = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            for part in re.split(r"[\s,]+", text):
                if not part:
                    continue
                try:
                    values.append(float(part))
                except ValueError:
                    break
    if not values:
        raise ValueError(f"No numeric values found in {path}")
    return np.asarray(values, dtype=np.float64)


def load_dataset(data_dir: str, max_output: int = MAX_OUTPUT):
    pattern = os.path.join(data_dir, "r0_*_r_*_stp_*_Iter_*_O_*_c0_*_a.txt")
    files = sorted(glob.glob(pattern))
    xs, ys, rs, stps, gkeys, used = [], [], [], [], [], []

    for path in tqdm(files, desc="Loading coefficient files"):
        meta = parse_filename(path)
        if meta is None:
            continue
        r0, r, stp, r0_str = meta
        try:
            coeffs = read_coefficients(path)
        except Exception as exc:
            print(f"[skip] {path}: {exc}")
            continue
        if coeffs.size < r + 1:
            print(f"[skip] {path}: expected at least {r + 1} values, got {coeffs.size}")
            continue

        y = np.zeros(max_output, dtype=np.float32)
        y[:r] = coeffs[:r].astype(np.float32)
        y[-1] = np.float32(coeffs[r])
        xs.append([r0, stp / ymax(r0)])
        ys.append(y)
        rs.append(r)
        stps.append(stp)
        gkeys.append((r0_str, r))
        used.append(path)

    if not xs:
        raise FileNotFoundError(f"No usable coefficient files found in {data_dir}")

    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(rs, dtype=np.int64),
        np.asarray(stps, dtype=np.float64),
        gkeys,
        used,
    )


def grouped_val_split_by_r0_r(
    stp: np.ndarray,
    gkeys: List[Tuple[str, int]],
    val_ratio: float,
    remove_extrema: bool = True,
    seed: int = DEFAULT_SEED,
    min_val_per_group: int = 2,
):
    rng = np.random.RandomState(seed)
    groups = defaultdict(list)
    for idx, key in enumerate(gkeys):
        groups[key].append(idx)

    train_parts, val_parts = [], []
    for _, idxs in groups.items():
        idxs = np.asarray(idxs, dtype=int)
        order = np.argsort(stp[idxs], kind="mergesort")
        sorted_idxs = idxs[order]

        candidate_mask = np.ones(len(sorted_idxs), dtype=bool)
        if remove_extrema and len(sorted_idxs) >= 2:
            candidate_mask[0] = False
            candidate_mask[-1] = False
        candidates = sorted_idxs[candidate_mask]

        if len(candidates) == 0:
            chosen = np.asarray([], dtype=int)
        else:
            n_val = int(math.floor(val_ratio * len(candidates)))
            if len(candidates) >= min_val_per_group:
                n_val = max(n_val, min_val_per_group)
            n_val = min(n_val, len(candidates))
            chosen = rng.choice(candidates, size=n_val, replace=False) if n_val else np.asarray([], dtype=int)

        chosen_set = set(chosen.tolist())
        train_parts.append(np.asarray([i for i in sorted_idxs.tolist() if i not in chosen_set], dtype=int))
        if chosen.size:
            val_parts.append(chosen)

    train_idx = np.sort(np.concatenate(train_parts))
    val_idx = np.sort(np.concatenate(val_parts)) if val_parts else np.asarray([], dtype=int)
    if set(train_idx.tolist()) & set(val_idx.tolist()):
        raise RuntimeError("Train/validation split overlap detected.")
    return train_idx, val_idx


def load_or_create_split(
    split_file: str,
    stp: np.ndarray,
    gkeys: List[Tuple[str, int]],
    val_ratio: float,
    seed: int,
):
    if split_file and os.path.isfile(split_file):
        split = np.load(split_file)
        train_idx = split["train_idx"]
        if "val_idx" in split:
            val_idx = split["val_idx"]
        elif "test_idx" in split:
            val_idx = split["test_idx"]
        else:
            raise KeyError(f"{split_file} must contain val_idx or test_idx")
        return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)

    train_idx, val_idx = grouped_val_split_by_r0_r(stp, gkeys, val_ratio=val_ratio, seed=seed)
    if split_file:
        os.makedirs(os.path.dirname(split_file) or ".", exist_ok=True)
        np.savez(split_file, train_idx=train_idx, val_idx=val_idx, test_idx=val_idx)
    return train_idx, val_idx


def mse_np(a, b) -> float:
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


class StochasticDepth(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x, residual):
        if (not self.training) or self.p == 0.0:
            return x + residual
        keep = 1.0 - self.p
        mask = torch.empty(x.size(0), 1, device=x.device).bernoulli_(keep).div(keep)
        return x + residual * mask


class ResBlockPlus(nn.Module):
    def __init__(
        self,
        hidden: int,
        hidden_inner: Optional[int] = None,
        mlp_layers: int = 0,
        dropout: float = 0.05,
        activation=nn.ReLU,
        conv_channels: int = 0,
        conv_kernel: int = 3,
        conv_layers: int = 1,
        drop_path: float = 0.0,
    ):
        super().__init__()
        hidden_inner = hidden_inner or hidden
        act = activation()
        layers = [nn.Linear(hidden, hidden_inner), act]
        for _ in range(mlp_layers):
            layers += [nn.Linear(hidden_inner, hidden_inner), activation()]
        layers += [nn.Linear(hidden_inner, hidden), nn.Dropout(dropout)]
        self.mlp = nn.Sequential(*layers)

        self.use_conv = conv_channels > 0
        if self.use_conv:
            if hidden % conv_channels != 0:
                raise ValueError("hidden must be divisible by conv_channels")
            self.conv_channels = conv_channels
            self.length = hidden // conv_channels
            tower = []
            for _ in range(conv_layers):
                tower += [
                    nn.Conv1d(conv_channels, conv_channels, kernel_size=conv_kernel, padding=conv_kernel // 2, groups=conv_channels, bias=False),
                    nn.GELU(),
                    nn.Conv1d(conv_channels, conv_channels, kernel_size=1, bias=False),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            self.conv = nn.Sequential(*tower)
            self.mix_linear = nn.Linear(hidden, hidden)

        self.norm = nn.LayerNorm(hidden)
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0.0 else None

    def forward(self, x):
        residual = x
        out = self.mlp(x)
        if self.use_conv:
            bsz = x.size(0)
            xc = x.view(bsz, self.conv_channels, self.length)
            out = out + self.mix_linear(self.conv(xc).reshape(bsz, -1))
        out = self.drop_path(residual, out) if self.drop_path is not None else residual + out
        return self.norm(out)


class Stage1MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden, conv_channels, num_blocks, hidden_inner, dropout=0.0, activation=nn.ReLU):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_size, hidden), nn.LayerNorm(hidden), activation())
        self.blocks = nn.Sequential(
            *[
                ResBlockPlus(
                    hidden=hidden,
                    hidden_inner=hidden_inner,
                    mlp_layers=2,
                    dropout=dropout,
                    activation=activation,
                    conv_channels=conv_channels,
                    conv_kernel=3,
                    conv_layers=3,
                )
                for _ in range(num_blocks)
            ]
        )
        head_hidden = max(hidden // 2, 16)
        self.head = nn.Sequential(nn.Linear(hidden, head_hidden), nn.LayerNorm(head_hidden), activation(), nn.Dropout(dropout), nn.Linear(head_hidden, output_size))

    def forward(self, x):
        return self.head(self.blocks(self.input_proj(x)))


class DNN_Scale1(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Stage1MLP(2, 10, hidden=16, conv_channels=8, num_blocks=6, hidden_inner=32)

    def forward(self, x):
        return self.net(x)


class DNN_Scale2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Stage1MLP(2, 90, hidden=32, conv_channels=16, num_blocks=6, hidden_inner=64)

    def forward(self, x):
        return self.net(x)


class DNN_Scale3(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Stage1MLP(2, 400, hidden=64, conv_channels=32, num_blocks=8, hidden_inner=64)

    def forward(self, x):
        return self.net(x)


class DNN_Scale4(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Stage1MLP(2, 1500, hidden=128, conv_channels=64, num_blocks=8, hidden_inner=128)

    def forward(self, x):
        return self.net(x)


class DNN_Scale5(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Stage1MLP(2, 8000, hidden=256, conv_channels=64, num_blocks=10, hidden_inner=256)

    def forward(self, x):
        return self.net(x)


class DNN_Scale6(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Stage1MLP(2, 15000, hidden=256, conv_channels=64, num_blocks=10, hidden_inner=512)

    def forward(self, x):
        return self.net(x)


class DNN_Scale7(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Stage1MLP(2, 25000, hidden=512, conv_channels=128, num_blocks=12, hidden_inner=512)

    def forward(self, x):
        return self.net(x)


class MLP_B(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Stage1MLP(2, 1, hidden=16, conv_channels=4, num_blocks=9, hidden_inner=64, activation=nn.GELU)

    def forward(self, x):
        return self.net(x)


def build_stage1_models() -> Dict[str, nn.Module]:
    return {
        "Scale1_DNN": DNN_Scale1(),
        "Scale2_DNN": DNN_Scale2(),
        "Scale3_DNN": DNN_Scale3(),
        "Scale4_DNN": DNN_Scale4(),
        "Scale5_DNN": DNN_Scale5(),
        "Scale6_DNN": DNN_Scale6(),
        "Scale7_DNN": DNN_Scale7(),
        "B_MLP": MLP_B(),
    }


class InputProj(nn.Module):
    def __init__(self, input_size, hidden, activation=nn.GELU, use_sine=False, sine_gamma=1.0):
        super().__init__()
        self.lin = nn.Linear(input_size, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.use_sine = use_sine
        self.sine_gamma = float(sine_gamma)
        self.act = activation()

    def forward(self, x):
        z = self.norm(self.lin(x))
        return torch.sin(self.sine_gamma * z) if self.use_sine else self.act(z)


class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden,
        conv_channels=32,
        num_blocks=12,
        hidden_inner=None,
        dropout=0.015,
        activation=nn.GELU,
        use_sine_first_layer=False,
        sine_gamma=1.0,
    ):
        super().__init__()
        self.input_proj = InputProj(input_size, hidden, activation=activation, use_sine=use_sine_first_layer, sine_gamma=sine_gamma)
        self.blocks = nn.Sequential(
            *[
                ResBlockPlus(hidden=hidden, hidden_inner=hidden_inner or hidden, mlp_layers=2, dropout=dropout, activation=activation, conv_channels=conv_channels, conv_kernel=3, conv_layers=3)
                for _ in range(num_blocks)
            ]
        )
        head_hidden = max(hidden // 2, 16)
        self.head = nn.Sequential(nn.Linear(hidden, head_hidden), nn.LayerNorm(head_hidden), activation(), nn.Dropout(dropout), nn.Linear(head_hidden, output_size))

    def forward(self, x):
        return self.head(self.blocks(self.input_proj(x)))


class BigRefiner(nn.Module):
    def __init__(self, d_in, d_out, rank=256, n_blocks=6, dropout=0.05, include_layernorm=True):
        super().__init__()
        self.norm = nn.LayerNorm(d_in) if include_layernorm else nn.Identity()
        self.down = nn.Linear(d_in, rank)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(rank, rank * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(rank * 4, rank),
                    nn.Dropout(dropout),
                    nn.LayerNorm(rank),
                )
                for _ in range(n_blocks)
            ]
        )
        self.up = nn.Linear(rank, d_out)

    def forward(self, z):
        h = self.down(self.norm(z))
        for block in self.blocks:
            h = h + block(h)
        return self.up(h)


S2_CFG = {
    "s1": {"hidden": 32, "hidden_inner": 32, "num_blocks": 6, "conv_channels": 16, "dropout": 0.00, "use_sine_first_layer": True, "sine_gamma": 8.0},
    "s2": {"hidden": 64, "hidden_inner": 64, "num_blocks": 6, "conv_channels": 32, "dropout": 0.01, "use_sine_first_layer": True, "sine_gamma": 20.0},
    "s3": {"hidden": 128, "hidden_inner": 128, "num_blocks": 8, "conv_channels": 64, "dropout": 0.00, "use_sine_first_layer": True, "sine_gamma": 25.0},
    "s4": {"hidden": 128, "hidden_inner": 256, "num_blocks": 8, "conv_channels": 64, "dropout": 0.00, "use_sine_first_layer": True, "sine_gamma": 30.0},
    "s5": {"hidden": 256, "hidden_inner": 256, "num_blocks": 8, "conv_channels": 128, "dropout": 0.00, "use_sine_first_layer": True, "sine_gamma": 35.0},
    "s6": {"hidden": 256, "hidden_inner": 512, "num_blocks": 10, "conv_channels": 128, "dropout": 0.00, "use_sine_first_layer": True, "sine_gamma": 40.0},
    "s7": {"hidden": 512, "hidden_inner": 512, "num_blocks": 10, "conv_channels": 256, "dropout": 0.00, "use_sine_first_layer": True, "sine_gamma": 45.0},
    "B": {"hidden": 64, "hidden_inner": 128, "num_blocks": 6, "conv_channels": 16, "dropout": 0.00, "use_sine_first_layer": True, "sine_gamma": 5.0},
}


def mk_s2_model(tag: str, out_dim: int) -> nn.Module:
    cfg = S2_CFG[tag]
    return ResidualMLP(
        input_size=2,
        output_size=out_dim,
        hidden=cfg["hidden"],
        hidden_inner=cfg["hidden_inner"],
        num_blocks=cfg["num_blocks"],
        conv_channels=cfg["conv_channels"],
        dropout=cfg["dropout"],
        activation=nn.GELU,
        use_sine_first_layer=cfg["use_sine_first_layer"],
        sine_gamma=cfg["sine_gamma"],
    )


def remap_input_proj_keys(state_dict: dict) -> dict:
    out = {}
    for key, value in state_dict.items():
        if key.startswith("net.input_proj.0."):
            out[key.replace("net.input_proj.0.", "net.input_proj.lin.")] = value
        elif key.startswith("net.input_proj.1."):
            out[key.replace("net.input_proj.1.", "net.input_proj.norm.")] = value
        else:
            out[key] = value
    return out


def load_state_dict_compat(model: nn.Module, state):
    state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        model.load_state_dict(remap_input_proj_keys(state_dict), strict=False)


def find_best_ckpt(pattern: str) -> Optional[str]:
    files = glob.glob(pattern)
    if not files:
        return None

    def score(path):
        match = re.search(r"loss_([0-9.]+e[-+]?\d+|[0-9.]+)\.pth$", os.path.basename(path))
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return float("inf")

    files.sort(key=lambda p: (score(p), -os.path.getmtime(p)))
    if math.isinf(score(files[0])):
        files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def save_stage1_ckpt(model: nn.Module, out_dir: str, name: str, epoch: int, val_loss: float) -> str:
    os.makedirs(out_dir, exist_ok=True)
    for old in glob.glob(os.path.join(out_dir, f"best_{name}_*.pth")):
        os.remove(old)
    path = os.path.join(out_dir, f"best_{name}_epoch_{epoch}_loss_{val_loss:.6e}.pth")
    torch.save(model.state_dict(), path)
    return path


def load_stage1_models(stage1_dir: str, device: torch.device) -> Dict[str, nn.Module]:
    models = build_stage1_models()
    for name, model in models.items():
        path = find_best_ckpt(os.path.join(stage1_dir, f"best_{name}_*.pth"))
        if path is None:
            raise FileNotFoundError(f"Missing Stage-1 checkpoint for {name} in {stage1_dir}")
        state = torch.load(path, map_location="cpu")
        load_state_dict_compat(model, state)
        model.to(device).eval()
        print(f"[load] {name}: {path}")
    return models


@torch.no_grad()
def predict_stage1(models: Dict[str, nn.Module], x: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    outputs = []
    for start in range(0, len(x), batch_size):
        xb = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
        yb = torch.cat(
            [
                models["Scale1_DNN"](xb),
                models["Scale2_DNN"](xb),
                models["Scale3_DNN"](xb),
                models["Scale4_DNN"](xb),
                models["Scale5_DNN"](xb),
                models["Scale6_DNN"](xb),
                models["Scale7_DNN"](xb),
                models["B_MLP"](xb),
            ],
            dim=1,
        )
        outputs.append(yb.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


@torch.no_grad()
def predict_refiner(refiner: nn.Module, include_x: bool, y1: np.ndarray, x: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    parts = []
    for start in range(0, len(x), batch_size):
        yb = torch.as_tensor(y1[start : start + batch_size], dtype=torch.float32, device=device)
        xb = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
        z = torch.cat([yb, xb], dim=1) if include_x else yb
        parts.append(refiner(z).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(parts, axis=0)


def combine_s1_gr(y1: np.ndarray, p: np.ndarray, gr_bundle: dict) -> np.ndarray:
    alpha0 = float(gr_bundle["alpha0"])
    per_slice = gr_bundle.get("slice_beta_lambda") or {tag: {"beta": 1.0, "lambda": 1.0} for tag in SLICES}
    ygr = y1.copy()
    for tag, (left, right) in SLICES.items():
        beta = float(per_slice[tag]["beta"])
        lam = float(per_slice[tag]["lambda"])
        ygr[:, left:right] = y1[:, left:right] + alpha0 * (beta * lam) * p[:, left:right]
    return ygr


def load_global_refiner(gr_path: str, device: torch.device):
    bundle = torch.load(gr_path, map_location="cpu")
    include_x = bool(bundle.get("include_x", True))
    rank = int(bundle.get("rank", 256))
    n_blocks = int(bundle.get("n_blocks", 6))
    model = BigRefiner(MAX_OUTPUT + (2 if include_x else 0), MAX_OUTPUT, rank=rank, n_blocks=n_blocks).to(device).eval()
    model.load_state_dict(bundle["state_dict"])
    return bundle, model


def load_s2_models(s2_dir: str, device: torch.device):
    models = {}
    if not os.path.isdir(s2_dir):
        return models
    for tag, (left, right) in SLICES.items():
        path = os.path.join(s2_dir, f"best_s2_{tag}.pth")
        if not os.path.isfile(path):
            continue
        bundle = torch.load(path, map_location="cpu")
        model = mk_s2_model(tag, right - left).to(device).eval()
        model.load_state_dict(bundle["state_dict"])
        models[tag] = {
            "model": model,
            "alpha": float(bundle.get("alpha", 1.0)),
            "beta": float(bundle.get("beta", 1.0)),
            "lambda": float(bundle.get("lambda", 1.0)),
            "scale_vec": np.asarray(bundle["scale_vec"], dtype=np.float32) if bundle.get("scale_vec") is not None else None,
        }
    return models


@torch.no_grad()
def apply_s2(x: np.ndarray, ygr: np.ndarray, s2_models: dict, device: torch.device, batch_size: int = 512) -> np.ndarray:
    yout = ygr.copy()
    for tag, (left, right) in SLICES.items():
        if tag not in s2_models:
            continue
        entry = s2_models[tag]
        pred_parts = []
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            pred_parts.append(entry["model"](xb).detach().cpu().numpy().astype(np.float32))
        pred = np.concatenate(pred_parts, axis=0)
        if entry["scale_vec"] is not None:
            pred = pred * entry["scale_vec"].reshape(1, -1)
        yout[:, left:right] += entry["alpha"] * (entry["beta"] * entry["lambda"]) * pred
    return yout


def train_stage1_single(model, name, x_train, y_train, x_val, y_val, stage1_dir, device, epochs, batch_size, lr, weight_decay, lr_tmax):
    model = model.to(device)
    x_train_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    y_val_t = torch.as_tensor(y_val, dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=lr_tmax, eta_min=0.0)
    best = float("inf")

    for epoch in tqdm(range(1, epochs + 1), desc=f"Stage-1 {name}"):
        model.train()
        order = torch.randperm(x_train_t.size(0), device=device)
        total, count = 0.0, 0
        for start in range(0, x_train_t.size(0), batch_size):
            idx = order[start : start + batch_size]
            pred = model(x_train_t[idx])
            loss = F.mse_loss(pred, y_train_t[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            count += 1

        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(F.mse_loss(model(x_val_t), y_val_t).item())
        if val_loss < best:
            best = val_loss
            save_stage1_ckpt(model, stage1_dir, name, epoch, val_loss)
        if epoch == 1 or epoch % max(1, epochs // 10) == 0:
            print(f"[Stage-1:{name}] epoch={epoch} train={total / max(count, 1):.4e} val_best={best:.4e}")
    return best


def compute_alpha0(y: np.ndarray, y1: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - y1) ** 2) + EPS))


def train_global_refiner(args, x_train, y_train, y1_train, x_val, y_val, y1_val, device):
    os.makedirs(args.gr_dir, exist_ok=True)
    alpha0 = compute_alpha0(y_train, y1_train)
    refiner = BigRefiner(MAX_OUTPUT + (2 if args.gr_include_x else 0), MAX_OUTPUT, rank=args.gr_rank, n_blocks=args.gr_blocks).to(device)
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=args.gr_lr, weight_decay=args.gr_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.gr_epochs, eta_min=1e-8)
    best = float("inf")
    best_path = os.path.join(args.gr_dir, "best_global_refiner.pth")

    for epoch in tqdm(range(1, args.gr_epochs + 1), desc="Global refiner"):
        refiner.train()
        order = np.random.permutation(len(x_train))
        total, count = 0.0, 0
        for start in range(0, len(x_train), args.gr_batch):
            idx = order[start : start + args.gr_batch]
            xb = torch.as_tensor(x_train[idx], dtype=torch.float32, device=device)
            yb = torch.as_tensor(y_train[idx], dtype=torch.float32, device=device)
            y1b = torch.as_tensor(y1_train[idx], dtype=torch.float32, device=device)
            z = torch.cat([y1b, xb], dim=1) if args.gr_include_x else y1b
            target = (yb - y1b) / max(alpha0, EPS)
            pred = refiner(z)
            loss = F.mse_loss(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            count += 1
        scheduler.step()

        if epoch % args.gr_val_interval == 0 or epoch == args.gr_epochs:
            refiner.eval()
            losses = []
            with torch.no_grad():
                for start in range(0, len(x_val), args.gr_batch):
                    xb = torch.as_tensor(x_val[start : start + args.gr_batch], dtype=torch.float32, device=device)
                    yb = torch.as_tensor(y_val[start : start + args.gr_batch], dtype=torch.float32, device=device)
                    y1b = torch.as_tensor(y1_val[start : start + args.gr_batch], dtype=torch.float32, device=device)
                    z = torch.cat([y1b, xb], dim=1) if args.gr_include_x else y1b
                    target = (yb - y1b) / max(alpha0, EPS)
                    losses.append(float(F.mse_loss(refiner(z), target).item()))
            val_loss = float(np.mean(losses))
            if val_loss < best:
                p_val = predict_refiner(refiner, args.gr_include_x, y1_val, x_val, device)
                per_slice = {}
                for tag, (left, right) in SLICES.items():
                    ps = torch.as_tensor(p_val[:, left:right])
                    rs = torch.as_tensor((y_val[:, left:right] - y1_val[:, left:right]) / max(alpha0, EPS))
                    beta = float((ps * rs).mean().item() / ((ps * ps).mean().item() + EPS))
                    best_mse, best_lam = float("inf"), 1.0
                    for lam in args.lambda_grid:
                        candidate = y1_val[:, left:right] + alpha0 * (beta * lam) * p_val[:, left:right]
                        candidate_mse = mse_np(candidate, y_val[:, left:right])
                        if candidate_mse < best_mse:
                            best_mse, best_lam = candidate_mse, lam
                    per_slice[tag] = {"beta": beta, "lambda": float(best_lam)}
                best = val_loss
                torch.save(
                    {
                        "state_dict": refiner.state_dict(),
                        "alpha0": alpha0,
                        "include_x": bool(args.gr_include_x),
                        "rank": int(args.gr_rank),
                        "n_blocks": int(args.gr_blocks),
                        "slice_beta_lambda": per_slice,
                        "val_loss": best,
                    },
                    best_path,
                )
                print(f"[GR] saved {best_path} val={best:.4e}")
    return best_path


def build_mask_for_slice(r_values: np.ndarray, left: int, right: int, tag: str):
    if tag == "B":
        return np.ones((len(r_values), right - left), dtype=np.float32)
    idx = np.arange(right - left)[None, :]
    return (idx + left < r_values[:, None]).astype(np.float32)


def fit_diag_scale_vec(pred: np.ndarray, target: np.ndarray):
    numerator = np.sum(pred * target, axis=0)
    denominator = np.sum(pred * pred, axis=0) + EPS
    return (numerator / denominator).astype(np.float32)


def train_s2_slice(args, tag, x_train, y_train, r_train, x_val, y_val, r_val, y1_train, y1_val, p_train, p_val, gr_bundle, device):
    left, right = SLICES[tag]
    alpha0 = float(gr_bundle["alpha0"])
    per_slice = gr_bundle.get("slice_beta_lambda") or {name: {"beta": 1.0, "lambda": 1.0} for name in SLICES}
    beta = float(per_slice[tag]["beta"])
    lam = float(per_slice[tag]["lambda"])

    ygr_train = y1_train[:, left:right] + alpha0 * (beta * lam) * p_train[:, left:right]
    ygr_val = y1_val[:, left:right] + alpha0 * (beta * lam) * p_val[:, left:right]
    residual_train = (y_train[:, left:right] - ygr_train).astype(np.float32)
    residual_val = (y_val[:, left:right] - ygr_val).astype(np.float32)
    alpha = float(np.sqrt(np.mean(residual_train**2) + EPS)) or 1.0
    target_train = residual_train / alpha
    target_val = residual_val / alpha
    w_train = build_mask_for_slice(r_train, left, right, tag) * args.s2_active_weight
    w_train += (1.0 - build_mask_for_slice(r_train, left, right, tag)) * args.s2_inactive_weight
    w_val = build_mask_for_slice(r_val, left, right, tag) * args.s2_active_weight
    w_val += (1.0 - build_mask_for_slice(r_val, left, right, tag)) * args.s2_inactive_weight

    model = mk_s2_model(tag, right - left).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.s2_lr, weight_decay=args.s2_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.s2_epochs, eta_min=1e-8)
    best = float("inf")
    best_path = os.path.join(args.s2_dir, f"best_s2_{tag}.pth")
    os.makedirs(args.s2_dir, exist_ok=True)

    for epoch in tqdm(range(1, args.s2_epochs + 1), desc=f"Stage-2 {tag}"):
        model.train()
        order = np.random.permutation(len(x_train))
        for start in range(0, len(x_train), args.s2_batch):
            idx = order[start : start + args.s2_batch]
            xb = torch.as_tensor(x_train[idx], dtype=torch.float32, device=device)
            tb = torch.as_tensor(target_train[idx], dtype=torch.float32, device=device)
            wb = torch.as_tensor(w_train[idx], dtype=torch.float32, device=device)
            pred = model(xb)
            loss = (((pred - tb) ** 2) * wb).sum() / wb.sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % args.s2_val_interval == 0 or epoch == args.s2_epochs:
            model.eval()
            losses = []
            with torch.no_grad():
                for start in range(0, len(x_val), args.s2_batch):
                    xb = torch.as_tensor(x_val[start : start + args.s2_batch], dtype=torch.float32, device=device)
                    tb = torch.as_tensor(target_val[start : start + args.s2_batch], dtype=torch.float32, device=device)
                    wb = torch.as_tensor(w_val[start : start + args.s2_batch], dtype=torch.float32, device=device)
                    pred = model(xb)
                    losses.append(float(((((pred - tb) ** 2) * wb).sum() / wb.sum().clamp_min(1.0)).item()))
            val_loss = float(np.mean(losses))
            if val_loss < best:
                pred_parts = []
                with torch.no_grad():
                    for start in range(0, len(x_val), 512):
                        xb = torch.as_tensor(x_val[start : start + 512], dtype=torch.float32, device=device)
                        pred_parts.append(model(xb).detach().cpu().numpy().astype(np.float32))
                pred_val = np.concatenate(pred_parts, axis=0)
                scale_vec = fit_diag_scale_vec(pred_val, target_val)
                best = val_loss
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "alpha": alpha,
                        "beta": 1.0,
                        "lambda": 1.0,
                        "scale_vec": scale_vec,
                        "val_loss": best,
                        "slice": [left, right],
                        "tag": tag,
                    },
                    best_path,
                )
                print(f"[S2:{tag}] saved {best_path} val={best:.4e}")


def write_file_list(path: str, files: Iterable[str], indices: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for idx in indices:
            handle.write(str(files[int(idx)]) + "\n")


def command_train(args):
    set_seed(args.seed)
    device = get_device(args.device)
    args.stage1_dir = args.stage1_dir or os.path.join(args.out_dir, "stage1")
    args.gr_dir = args.gr_dir or os.path.join(args.out_dir, "global_refiner")
    args.s2_dir = args.s2_dir or os.path.join(args.out_dir, "stage2")
    args.cache_dir = args.cache_dir or os.path.join(args.out_dir, "cache")
    args.split_file = args.split_file or os.path.join(args.out_dir, "split_idx.npz")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    x, y, r_values, stp, gkeys, files = load_dataset(args.data_dir, args.max_output)
    train_idx, val_idx = load_or_create_split(args.split_file, stp, gkeys, args.val_ratio, args.seed)
    write_file_list(os.path.join(args.out_dir, "train_files.txt"), files, train_idx)
    write_file_list(os.path.join(args.out_dir, "val_files.txt"), files, val_idx)

    x_train, y_train, r_train = x[train_idx], y[train_idx], r_values[train_idx]
    x_val, y_val, r_val = x[val_idx], y[val_idx], r_values[val_idx]
    print(f"[data] train={len(x_train)} val={len(x_val)} device={device}")

    if args.train_stage1:
        for name, model in build_stage1_models().items():
            left, right = STAGE1_SLICES[name]
            train_stage1_single(
                model,
                name,
                x_train,
                y_train[:, left:right],
                x_val,
                y_val[:, left:right],
                args.stage1_dir,
                device,
                args.s1_epochs,
                args.s1_batch,
                args.s1_lr,
                args.s1_weight_decay,
                args.s1_lr_tmax,
            )
    else:
        print("[Stage-1] skipped; existing checkpoints will be loaded.")

    s1_models = load_stage1_models(args.stage1_dir, device)
    y1_train_path = os.path.join(args.cache_dir, "Y1_train.npy")
    y1_val_path = os.path.join(args.cache_dir, "Y1_val.npy")
    if os.path.isfile(y1_train_path) and os.path.isfile(y1_val_path) and not args.rebuild_cache:
        y1_train = np.load(y1_train_path).astype(np.float32)
        y1_val = np.load(y1_val_path).astype(np.float32)
    else:
        y1_train = predict_stage1(s1_models, x_train, device)
        y1_val = predict_stage1(s1_models, x_val, device)
        np.save(y1_train_path, y1_train)
        np.save(y1_val_path, y1_val)

    if args.train_gr:
        gr_path = train_global_refiner(args, x_train, y_train, y1_train, x_val, y_val, y1_val, device)
    else:
        gr_path = os.path.join(args.gr_dir, "best_global_refiner.pth")
        print(f"[GR] skipped; loading {gr_path}")

    gr_bundle, gr_model = load_global_refiner(gr_path, device)
    p_train_path = os.path.join(args.cache_dir, "P_train.npy")
    p_val_path = os.path.join(args.cache_dir, "P_val.npy")
    include_x = bool(gr_bundle.get("include_x", True))
    if os.path.isfile(p_train_path) and os.path.isfile(p_val_path) and not args.rebuild_cache:
        p_train = np.load(p_train_path).astype(np.float32)
        p_val = np.load(p_val_path).astype(np.float32)
    else:
        p_train = predict_refiner(gr_model, include_x, y1_train, x_train, device)
        p_val = predict_refiner(gr_model, include_x, y1_val, x_val, device)
        np.save(p_train_path, p_train)
        np.save(p_val_path, p_val)

    if args.train_stage2:
        for tag in SLICES:
            train_s2_slice(args, tag, x_train, y_train, r_train, x_val, y_val, r_val, y1_train, y1_val, p_train, p_val, gr_bundle, device)
    else:
        print("[Stage-2] skipped.")

    ygr_val = combine_s1_gr(y1_val, p_val, gr_bundle)
    yfinal_val = apply_s2(x_val, ygr_val, load_s2_models(args.s2_dir, device), device)
    print("[validation]")
    print(f"  Stage-1      MSE = {mse_np(y1_val, y_val):.6e}")
    print(f"  Stage-1+GR   MSE = {mse_np(ygr_val, y_val):.6e}")
    print(f"  Full MSN     MSE = {mse_np(yfinal_val, y_val):.6e}")

    manifest = {
        "script": "MSN.py",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": args.data_dir,
        "out_dir": args.out_dir,
        "train_samples": int(len(x_train)),
        "val_samples": int(len(x_val)),
        "max_output": int(args.max_output),
        "seed": int(args.seed),
        "device": str(device),
        "stage1_dir": args.stage1_dir,
        "global_refiner": gr_path,
        "stage2_dir": args.s2_dir,
    }
    with open(os.path.join(args.out_dir, "run_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def load_public_msn(ckpt_root: str, device: torch.device):
    stage1_dir = os.path.join(ckpt_root, "stage1")
    gr_path = os.path.join(ckpt_root, "global_refiner", "best_global_refiner.pth")
    s2_dir = os.path.join(ckpt_root, "stage2")
    s1_models = load_stage1_models(stage1_dir, device)
    gr_bundle, gr_model = load_global_refiner(gr_path, device)
    s2_models = load_s2_models(s2_dir, device)
    return s1_models, gr_bundle, gr_model, s2_models


def predict_msn_coefficients(r0: np.ndarray, stp: np.ndarray, models, device: torch.device) -> np.ndarray:
    s1_models, gr_bundle, gr_model, s2_models = models
    x = np.column_stack([r0, stp / ymax(r0)]).astype(np.float32)
    y1 = predict_stage1(s1_models, x, device)
    p = predict_refiner(gr_model, bool(gr_bundle.get("include_x", True)), y1, x, device)
    ygr = combine_s1_gr(y1, p, gr_bundle)
    return apply_s2(x, ygr, s2_models, device)


def evaluate_dataset(args):
    set_seed(args.seed)
    device = get_device(args.device)
    x, y, r_values, stp, gkeys, files = load_dataset(args.data_dir, args.max_output)
    if args.split_file:
        split_file = args.split_file
    else:
        trained_split = os.path.join(args.ckpt_root, "split_idx.npz")
        split_file = trained_split if os.path.isfile(trained_split) else os.path.join(args.out_dir, "split_idx.npz")
    _, test_idx = load_or_create_split(split_file, stp, gkeys, args.test_ratio, args.seed)
    x_test, y_test = x[test_idx], y[test_idx]

    models = load_public_msn(args.ckpt_root, device)
    y1 = predict_stage1(models[0], x_test, device)
    p = predict_refiner(models[2], bool(models[1].get("include_x", True)), y1, x_test, device)
    ygr = combine_s1_gr(y1, p, models[1])
    yfull = apply_s2(x_test, ygr, models[3], device)

    os.makedirs(args.out_dir, exist_ok=True)
    summary_rows = []
    for name, pred in [("stage1", y1), ("stage1_gr", ygr), ("full_msn", yfull)]:
        row = {"model": name, "overall_mse": mse_np(pred, y_test)}
        for tag, (left, right) in SLICES.items():
            row[f"{tag}_mse"] = mse_np(pred[:, left:right], y_test[:, left:right])
        summary_rows.append(row)
    with open(os.path.join(args.out_dir, "msn_test_summary.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    sample_i = int(args.sample_index) if args.sample_index >= 0 else random.randrange(len(x_test))
    compare = np.column_stack(
        [
            np.arange(y_test.shape[1], dtype=np.int32),
            y_test[sample_i],
            y1[sample_i],
            ygr[sample_i],
            yfull[sample_i],
            yfull[sample_i] - y_test[sample_i],
            np.abs(yfull[sample_i] - y_test[sample_i]),
        ]
    )
    np.savetxt(
        os.path.join(args.out_dir, f"sample_{sample_i:05d}_compare.csv"),
        compare,
        delimiter=",",
        header="dim,gt,stage1,stage1_gr,full_msn,full_error,full_abs_error",
        comments="",
        fmt=["%d", "%.10e", "%.10e", "%.10e", "%.10e", "%.10e", "%.10e"],
    )
    print(f"[test] overall MSE: Stage-1={summary_rows[0]['overall_mse']:.6e}, Stage-1+GR={summary_rows[1]['overall_mse']:.6e}, Full={summary_rows[2]['overall_mse']:.6e}")


def predict_directory(args):
    device = get_device(args.device)
    models = load_public_msn(args.ckpt_root, device)
    files = sorted(glob.glob(os.path.join(args.input_dir, "r0_*_r_*_stp_*_Iter_*_O_*_c0_*_a.txt")))
    if not files:
        raise FileNotFoundError(f"No coefficient files found in {args.input_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    metas, r0_values, stp_values = [], [], []
    for path in files:
        meta = parse_filename(path)
        if meta is None:
            continue
        r0, r, stp, _ = meta
        metas.append((path, r))
        r0_values.append(r0)
        stp_values.append(stp)

    pred = predict_msn_coefficients(np.asarray(r0_values), np.asarray(stp_values), models, device)
    for i, (path, r) in enumerate(metas):
        vec = np.concatenate([pred[i, :r], [pred[i, -1]]])
        out_path = os.path.join(args.out_dir, args.pred_prefix + os.path.basename(path))
        np.savetxt(out_path, vec.reshape(-1, 1), fmt="%.10e")
    print(f"[predict] wrote {len(metas)} files to {args.out_dir}")


def add_train_args(sub):
    sub.add_argument("--data-dir", required=True)
    sub.add_argument("--out-dir", default="runs/msn")
    sub.add_argument("--stage1-dir", default=None)
    sub.add_argument("--gr-dir", default=None)
    sub.add_argument("--s2-dir", default=None)
    sub.add_argument("--cache-dir", default=None)
    sub.add_argument("--split-file", default=None)
    sub.add_argument("--max-output", type=int, default=MAX_OUTPUT)
    sub.add_argument("--val-ratio", type=float, default=0.18)
    sub.add_argument("--seed", type=int, default=DEFAULT_SEED)
    sub.add_argument("--device", default="auto")
    sub.add_argument("--rebuild-cache", action="store_true")
    sub.add_argument("--train-stage1", action=argparse.BooleanOptionalAction, default=True)
    sub.add_argument("--train-gr", action=argparse.BooleanOptionalAction, default=True)
    sub.add_argument("--train-stage2", action=argparse.BooleanOptionalAction, default=True)
    sub.add_argument("--s1-epochs", type=int, default=3000)
    sub.add_argument("--s1-batch", type=int, default=4)
    sub.add_argument("--s1-lr", type=float, default=1e-3)
    sub.add_argument("--s1-weight-decay", type=float, default=1e-5)
    sub.add_argument("--s1-lr-tmax", type=int, default=3000)
    sub.add_argument("--gr-include-x", action=argparse.BooleanOptionalAction, default=True)
    sub.add_argument("--gr-rank", type=int, default=256)
    sub.add_argument("--gr-blocks", type=int, default=6)
    sub.add_argument("--gr-epochs", type=int, default=2000)
    sub.add_argument("--gr-batch", type=int, default=16)
    sub.add_argument("--gr-lr", type=float, default=1e-3)
    sub.add_argument("--gr-weight-decay", type=float, default=1e-5)
    sub.add_argument("--gr-val-interval", type=int, default=1)
    sub.add_argument("--lambda-grid", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
    sub.add_argument("--s2-epochs", type=int, default=2000)
    sub.add_argument("--s2-batch", type=int, default=16)
    sub.add_argument("--s2-lr", type=float, default=1e-3)
    sub.add_argument("--s2-weight-decay", type=float, default=1e-5)
    sub.add_argument("--s2-val-interval", type=int, default=1)
    sub.add_argument("--s2-active-weight", type=float, default=1.0)
    sub.add_argument("--s2-inactive-weight", type=float, default=0.2)


def main():
    parser = argparse.ArgumentParser(description="MSN train/test/predict entry point.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the three-stage MSN.")
    add_train_args(train_parser)
    train_parser.set_defaults(func=command_train)

    test_parser = subparsers.add_parser("test", help="Evaluate a saved MSN checkpoint root.")
    test_parser.add_argument("--data-dir", required=True)
    test_parser.add_argument("--ckpt-root", required=True)
    test_parser.add_argument("--out-dir", default="eval_msn")
    test_parser.add_argument("--split-file", default=None)
    test_parser.add_argument("--test-ratio", type=float, default=0.2)
    test_parser.add_argument("--sample-index", type=int, default=-1)
    test_parser.add_argument("--max-output", type=int, default=MAX_OUTPUT)
    test_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    test_parser.add_argument("--device", default="auto")
    test_parser.set_defaults(func=evaluate_dataset)

    pred_parser = subparsers.add_parser("predict", help="Load pth files and export predicted coefficient files.")
    pred_parser.add_argument("--input-dir", required=True)
    pred_parser.add_argument("--ckpt-root", required=True)
    pred_parser.add_argument("--out-dir", default="pred_msn")
    pred_parser.add_argument("--pred-prefix", default="pred_")
    pred_parser.add_argument("--device", default="auto")
    pred_parser.set_defaults(func=predict_directory)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
