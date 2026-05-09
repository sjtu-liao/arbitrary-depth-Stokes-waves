#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Public FNO training and testing script for inverse conformal mapping.

The model maps physical-plane fields and wave parameters to conformal
coordinates:
  input  : (x_norm, y_norm, r0, steepness_channel)
  output : (theta, R)

The script is self-contained. It includes the coefficient reader and the
minimal wave-processing routines needed to generate supervised FNO data and
to reconstruct velocity-field errors during testing.
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
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


DEFAULT_SEED = 3407
NPZ_VERSION = "public-fno-grid-v1"

H_R_DEFAULT = 100
W_TH_DEFAULT = 101
MODES_TH_DEFAULT = 48
MODES_R_DEFAULT = 48
WIDTH_DEFAULT = 128
LAYERS_DEFAULT = 6

FOCUS_TH_SIGMA = 0.05 * math.pi
FOCUS_R_SIGMA = 0.05
FOCUS_GAIN = 20.0
PIN_GAIN_THETA = 10.0
PIN_GAIN_R = 10.0
RANGE_PEN_W = 1e-3
XY_J_BLOCK = 50
XY_MID_F32 = True

FILENAME_RE = re.compile(r"r0_([0-9.]+)_r_(\d+)_stp_([0-9.eE+-]+).*_a\.txt$")


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


def parse_filename(path: str):
    match = FILENAME_RE.match(os.path.basename(path))
    if not match:
        raise ValueError(f"Cannot parse r0, r, stp from filename: {os.path.basename(path)}")
    return float(match.group(1)), int(match.group(2)), float(match.group(3))


def parameter_channel_value(r0: float, stp: float, use_normalized_stp: bool) -> float:
    return float(stp / ymax(r0)) if use_normalized_stp else float(stp)


class WaveProcessor:
    """Minimal Stokes-wave coefficient processor used by the FNO pipeline."""

    def __init__(self, theta_num: int = 81, r_num: int = 10):
        self.r = None
        self.r0 = None
        self.theta_num = theta_num
        self.r_num = r_num
        self.r_begin = None
        self.a = np.empty((0,), dtype=np.float64)
        self.c = 0.0
        self._sigma = None
        self._delta = None
        self._A = None
        self._K = None
        self._f_all = None
        self._cache_valid = False

    @staticmethod
    def _fft_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        n = len(a) + len(b) - 1
        nfft = 1 << (n - 1).bit_length()
        return np.fft.irfft(np.fft.rfft(a, nfft) * np.fft.rfft(b, nfft), nfft)[:n]

    @staticmethod
    def _pos_lag_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = len(x)
        conv = WaveProcessor._fft_convolve(x[::-1], y)
        return conv[n - 1 : n - 1 + n]

    def _clear_cache(self):
        self._sigma = None
        self._delta = None
        self._A = None
        self._K = None
        self._f_all = None
        self._cache_valid = False

    @staticmethod
    def _read_numbers_skip_comments(path: str) -> np.ndarray:
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

    def load_coefficients(self, path: str) -> None:
        r0, r, _ = parse_filename(path)
        self.r = int(r)
        self.r0 = float(r0)
        self.r_begin = 1e-8 if self.r0 <= 0.0 else self.r0
        self._clear_cache()

        arr = self._read_numbers_skip_comments(path)
        if len(arr) < self.r + 1:
            raise ValueError(f"Expected at least r+1={self.r + 1} values, got {len(arr)}: {path}")
        self.a = arr
        self._A = self.a[: self.r].copy()
        self._K = float(self.a[self.r])

        j = np.arange(self.r + 1, dtype=np.float64)
        if self.r0 <= 0.0:
            r0_pow_2j = np.zeros_like(j)
        else:
            r0_pow_2j = np.exp(np.minimum(0.0, (2.0 * j) * np.log(self.r0)))
        self._sigma = 1.0 + r0_pow_2j
        self._delta = 1.0 - r0_pow_2j
        self._precompute_f_all()
        self._cache_valid = True

    def _precompute_f_all(self):
        a = self._A.astype(np.float64)
        r = int(self.r)
        r0 = float(self.r0)
        f = np.zeros(r + 1, dtype=np.float64)

        ell = np.arange(1, r + 1, dtype=np.float64)
        sigma_2l = np.ones_like(ell) if r0 <= 0.0 else 1.0 + np.exp((4.0 * ell) * np.log(r0))
        f[0] = 1.0 + np.sum((a**2) * sigma_2l)

        a_pad = np.zeros(r + 1, dtype=np.float64)
        a_pad[1:] = a
        f1 = np.zeros(r + 1, dtype=np.float64)
        f1[1:] = a * self._sigma[1:]

        corr_plain = self._pos_lag_correlation(a, a)
        if r0 <= 0.0:
            corr_weighted = np.zeros_like(corr_plain)
            r0_pow = np.zeros(r, dtype=np.float64)
        else:
            weight = r0**4
            idx = np.arange(r, dtype=np.float64) + 1.0
            corr_weighted = self._pos_lag_correlation(a * (weight**idx), a)
            r0_pow = r0 ** (2.0 * np.arange(r, dtype=np.float64))

        f2 = np.zeros(r + 1, dtype=np.float64)
        if r >= 1:
            f2[1:r] = corr_plain[1:r] + r0_pow[1:r] * corr_weighted[1:r]

        conv1 = self._fft_convolve(a_pad * self._sigma, a_pad)
        conv2 = self._fft_convolve(a_pad, a_pad * self._delta)
        f3 = np.zeros(r + 1, dtype=np.float64)
        if r >= 2:
            f3[2:] = 0.5 * (conv1[2 : 2 + (r - 1)] - conv2[2 : 2 + (r - 1)])

        f[1:] = f1[1:] + f2[1:] + f3[1:]
        self._f_all = f

    def calculate_wave_speed(self) -> float:
        if not self._cache_valid:
            raise RuntimeError("load_coefficients must be called first")
        ell = np.arange(1, self.r + 1, dtype=np.float64)
        aff = np.sum(self._A * self._delta[1:] * self._f_all[1:] / ell)
        c2 = float(self._K) * self._f_all[0] - 2.0 * aff
        self.c = float(np.sqrt(np.maximum(0.0, np.abs(c2))))
        return self.c

    def _compute_coordinates_vectorized(self, theta, R, max_modes_per_batch: int = 256):
        theta = np.asarray(theta, dtype=np.float64)
        R = np.asarray(R, dtype=np.float64)
        shape = theta.shape
        theta_flat = theta.ravel()
        R_flat = R.ravel()
        i_all = np.arange(1, self.r + 1, dtype=np.float64)
        coef_all = self._A / i_all
        x_sum = np.zeros(theta_flat.size, dtype=np.float64)
        y_sum = np.zeros(theta_flat.size, dtype=np.float64)
        r0_sq = float(self.r0) ** 2

        for start in range(0, self.r, max_modes_per_batch):
            end = min(self.r, start + max_modes_per_batch)
            i_blk = i_all[start:end]
            coef = coef_all[start:end]
            R_pow = R_flat[:, None] ** i_blk[None, :]
            sin_it = np.sin(theta_flat[:, None] * i_blk[None, :])
            cos_it = np.cos(theta_flat[:, None] * i_blk[None, :])
            if abs(float(self.r0)) < 1e-15:
                x_sum += np.sum(R_pow * sin_it * coef[None, :], axis=1)
                y_sum += np.sum(R_pow * cos_it * coef[None, :], axis=1)
            else:
                R_inv_pow = (r0_sq / R_flat[:, None]) ** i_blk[None, :]
                x_sum += np.sum((R_pow + R_inv_pow) * sin_it * coef[None, :], axis=1)
                y_sum += np.sum((R_pow - R_inv_pow) * cos_it * coef[None, :], axis=1)

        x = -theta_flat - x_sum
        y = np.log(R_flat) + y_sum
        return x.reshape(shape), y.reshape(shape)

    def _compute_velocities_vectorized(self, theta, R, max_modes_per_batch: int = 256):
        theta = np.asarray(theta, dtype=np.float64)
        R = np.asarray(R, dtype=np.float64)
        shape = theta.shape
        theta_flat = theta.ravel()
        R_flat = R.ravel()
        i_all = np.arange(1, self.r + 1, dtype=np.float64)
        sum1 = np.zeros(theta_flat.size, dtype=np.float64)
        sum2 = np.zeros(theta_flat.size, dtype=np.float64)
        r0_sq = float(self.r0) ** 2

        for start in range(0, self.r, max_modes_per_batch):
            end = min(self.r, start + max_modes_per_batch)
            i_blk = i_all[start:end]
            a_blk = self._A[start:end]
            R_pow = R_flat[:, None] ** i_blk[None, :]
            cos_it = np.cos(theta_flat[:, None] * i_blk[None, :])
            sin_it = np.sin(theta_flat[:, None] * i_blk[None, :])
            if abs(float(self.r0)) < 1e-15:
                sum1 += np.sum(a_blk[None, :] * R_pow * cos_it, axis=1)
                sum2 += np.sum(a_blk[None, :] * R_pow * sin_it, axis=1)
            else:
                R_inv_pow = (r0_sq / R_flat[:, None]) ** i_blk[None, :]
                sum1 += np.sum(a_blk[None, :] * (R_pow + R_inv_pow) * cos_it, axis=1)
                sum2 += np.sum(a_blk[None, :] * (R_pow - R_inv_pow) * sin_it, axis=1)

        denom = np.maximum((1.0 + sum1) ** 2 + sum2**2, 1e-15)
        vx = self.c * (1.0 + sum1) / denom
        vy = self.c * sum2 / denom
        return vx.reshape(shape), vy.reshape(shape)

    @staticmethod
    def _ensure_zero_included(theta):
        if np.any(np.isclose(theta, 0.0)):
            return theta
        return np.sort(np.concatenate([theta, np.array([0.0])]))

    @staticmethod
    def _theta_cluster_sinh(theta_min, theta_max, n, alpha):
        if n % 2 == 0:
            n += 1
        center = 0.5 * (theta_min + theta_max)
        half = 0.5 * (theta_max - theta_min)
        u = np.linspace(-1.0, 1.0, n)
        return center + half * np.sinh(alpha * u) / np.sinh(alpha)

    @staticmethod
    def _solve_alpha_for_min_spacing(half, n, target_min_dtheta, alpha_lo=1e-6, alpha_hi=40.0):
        if target_min_dtheta is None or target_min_dtheta <= 0:
            return 0.0

        def dtheta_min(alpha):
            return half * (alpha / np.sinh(alpha)) * (2.0 / (n - 1))

        lo, hi = alpha_lo, alpha_hi
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if dtheta_min(mid) <= target_min_dtheta:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    def _suggest_theta_target(self, theta_range):
        th_min, th_max = theta_range
        if self.r0 is None:
            base = 0.005 * np.pi
        elif self.r0 >= 0.99:
            base = 0.005 * np.pi / 50.0
        elif self.r0 >= 0.95:
            base = 0.005 * np.pi
        else:
            base = 0.001 * np.pi
        return float(min(base, 0.1 * (th_max - th_min)))

    @staticmethod
    def _R_cluster_towards_one(r_begin, n, alpha):
        u = np.linspace(0.0, 1.0, n)
        if alpha <= 1e-12:
            R = r_begin + (1.0 - r_begin) * u
        else:
            R = 1.0 - (1.0 - r_begin) * np.sinh(alpha * (1.0 - u)) / np.sinh(alpha)
        R[-1] = 1.0
        return R

    def _suggest_R_target(self):
        if self.r0 is None:
            return 0.01
        if self.r0 >= 0.99:
            return 0.0001
        if self.r0 >= 0.95:
            return 0.0005
        return 0.001

    @staticmethod
    def _alpha_from_target_spacing_R(r_begin, n, target_min_dR, alpha_lo=1e-6, alpha_hi=40.0):
        if target_min_dR is None or target_min_dR <= 0 or (1.0 - r_begin) <= 0:
            return 0.0

        def dR_min(alpha):
            return (1.0 - r_begin) * (alpha / np.sinh(alpha)) * (1.0 / (n - 1))

        lo, hi = alpha_lo, alpha_hi
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if dR_min(mid) <= target_min_dR:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    def make_theta_R_grid(
        self,
        theta_num: Optional[int] = None,
        r_num: Optional[int] = None,
        theta_range: Tuple[float, float] = (-math.pi, math.pi),
        sample_strategy: str = "uniform",
        theta_min_spacing_target: Optional[float] = None,
        r_min_spacing_target: Optional[float] = None,
    ):
        actual_theta_num = theta_num or self.theta_num
        actual_r_num = r_num or self.r_num
        th_min, th_max = theta_range

        if sample_strategy == "adaptive":
            ntheta = actual_theta_num if actual_theta_num % 2 == 1 else actual_theta_num + 1
            target = theta_min_spacing_target if theta_min_spacing_target is not None else self._suggest_theta_target(theta_range)
            alpha_th = self._solve_alpha_for_min_spacing(0.5 * (th_max - th_min), ntheta, target)
            theta = self._theta_cluster_sinh(th_min, th_max, ntheta, alpha_th)
            if not (th_min <= 0.0 <= th_max):
                theta = self._ensure_zero_included(theta)
        else:
            theta = np.linspace(th_min, th_max, actual_theta_num)
            if not (th_min <= 0.0 <= th_max):
                theta = self._ensure_zero_included(theta)

        if sample_strategy == "adaptive":
            target_R = r_min_spacing_target if r_min_spacing_target is not None else self._suggest_R_target()
            alpha_R = self._alpha_from_target_spacing_R(self.r_begin, actual_r_num, target_R)
            R = self._R_cluster_towards_one(self.r_begin, actual_r_num, alpha_R)
        else:
            R = np.linspace(self.r_begin, 1.0, actual_r_num)

        return np.meshgrid(theta, R, indexing="xy")

    def generate_grid_data(
        self,
        theta_num: Optional[int] = None,
        r_num: Optional[int] = None,
        theta_range: Tuple[float, float] = (-math.pi, math.pi),
        sample_strategy: str = "adaptive",
        theta_min_spacing_target: Optional[float] = None,
        r_min_spacing_target: Optional[float] = None,
    ):
        theta_grid, R_grid = self.make_theta_R_grid(
            theta_num=theta_num,
            r_num=r_num,
            theta_range=theta_range,
            sample_strategy=sample_strategy,
            theta_min_spacing_target=theta_min_spacing_target,
            r_min_spacing_target=r_min_spacing_target,
        )
        x, y = self._compute_coordinates_vectorized(theta_grid, R_grid)
        vx, vy = self._compute_velocities_vectorized(theta_grid, R_grid)
        v = np.sqrt(vx**2 + vy**2)
        return x, y, vx, vy, v, theta_grid, R_grid


def compute_xy_thR_from_coeffs(
    coeff_path: str,
    H: int,
    W: int,
    sample_strategy: str,
    theta_min_spacing_target=None,
    r_min_spacing_target=None,
):
    wp = WaveProcessor(theta_num=W, r_num=H)
    wp.load_coefficients(coeff_path)
    r0, _, stp = parse_filename(coeff_path)
    theta_grid, R_grid = wp.make_theta_R_grid(
        theta_num=W,
        r_num=H,
        theta_range=(-math.pi, math.pi),
        sample_strategy=sample_strategy,
        theta_min_spacing_target=theta_min_spacing_target,
        r_min_spacing_target=r_min_spacing_target,
    )
    x, y = wp._compute_coordinates_vectorized(theta_grid, R_grid, max_modes_per_batch=XY_J_BLOCK)
    x_mean, x_std = x.mean(), x.std() + 1e-8
    y_mean, y_std = y.mean(), y.std() + 1e-8
    return {
        "Xn": ((x - x_mean) / x_std).astype(np.float32),
        "Yn": ((y - y_mean) / y_std).astype(np.float32),
        "TH": theta_grid.astype(np.float32),
        "RR": R_grid.astype(np.float32),
        "r0": np.float32(r0),
        "stp": np.float32(stp),
        "x_mean": np.float64(x_mean),
        "x_std": np.float64(x_std),
        "y_mean": np.float64(y_mean),
        "y_std": np.float64(y_std),
    }


def _read_npz_string(data, key: str) -> str:
    value = data[key]
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def maybe_precompute_to_npz(
    coeff_path: str,
    out_dir: str,
    H: int,
    W: int,
    sample_strategy: str,
    theta_min_spacing_target=None,
    r_min_spacing_target=None,
):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(coeff_path))[0]
    npz_path = os.path.join(out_dir, base + ".npz")

    def norm_target(value):
        return np.nan if value is None else float(value)

    if os.path.isfile(npz_path):
        try:
            cached = np.load(npz_path, mmap_mode="r")
            if _read_npz_string(cached, "ver") != NPZ_VERSION:
                raise ValueError("cache version mismatch")
            if int(cached["H"]) != int(H) or int(cached["W"]) != int(W):
                raise ValueError("grid size mismatch")
            if _read_npz_string(cached, "sample_strategy") != sample_strategy:
                raise ValueError("sample strategy mismatch")
            if not np.allclose(float(cached["theta_min_spacing_target"]), norm_target(theta_min_spacing_target), equal_nan=True):
                raise ValueError("theta target mismatch")
            if not np.allclose(float(cached["r_min_spacing_target"]), norm_target(r_min_spacing_target), equal_nan=True):
                raise ValueError("R target mismatch")
            return npz_path
        except Exception:
            pass

    data = compute_xy_thR_from_coeffs(
        coeff_path,
        H=H,
        W=W,
        sample_strategy=sample_strategy,
        theta_min_spacing_target=theta_min_spacing_target,
        r_min_spacing_target=r_min_spacing_target,
    )
    np.savez_compressed(
        npz_path,
        **data,
        H=np.int32(H),
        W=np.int32(W),
        sample_strategy=np.array(sample_strategy),
        theta_min_spacing_target=np.float64(norm_target(theta_min_spacing_target)),
        r_min_spacing_target=np.float64(norm_target(r_min_spacing_target)),
        ver=np.array(NPZ_VERSION),
    )
    return npz_path


def focus_weights(theta, R):
    w_theta = np.exp(-(np.asarray(theta, dtype=np.float64) / FOCUS_TH_SIGMA) ** 2)
    w_R = np.exp(-((1.0 - np.asarray(R, dtype=np.float64)) / FOCUS_R_SIGMA) ** 2)
    return (1.0 + FOCUS_GAIN * w_theta * w_R).astype(np.float32)


def pin_indices(H: int, W: int):
    return H - 1, W // 2


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes_h, modes_w):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_h = modes_h
        self.modes_w = modes_w
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, modes_h, modes_w, 2, dtype=torch.float32) * scale)

    def compl_mul2d(self, a, b):
        a_r, a_i = a[..., 0], a[..., 1]
        b_r, b_i = b[..., 0], b[..., 1]
        out_r = torch.einsum("bixy,ioxy->boxy", a_r, b_r) - torch.einsum("bixy,ioxy->boxy", a_i, b_i)
        out_i = torch.einsum("bixy,ioxy->boxy", a_r, b_i) + torch.einsum("bixy,ioxy->boxy", a_i, b_r)
        return torch.stack([out_r, out_i], dim=-1)

    def forward(self, x):
        bsz, _, H, W = x.shape
        in_dtype = x.dtype
        x_ft = torch.fft.rfft2(x.to(torch.float32), norm="ortho")
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)
        out_ft = torch.zeros(bsz, self.out_channels, H, W // 2 + 1, 2, device=x.device, dtype=torch.float32)
        mh = min(self.modes_h, H)
        mw = min(self.modes_w, W // 2 + 1)
        out_ft[:, :, :mh, :mw] = self.compl_mul2d(x_ft[:, :, :mh, :mw], self.weight[:, :, :mh, :mw])
        out = torch.fft.irfft2(torch.complex(out_ft[..., 0], out_ft[..., 1]), s=(H, W), norm="ortho")
        return out.to(in_dtype)


class FNO2d(nn.Module):
    def __init__(self, in_channels=4, out_channels=2, width=WIDTH_DEFAULT, modes_h=MODES_R_DEFAULT, modes_w=MODES_TH_DEFAULT, layers=LAYERS_DEFAULT):
        super().__init__()
        self.fc0 = nn.Conv2d(in_channels, width, 1)
        self.scs = nn.ModuleList([SpectralConv2d(width, width, modes_h, modes_w) for _ in range(layers)])
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.act = nn.GELU()
        self.fc1 = nn.Conv2d(width, width, 1)
        self.fc2 = nn.Conv2d(width, out_channels, 1)

    def forward(self, x):
        x = self.fc0(x)
        for spectral, pointwise in zip(self.scs, self.ws):
            x = self.act(spectral(x) + pointwise(x))
        return self.fc2(self.act(self.fc1(x)))


class InverseDataset(Dataset):
    def __init__(
        self,
        coeffs_dir: str,
        H: int,
        W: int,
        cache_dir: str,
        use_cache: bool = True,
        sample_strategy: str = "adaptive",
        use_normalized_stp: bool = False,
        theta_min_spacing_target=None,
        r_min_spacing_target=None,
    ):
        self.files = sorted(glob.glob(os.path.join(coeffs_dir, "*.txt")))
        if not self.files:
            raise FileNotFoundError(f"No .txt coefficient files found in {coeffs_dir}")
        self.H = int(H)
        self.W = int(W)
        self.cache_dir = os.path.join(cache_dir, os.path.basename(os.path.normpath(coeffs_dir)) or "data")
        self.use_cache = bool(use_cache)
        self.sample_strategy = sample_strategy
        self.use_normalized_stp = bool(use_normalized_stp)
        self.theta_min_spacing_target = theta_min_spacing_target
        self.r_min_spacing_target = r_min_spacing_target
        self.cached_paths = []

        if self.use_cache:
            for path in self.files:
                self.cached_paths.append(
                    maybe_precompute_to_npz(
                        path,
                        self.cache_dir,
                        self.H,
                        self.W,
                        self.sample_strategy,
                        self.theta_min_spacing_target,
                        self.r_min_spacing_target,
                    )
                )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.use_cache:
            data = np.load(self.cached_paths[idx], mmap_mode="r")
            Xn, Yn, TH, RR = data["Xn"], data["Yn"], data["TH"], data["RR"]
            r0, stp = float(data["r0"]), float(data["stp"])
        else:
            data = compute_xy_thR_from_coeffs(
                self.files[idx],
                self.H,
                self.W,
                self.sample_strategy,
                self.theta_min_spacing_target,
                self.r_min_spacing_target,
            )
            Xn, Yn, TH, RR = data["Xn"], data["Yn"], data["TH"], data["RR"]
            r0, stp = float(data["r0"]), float(data["stp"])

        param = parameter_channel_value(r0, stp, self.use_normalized_stp)
        inp = np.stack(
            [
                Xn.astype(np.float32),
                Yn.astype(np.float32),
                np.full_like(Xn, r0, dtype=np.float32),
                np.full_like(Xn, param, dtype=np.float32),
            ],
            axis=0,
        )
        target = np.stack([TH, RR], axis=0).astype(np.float32)
        weight = focus_weights(TH, RR)
        r_idx, th_idx = pin_indices(self.H, self.W)
        meta = {"r0": np.float32(r0), "stp": np.float32(stp), "r_idx": r_idx, "th_idx": th_idx}
        return torch.from_numpy(inp), torch.from_numpy(target), torch.from_numpy(weight), meta


def build_amp(amp_on: bool, device: torch.device):
    amp_on = bool(amp_on and device.type == "cuda")
    try:
        from torch.amp import GradScaler, autocast

        scaler = GradScaler("cuda", enabled=amp_on)

        def autocast_ctx():
            return autocast(device_type="cuda", dtype=torch.float16, enabled=amp_on)

    except Exception:
        from torch.cuda.amp import GradScaler, autocast

        scaler = GradScaler(enabled=amp_on)

        def autocast_ctx():
            return autocast(enabled=amp_on)

    return scaler, autocast_ctx


def weighted_mse(pred, target, weight):
    if weight.dim() == 3:
        weight = weight.unsqueeze(1)
    return (((pred - target) ** 2) * weight).mean()


def range_penalty(x, lo, hi):
    return (F.relu(lo - x) ** 2 + F.relu(x - hi) ** 2).mean()


@torch.no_grad()
def rel_l2(pred, target, eps=1e-12):
    return (torch.norm(pred - target) / (torch.norm(target) + eps)).item()


def fno_loss(model, inp, target, weight, meta, autocast_ctx, range_pen_w):
    r0_img = inp[:, 2:3]
    lo_theta, hi_theta = -math.pi, math.pi
    lo_R, hi_R = r0_img, torch.ones_like(r0_img)

    with autocast_ctx():
        out = model(inp)
        theta_raw = out[:, 0:1]
        R_raw = out[:, 1:2]
        theta_pred = torch.clamp(theta_raw, lo_theta, hi_theta)
        R_pred = torch.clamp(R_raw, lo_R, hi_R)
        theta_true = target[:, 0:1]
        R_true = target[:, 1:2]
        loss_theta = weighted_mse(theta_pred, theta_true, weight)
        loss_R = weighted_mse(R_pred, R_true, weight)
        loss_range = range_penalty(theta_raw, lo_theta, hi_theta) + range_penalty(R_raw, lo_R, hi_R)

        pin_loss = 0.0
        for b in range(inp.size(0)):
            r_idx = int(meta["r_idx"][b])
            th_idx = int(meta["th_idx"][b])
            pin_loss = pin_loss + PIN_GAIN_THETA * (theta_pred[b, 0, r_idx, th_idx] ** 2)
            pin_loss = pin_loss + PIN_GAIN_R * ((R_pred[b, 0, r_idx, th_idx] - 1.0) ** 2)
        pin_loss = pin_loss / inp.size(0)
        loss = loss_theta + loss_R + range_pen_w * loss_range + pin_loss
    return loss, theta_pred, R_pred, theta_true, R_true


def train_one_epoch(model, loader, optimizer, scaler, autocast_ctx, device, accum_steps, clip_norm, range_pen_w):
    model.train()
    totals = {"loss": 0.0, "theta": 0.0, "R": 0.0}
    optimizer.zero_grad(set_to_none=True)
    nsteps = 0
    for step, (inp, target, weight, meta) in enumerate(loader, 1):
        inp = inp.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        weight = weight.to(device, non_blocking=True).unsqueeze(1)
        loss, theta_pred, R_pred, theta_true, R_true = fno_loss(model, inp, target, weight, meta, autocast_ctx, range_pen_w)

        if scaler.is_enabled():
            scaler.scale(loss / accum_steps).backward()
        else:
            (loss / accum_steps).backward()

        if step % accum_steps == 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        totals["loss"] += float(loss.item())
        totals["theta"] += rel_l2(theta_pred.squeeze(1), theta_true.squeeze(1))
        totals["R"] += rel_l2(R_pred.squeeze(1), R_true.squeeze(1))
        nsteps += 1
    if nsteps % accum_steps != 0:
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {key: val / max(nsteps, 1) for key, val in totals.items()}


@torch.no_grad()
def evaluate(model, loader, autocast_ctx, device, range_pen_w):
    model.eval()
    totals = {"loss": 0.0, "theta": 0.0, "R": 0.0}
    nsteps = 0
    for inp, target, weight, meta in loader:
        inp = inp.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        weight = weight.to(device, non_blocking=True).unsqueeze(1)
        loss, theta_pred, R_pred, theta_true, R_true = fno_loss(model, inp, target, weight, meta, autocast_ctx, range_pen_w)
        totals["loss"] += float(loss.item())
        totals["theta"] += rel_l2(theta_pred.squeeze(1), theta_true.squeeze(1))
        totals["R"] += rel_l2(R_pred.squeeze(1), R_true.squeeze(1))
        nsteps += 1
    return {key: val / max(nsteps, 1) for key, val in totals.items()}


def save_tecplot_dat(path, X, Y, VX, VY, V):
    H, W = X.shape
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = np.column_stack([X.reshape(-1), Y.reshape(-1), VX.reshape(-1), VY.reshape(-1), V.reshape(-1)])
    header = f'VARIABLES = "X","Y","VX","VY","V"\nZONE T="MyZone" I={W} J={H} F=POINT\n'
    np.savetxt(path, data, header=header, comments="", fmt="%.10e", delimiter="\t")


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("cfg", {})
    model = FNO2d(
        in_channels=4,
        out_channels=2,
        width=int(cfg.get("WIDTH", WIDTH_DEFAULT)),
        modes_h=int(cfg.get("MODES_R", MODES_R_DEFAULT)),
        modes_w=int(cfg.get("MODES_TH", MODES_TH_DEFAULT)),
        layers=int(cfg.get("LAYERS", LAYERS_DEFAULT)),
    ).to(device)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg


def evaluate_single_file(
    model,
    cfg,
    file_path,
    H,
    W,
    sample_strategy,
    use_normalized_stp,
    device,
    theta_min_spacing_target=None,
    r_min_spacing_target=None,
    dat_dir=None,
):
    wp = WaveProcessor(theta_num=W, r_num=H)
    wp.load_coefficients(file_path)
    r0, r_from_name, stp = parse_filename(file_path)
    c = wp.calculate_wave_speed()
    c_abs = max(abs(c), 1e-12)
    theta_range = (float(cfg.get("THETA_MIN", -math.pi)), float(cfg.get("THETA_MAX", math.pi)))

    x_gt, y_gt, vx_gt, vy_gt, v_gt, theta_grid, R_grid = wp.generate_grid_data(
        theta_num=W,
        r_num=H,
        theta_range=theta_range,
        sample_strategy=sample_strategy,
        theta_min_spacing_target=theta_min_spacing_target,
        r_min_spacing_target=r_min_spacing_target,
    )

    Xn = ((x_gt - x_gt.mean()) / (x_gt.std() + 1e-8)).astype(np.float32)
    Yn = ((y_gt - y_gt.mean()) / (y_gt.std() + 1e-8)).astype(np.float32)
    param = parameter_channel_value(r0, stp, use_normalized_stp)
    inp = np.stack([Xn, Yn, np.full_like(Xn, r0, dtype=np.float32), np.full_like(Xn, param, dtype=np.float32)], axis=0)[None, ...]
    inp_t = torch.from_numpy(inp).to(device)

    with torch.no_grad():
        out = model(inp_t)
        theta_pred = torch.clamp(out[:, 0:1], -math.pi, math.pi)
        R_pred = torch.clamp(out[:, 1:2], inp_t[:, 2:3], torch.ones_like(inp_t[:, 2:3]))

    theta_pred_np = theta_pred.cpu().numpy()[0, 0]
    R_pred_np = R_pred.cpu().numpy()[0, 0]
    vx_pred, vy_pred = wp._compute_velocities_vectorized(theta_pred_np, R_pred_np)
    v_pred = np.sqrt(vx_pred**2 + vy_pred**2)

    def mae(a, b):
        return float(np.mean(np.abs(a - b)))

    if dat_dir:
        os.makedirs(dat_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(file_path))[0]
        save_tecplot_dat(os.path.join(dat_dir, base + "_gt.dat"), x_gt, y_gt, vx_gt, vy_gt, v_gt)
        save_tecplot_dat(os.path.join(dat_dir, base + "_pred.dat"), x_gt, y_gt, vx_pred, vy_pred, v_pred)
        save_tecplot_dat(os.path.join(dat_dir, base + "_err.dat"), x_gt, y_gt, np.abs(vx_pred - vx_gt), np.abs(vy_pred - vy_gt), np.abs(v_pred - v_gt))

    return {
        "file": os.path.basename(file_path),
        "r": int(wp.r),
        "r_from_name": int(r_from_name),
        "r0": float(wp.r0),
        "stp": float(stp),
        "theta_points": int(W),
        "r_points": int(H),
        "c": float(c),
        "mae_theta": mae(theta_pred_np, theta_grid),
        "mae_R": mae(R_pred_np, R_grid),
        "mae_vx": mae(vx_pred, vx_gt),
        "mae_vy": mae(vy_pred, vy_gt),
        "mae_v": mae(v_pred, v_gt),
        "mae_vx_over_c": float(np.mean(np.abs(vx_pred - vx_gt) / c_abs)),
        "mae_vy_over_c": float(np.mean(np.abs(vy_pred - vy_gt) / c_abs)),
        "mae_v_over_c": float(np.mean(np.abs(v_pred - v_gt) / c_abs)),
    }


def command_train(args):
    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    train_dir = args.train_coeffs_dir or args.coeffs_dir
    if args.val_coeffs_dir:
        train_set = InverseDataset(train_dir, args.hr, args.wth, args.cache_dir, bool(args.precompute), args.sample_strategy, args.use_normalized_stp, args.theta_min_spacing_target, args.r_min_spacing_target)
        val_set = InverseDataset(args.val_coeffs_dir, args.hr, args.wth, args.cache_dir, bool(args.precompute), args.sample_strategy, args.use_normalized_stp, args.theta_min_spacing_target, args.r_min_spacing_target)
        split_mode = "directory"
    else:
        dataset = InverseDataset(train_dir, args.hr, args.wth, args.cache_dir, bool(args.precompute), args.sample_strategy, args.use_normalized_stp, args.theta_min_spacing_target, args.r_min_spacing_target)
        n_train = int(len(dataset) * args.train_split)
        n_val = len(dataset) - n_train
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
        split_mode = "random"

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    model = FNO2d(width=args.width, modes_h=args.modes_r, modes_w=args.modes_th, layers=args.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=8)
    scaler, autocast_ctx = build_amp(bool(args.amp), device)
    best = float("inf")
    ckpt_path = os.path.join(args.out_dir, "best_fno_thetaR.pt")

    print(f"[train] device={device} train={len(train_set)} val={len(val_set)} grid={args.hr}x{args.wth} split={split_mode}")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, autocast_ctx, device, args.accum_steps, args.clip_norm, args.range_penalty_weight)
        val_metrics = evaluate(model, val_loader, autocast_ctx, device, args.range_penalty_weight)
        scheduler.step(val_metrics["loss"])
        elapsed = time.time() - start
        print(
            f"[Epoch {epoch:03d}] train={train_metrics['loss']:.4e} "
            f"val={val_metrics['loss']:.4e} theta_rel={val_metrics['theta']:.3e} R_rel={val_metrics['R']:.3e} {elapsed:.1f}s"
        )
        if val_metrics["loss"] < best:
            best = val_metrics["loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": {
                        "H_R": int(args.hr),
                        "W_TH": int(args.wth),
                        "H": int(args.hr),
                        "W": int(args.wth),
                        "MODES_R": int(args.modes_r),
                        "MODES_TH": int(args.modes_th),
                        "WIDTH": int(args.width),
                        "LAYERS": int(args.layers),
                        "SAMPLE_STRATEGY": args.sample_strategy,
                        "USE_NORMALIZED_STP": bool(args.use_normalized_stp),
                        "THETA_MIN": -math.pi,
                        "THETA_MAX": math.pi,
                    },
                    "best_val_loss": best,
                },
                ckpt_path,
            )
            print(f"  saved {ckpt_path}")

    manifest = {
        "script": "FNO.py",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_coeffs_dir": train_dir,
        "val_coeffs_dir": args.val_coeffs_dir,
        "out_dir": args.out_dir,
        "best_checkpoint": ckpt_path,
        "use_normalized_stp": bool(args.use_normalized_stp),
        "sample_strategy": args.sample_strategy,
        "grid": [int(args.hr), int(args.wth)],
        "device": str(device),
    }
    with open(os.path.join(args.out_dir, "run_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def command_test(args):
    device = get_device(args.device)
    model, cfg = load_model(args.ckpt, device)
    H = args.hr or int(cfg.get("H_R", cfg.get("H", H_R_DEFAULT)))
    W = args.wth or int(cfg.get("W_TH", cfg.get("W", W_TH_DEFAULT)))
    sample_strategy = args.sample_strategy or cfg.get("SAMPLE_STRATEGY", "adaptive")
    if args.use_normalized_stp == "auto":
        use_normalized_stp = bool(cfg.get("USE_NORMALIZED_STP", False))
    else:
        use_normalized_stp = bool(int(args.use_normalized_stp))

    os.makedirs(args.out_dir, exist_ok=True)
    dat_dir = os.path.join(args.out_dir, "dat") if args.write_dat else None
    files = sorted(glob.glob(os.path.join(args.test_dir, "*.txt")))
    if not files:
        raise FileNotFoundError(f"No .txt coefficient files found in {args.test_dir}")

    results = []
    for idx, path in enumerate(files, 1):
        try:
            res = evaluate_single_file(
                model,
                cfg,
                path,
                H=H,
                W=W,
                sample_strategy=sample_strategy,
                use_normalized_stp=use_normalized_stp,
                device=device,
                theta_min_spacing_target=args.theta_min_spacing_target,
                r_min_spacing_target=args.r_min_spacing_target,
                dat_dir=dat_dir,
            )
            results.append(res)
            print(f"[{idx:03d}/{len(files)}] {res['file']} mae(v/c)={res['mae_v_over_c']:.3e} mae(theta)={res['mae_theta']:.3e} mae(R)={res['mae_R']:.3e}")
        except Exception as exc:
            print(f"[{idx:03d}/{len(files)}] {os.path.basename(path)} ERROR: {exc}")

    if not results:
        raise RuntimeError("No test case finished successfully.")

    csv_path = args.out_csv if os.path.isabs(args.out_csv) else os.path.join(args.out_dir, args.out_csv)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    def avg(key):
        return float(np.mean([row[key] for row in results]))

    print("[summary]")
    print(f"  mae(vx/c) = {avg('mae_vx_over_c'):.6e}")
    print(f"  mae(vy/c) = {avg('mae_vy_over_c'):.6e}")
    print(f"  mae(v/c)  = {avg('mae_v_over_c'):.6e}")
    print(f"  mae(theta)= {avg('mae_theta'):.6e}")
    print(f"  mae(R)    = {avg('mae_R'):.6e}")
    print(f"  csv       = {csv_path}")


def add_train_args(parser):
    parser.add_argument("--coeffs-dir", default="data")
    parser.add_argument("--train-coeffs-dir", default=None)
    parser.add_argument("--val-coeffs-dir", default=None)
    parser.add_argument("--out-dir", default="runs/fno")
    parser.add_argument("--cache-dir", default="runs/fno_cache")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--precompute", type=int, default=1)
    parser.add_argument("--hr", type=int, default=H_R_DEFAULT)
    parser.add_argument("--wth", type=int, default=W_TH_DEFAULT)
    parser.add_argument("--sample-strategy", choices=["adaptive", "uniform"], default="adaptive")
    parser.add_argument("--theta-min-spacing-target", type=float, default=None)
    parser.add_argument("--r-min-spacing-target", type=float, default=None)
    parser.add_argument("--use-normalized-stp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--width", type=int, default=WIDTH_DEFAULT)
    parser.add_argument("--layers", type=int, default=LAYERS_DEFAULT)
    parser.add_argument("--modes-r", type=int, default=MODES_R_DEFAULT)
    parser.add_argument("--modes-th", type=int, default=MODES_TH_DEFAULT)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--range-penalty-weight", type=float, default=RANGE_PEN_W)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto")


def main():
    parser = argparse.ArgumentParser(description="FNO train/test entry point.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the inverse-mapping FNO.")
    add_train_args(train_parser)
    train_parser.set_defaults(func=command_train)

    test_parser = subparsers.add_parser("test", help="Evaluate a trained FNO checkpoint.")
    test_parser.add_argument("--test-dir", required=True)
    test_parser.add_argument("--ckpt", required=True)
    test_parser.add_argument("--out-dir", default="eval_fno")
    test_parser.add_argument("--out-csv", default="fno_test_summary.csv")
    test_parser.add_argument("--hr", type=int, default=None)
    test_parser.add_argument("--wth", type=int, default=None)
    test_parser.add_argument("--sample-strategy", choices=["adaptive", "uniform"], default=None)
    test_parser.add_argument("--theta-min-spacing-target", type=float, default=None)
    test_parser.add_argument("--r-min-spacing-target", type=float, default=None)
    test_parser.add_argument("--use-normalized-stp", choices=["auto", "0", "1"], default="auto")
    test_parser.add_argument("--write-dat", action=argparse.BooleanOptionalAction, default=True)
    test_parser.add_argument("--device", default="auto")
    test_parser.set_defaults(func=command_test)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
