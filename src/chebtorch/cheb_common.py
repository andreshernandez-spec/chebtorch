# Copyright (c) 2026, Andres Hernandez
# SPDX-License-Identifier: BSD-3-Clause (https://github.com/andreshernandez-spec/chebtorch)
"""Helpers every chebtorch family imports: Clenshaw evaluation, the table
cache, dtype and device resolution, and the check for parameters that are
uniform per call. Imports only torch.

A project that copies a family copies this file next to it, under this name,
which is specific enough not to clash with the project's own modules.
"""

import torch


def cheb(x, c):
    """sum_k c[k] T_k(x) by Clenshaw. c's first axis is the degree, so a
    (degree, n) table gives n series at once."""
    b1 = torch.zeros_like(x * c[0])
    b2 = torch.zeros_like(b1)
    for k in range(c.shape[0] - 1, 0, -1):
        b1, b2 = 2 * x * b1 - b2 + c[k], b1
    return x * b1 - b2 + c[0]


class Tables:
    """A family's generated tables as tensors, each built once per (dtype,
    device) from its float64 literals."""

    def __init__(self, shapes, data):
        self._shapes, self._data, self._cache = shapes, data, {}

    def __call__(self, name, dtype, device):
        key = (name, dtype, device)
        t = self._cache.get(key)
        if t is None:
            t = torch.tensor(self._data[name], dtype=torch.float64).reshape(self._shapes[name])
            t = self._cache[key] = t.to(dtype=dtype, device=device)
        return t


def work_dtype(*xs):
    """(compute dtype, result dtype) for these arguments: the widest
    floating tensor among them, else the default dtype. Half precision
    computes in float32 and is cast back."""
    out = None
    for x in xs:
        if isinstance(x, torch.Tensor) and x.is_floating_point():
            out = x.dtype if out is None else torch.promote_types(out, x.dtype)
    out = torch.get_default_dtype() if out is None else out
    work = torch.float32 if out in (torch.float16, torch.bfloat16) else out
    return work, out


def device_of(*xs):
    """The device of the first tensor argument, else the CPU."""
    for x in xs:
        if isinstance(x, torch.Tensor):
            return x.device
    return torch.device("cpu")


def scalar(v, dtype, device, who, name, lo, hi):
    """A uniform-per-call parameter as a 0-d tensor. A python number is
    checked against the box here; a tensor is not (it may be traced) and
    gives nan downstream when out of the box."""
    if not isinstance(v, torch.Tensor):
        if not lo <= float(v) <= hi:
            raise ValueError(f"{who} covers {name} in [{lo}, {hi}], got {v}")
        return torch.tensor(float(v), dtype=dtype, device=device)
    if v.numel() != 1:
        raise ValueError(f"{who} takes {name} as a scalar (uniform per call), "
                         f"got shape {tuple(v.shape)}")
    return v.reshape(()).to(dtype=dtype, device=device)
