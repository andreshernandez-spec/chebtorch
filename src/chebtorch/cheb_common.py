# Copyright (c) 2026, Andres Hernandez
# SPDX-License-Identifier: BSD-3-Clause
# From chebtorch: https://github.com/andreshernandez-spec/chebtorch
"""Helpers every chebtorch family imports: Clenshaw evaluation, the table
cache, dtype and device resolution, tensors from arguments without a host
sync, and the check for parameters that are uniform per call. Imports only
torch.

Chebyshev coefficients follow numpy.polynomial: plain c[0], lowest degree
first, the coordinate mapped to [-1, 1] by the caller. The names here carry no
underscore so families can import them, but they are internal: chebtorch's API
is what chebtorch exports.

A project that copies a family copies this file next to it, under this name,
which is specific enough not to clash with the project's own modules.
"""

import torch


def cheb(x, c):
    """sum_k c[k] T_k(x) by Clenshaw. c's first axis is the degree, so a
    (degree, n) table gives n series at once."""
    x2 = 2 * x                   # exact, so the same bits as 2 * x * b1 per step
    b1 = torch.zeros_like(x * c[0])
    b2 = torch.zeros_like(b1)
    for k in range(c.shape[0] - 1, 0, -1):
        b1, b2 = x2 * b1 - b2 + c[k], b1
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
            # the first call may come from an inference_mode block, and
            # autograd cannot save inference tensors in later calls
            with torch.inference_mode(False):
                t = torch.tensor(self._data[name], dtype=torch.float64)
                t = t.reshape(self._shapes[name]).to(dtype=dtype, device=device)
            self._cache[key] = t
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


def tensor_of(v, dtype, device):
    """v as a tensor in dtype on device. A python number is filled in on the
    device: torch.tensor(v, device="cuda") copies it from the host, which
    waits for all queued GPU work. A number beyond dtype's largest is inf,
    where torch.full would raise."""
    if isinstance(v, (int, float)):
        v = float(v)
        if abs(v) > torch.finfo(dtype).max:
            v = float("inf") if v > 0 else -float("inf")
        return torch.full((), v, dtype=dtype, device=device)
    return torch.as_tensor(v, dtype=dtype, device=device)


def scalar(v, dtype, device, who, name, lo, hi):
    """A uniform-per-call parameter as a 0-d tensor. A python number is
    checked against the box here; a tensor is not (it may be traced) and
    gives nan downstream when out of the box."""
    if not isinstance(v, torch.Tensor):
        if not lo <= float(v) <= hi:
            raise ValueError(f"{who} covers {name} in [{lo}, {hi}], got {v}")
        return tensor_of(float(v), dtype, device)
    if v.numel() != 1:
        raise ValueError(f"{who} takes {name} as a scalar (uniform per call), "
                         f"got shape {tuple(v.shape)}")
    return v.reshape(()).to(dtype=dtype, device=device)
