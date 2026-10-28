# Copyright (c) 2026, Andres Hernandez
# SPDX-License-Identifier: BSD-3-Clause
# From chebtorch: https://github.com/andreshernandez-spec/chebtorch
"""Helpers every chebtorch family imports: Clenshaw evaluation, the table
cache, dtype and device resolution, tensors from arguments without a host
sync, the check for parameters that are uniform per call, and the cache of
what depends on such a parameter alone. Imports only torch.

Chebyshev coefficients follow numpy.polynomial: plain c[0], lowest degree
first, the coordinate mapped to [-1, 1] by the caller. The names here carry no
underscore so families can import them, but they are internal: chebtorch's API
is what chebtorch exports.

A project that copies a family copies this file next to it, under this name,
which is specific enough not to clash with the project's own modules.
"""

import torch

_KEEP = 64                       # values of a parameter a Memo keeps


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
    """A uniform-per-call parameter: a python number checked against the box
    and returned as a float (Memo keys on it), or a one-element tensor as a
    0-d tensor in dtype on device. A tensor is not checked (it may be
    traced) and gives nan downstream when out of the box."""
    if not isinstance(v, torch.Tensor):
        if not lo <= float(v) <= hi:
            raise ValueError(f"{who} covers {name} in [{lo}, {hi}], got {v}")
        return float(v)
    if v.numel() != 1:
        raise ValueError(f"{who} takes {name} as a scalar (uniform per call), "
                         f"got shape {tuple(v.shape)}")
    return v.reshape(()).to(dtype=dtype, device=device)


class Memo:
    """fn of a uniform-per-call parameter, kept per (value, dtype, device)
    when the parameter is a python number: what depends on it alone then
    costs a dict lookup on later calls, not fn's kernels. Called as
    memo(v, dtype, device) with v from scalar(); fn gets v as a 0-d tensor.
    A tensor v runs fn every time, as do calls under torch.compile, which
    puts that work in the graph, and calls a CUDA graph captures.

    Keeping a result waits for the GPU once, so the result is complete before
    another stream reads it, and builds it outside inference mode, like
    Tables. Nothing kept is dropped, since a captured CUDA graph may still
    read it; past _KEEP values fn runs every time."""

    def __init__(self, fn):
        self._fn, self._kept = fn, {}

    def __call__(self, v, dtype, device):
        if isinstance(v, torch.Tensor) or torch.compiler.is_compiling():
            return self._fn(tensor_of(v, dtype, device))
        key = (v, dtype, device)
        out = self._kept.get(key)
        if out is None:
            if len(self._kept) >= _KEEP or _capturing(device):
                return self._fn(tensor_of(v, dtype, device))
            with torch.inference_mode(False):
                out = self._fn(tensor_of(v, dtype, device))
            if device.type == "cuda":
                torch.cuda.current_stream(device).synchronize()
            self._kept[key] = out
        return out


def _capturing(device):
    """Whether a CUDA graph is capturing device's current stream."""
    if device.type != "cuda":
        return False
    with torch.cuda.device(device):
        return torch.cuda.is_current_stream_capturing()
