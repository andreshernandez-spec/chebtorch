#!/usr/bin/env python3
"""Tweedie negative log-likelihood on the GPU: chebtorch, eager and compiled,
against the saddlepoint approximation and neuralforecast's kernel-only loss.

Time per call of the mean NLL of N = 2^12, 2^16 and 2^20 draws (mu = 2, phi = 1,
p = 1.5), float32 and float64, values alone and with gradients in p, phi and a
per-element mu. Losses: kernel, neuralforecast's -(y mu^(1-p)/(1-p) -
mu^(2-p)/(2-p)), which has no phi or normalizing term; saddle, exact at y = 0;
chebtorch's tweedie_logpdf with a tensor and a python p; tweedie_logpdf_fixed_p.
Also compile time, CUDA kernels and peak memory per eager call, and the saddle's
error per observation at p = 1.2, 1.5, 1.8.

Findings (2026-10-28, RTX 3080 Laptop, torch 2.13; results/tweedie_gpu.txt):
- Eager chebtorch is launch-bound. float32 with gradients at 2^16: 19 ms with a
  tensor p (1466 kernels), 9.3 ms with a python p, 1.9 ms for
  tweedie_logpdf_fixed_p (110 kernels), against 2.4 ms for the saddle.
- Compiled float32 up to 2^16, with gradients: 0.4-0.5 ms for
  tweedie_logpdf_fixed_p, 0.3-0.4 ms for the kernel loss. Cold compiles 5-66 s.
- Compiled float64 can lose to eager: inductor fuses the density into the
  series' reduction (tweedie_logpdf_fixed_p 89 ms against 43 at 2^20).
- The saddle is above the exact log-density by 0.08 to 0.11 per observation;
  its errors above 1 are all at y <= 0.29.
- torch.compile(dynamic=True) was slower than eager from 2^16 up, so the
  compiles are static.

Times are medians of single synchronized calls on cuda:0, launch overhead
included. Nothing here covers other GPUs, CPUs or training quality.

Run:  python benchmarks/tweedie_gpu.py   (~15 min, most of it compiling)
"""

import atexit
import datetime
import math
import os
import shutil
import statistics
import tempfile
import time

# cold compiles: no kernels from an earlier run's on-disk caches
_CACHE = tempfile.mkdtemp(prefix="chebtorch-bench-")
atexit.register(shutil.rmtree, _CACHE, ignore_errors=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = _CACHE
os.environ["TRITON_CACHE_DIR"] = os.path.join(_CACHE, "triton")

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

import chebtorch  # noqa: E402

NS = (2 ** 12, 2 ** 16, 2 ** 20)
P_TIMED = 1.5
PS = (1.2, 1.5, 1.8)
MU, PHI = 2.0, 1.0
REPS = 15


def kernel_nll(p, mu, phi, y):
    return -(y * mu ** (1 - p) / (1 - p) - mu ** (2 - p) / (2 - p)).mean()


def saddle_ll(p, mu, phi, y):
    a = 2 - p
    pos = y > 0
    ys = torch.where(pos, y, torch.ones_like(y))
    half_dev = (torch.where(pos, ys ** a / ((1 - p) * a), torch.zeros_like(y))
                - y * mu ** (1 - p) / (1 - p) + mu ** a / a)
    norm = torch.where(pos, 0.5 * torch.log(2 * math.pi * phi * ys ** p),
                       torch.zeros_like(y))
    return -half_dev / phi - norm


def saddle_nll(p, mu, phi, y):
    return -saddle_ll(p, mu, phi, y).mean()


def exact_nll(p, mu, phi, y):
    return -chebtorch.tweedie_logpdf(p, mu, phi, y).mean()


def exact_nll_python_p(p, mu, phi, y):
    # the tensor p goes unused, so its gradient is None
    return -chebtorch.tweedie_logpdf(P_TIMED, mu, phi, y).mean()


def exact_nll_fixed_p(p, mu, phi, y):
    return -chebtorch.tweedie_logpdf_fixed_p(P_TIMED, mu, phi, y).mean()


LOSSES = {"kernel (neuralforecast)": kernel_nll, "saddle": saddle_nll,
          "chebtorch": exact_nll, "chebtorch, python p": exact_nll_python_p,
          "chebtorch, fixed p": exact_nll_fixed_p}
COMPILED = {name: torch.compile(fn, dynamic=False) for name, fn in LOSSES.items()}
# one compile per (N, dtype, grad mode) and loss: 12, over the default limit of 8
for _k in ("recompile_limit", "cache_size_limit"):
    if hasattr(torch._dynamo.config, _k):
        setattr(torch._dynamo.config, _k, 64)


def call(fn, args, grad):
    if grad:
        loss = fn(*args)
        # the kernel loss never sees phi: it cannot learn it
        torch.autograd.grad(loss, args[:3], allow_unused=True)
    else:
        with torch.no_grad():
            fn(*args)


def timed(fn, args, grad):
    """(first call, median of REPS) in ms; the first includes any compile."""
    out = []
    for _ in range(REPS + 1):
        torch.cuda.synchronize()
        t = time.perf_counter()
        call(fn, args, grad)
        torch.cuda.synchronize()
        out.append(1e3 * (time.perf_counter() - t))
    return out[0], statistics.median(out[1:])


def kernels(fn, args, grad):
    """CUDA kernels one call launches, copies and memsets not counted."""
    call(fn, args, grad)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        call(fn, args, grad)
        torch.cuda.synchronize()
    return sum(e.device_type == torch.autograd.DeviceType.CUDA
               and not e.name.startswith(("Memcpy", "Memset")) for e in prof.events())


def peak_mb(fn, args, grad):
    """GPU memory one call allocates at its peak, beyond what it starts with."""
    call(fn, args, grad)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    call(fn, args, grad)
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2 ** 20


def draws(n, p0, dtype, dev):
    torch.manual_seed(0)
    mu = torch.full((n,), MU, dtype=dtype, device=dev)
    y = chebtorch.Tweedie(mu, PHI, p0).sample()
    p = torch.tensor(p0, dtype=dtype, device=dev, requires_grad=True)
    phi = torch.tensor(PHI, dtype=dtype, device=dev, requires_grad=True)
    mu = torch.full((n,), MU, dtype=dtype, device=dev, requires_grad=True)
    return p, mu, phi, y


def main():
    dev = torch.device("cuda")
    print(f"{torch.cuda.get_device_name(dev)}, torch {torch.__version__}, "
          f"{datetime.date.today()}")
    print(f"ms per call, median of {REPS}, p = {P_TIMED}; +grad = values and "
          f"d/dp (but with a python p), d/dphi, d/dmu (per element)")
    for dtype in (torch.float32, torch.float64):
        for n in NS:
            args = draws(n, P_TIMED, dtype, dev)
            print(f"\n{str(dtype)[6:]}, N = 2^{n.bit_length() - 1}"
                  f"{'eager':>23}{'+grad':>9}{'compiled':>11}{'+grad':>9}")
            for name in LOSSES:
                row, comp = [], []
                for fn in (LOSSES[name], COMPILED[name]):
                    for grad in (False, True):
                        first, med = timed(fn, args, grad)
                        row.append(med)
                        comp.append(first - med)
                line = (f"  {name:<28}" + "".join(f"{v:9.2f}" for v in row[:2])
                        + f"{row[2]:11.2f}{row[3]:9.2f}")
                if name.startswith("chebtorch"):
                    line += (f"   (compile {comp[2] / 1e3:.0f} s, "
                             f"{comp[3] / 1e3:.0f} s with grad)")
                print(line)
            for name in ("chebtorch", "chebtorch, python p", "chebtorch, fixed p"):
                fn = LOSSES[name]
                k = [kernels(fn, args, grad) for grad in (False, True)]
                m = [peak_mb(fn, args, grad) for grad in (False, True)]
                print(f"  {name + ', eager:':28}{k[0]} / {k[1]} CUDA kernels, "
                      f"peak {m[0]:.1f} / {m[1]:.1f} MB")
    print("\nsaddle - exact log-density per observation, float64, N = 2^20")
    for p0 in PS:
        p, mu, phi, y = draws(NS[-1], p0, torch.float64, dev)
        with torch.no_grad():
            d = saddle_ll(p, mu, phi, y) - chebtorch.tweedie_logpdf(p, mu, phi, y)
        pos = y > 0
        at_zero = d[~pos].abs().max().item() if (~pos).any() else 0.0
        print(f"  p = {p0}: zeros {(~pos).double().mean().item():.3f}, "
              f"mean {d.mean().item():+.4f}, "
              f"max |.| {d[pos].abs().max().item():.3g} (y > 0), "
              f"{at_zero:.1g} at y = 0")


if __name__ == "__main__":
    main()
