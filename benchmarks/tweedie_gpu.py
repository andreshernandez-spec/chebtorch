#!/usr/bin/env python3
"""Tweedie negative log-likelihood on the GPU: chebtorch, eager and compiled,
against the saddlepoint approximation and neuralforecast's kernel-only loss.

WHAT THIS MEASURES
------------------
Time per call of the mean negative log-likelihood of N = 2^12, 2^16 and 2^20
observations drawn from the law (mu = 2, phi = 1, p = 1.5), float32 and
float64: values alone (under no_grad), and values with the gradients in p, phi
and a per-element mu, the shape a network's output head has. Three losses,
each eager and under torch.compile (static shapes, one compile per N):

- kernel: -(y mu^(1-p)/(1-p) - mu^(2-p)/(2-p)), what neuralforecast's
  Tweedie DistributionLoss computes. Not a density: no dispersion and no
  normalizing term, so neither p nor phi can be learned through it.
- saddle: the saddlepoint approximation -d(y, mu)/(2 phi) - ln(2 pi phi y^p)/2
  for y > 0, and the exact -lambda at y = 0.
- chebtorch: tweedie_logpdf, the exact density.
- chebtorch, python p: the same with p a python number, the way a model with
  a fixed power calls it. What depends on p alone is kept between calls, and
  there is no d/dp.
- chebtorch, fixed p: tweedie_logpdf_fixed_p at the same python p, the call
  restricted to derivatives in mu and phi.

Also: chebtorch's compile time (first call less a steady one), the CUDA
kernels one eager chebtorch call launches and the peak memory it allocates,
and how far the saddle is from the exact density on N = 2^20 draws at
p = 1.2, 1.5, 1.8 (float64, per observation).

FINDINGS (2026-10-28, RTX 3080 Laptop, torch 2.13; results/tweedie_gpu.txt)
-----------------------------------------------------------------------------
- Eager, chebtorch still costs mostly per call. With a tensor p, float32 is
  5.0 ms for values and 16 ms with gradients at N = 2^12, 6.1 / 19 ms at
  2^16 and 20 / 33 ms at 2^20, from 510 CUDA kernels, 1466 with gradients
  (738 / 1694 at 2^20, where the series takes its lanes 2^16 at a time).
  With a python p, which keeps what depends on p alone and has no d/dp: 2.1
  / 5.0 ms at 2^12, 4.5 / 9.3 at 2^16 and 17 / 24 at 2^20, from 279 and 552
  kernels (507 / 780 at 2^20). Single eager runs move by ~30%.
- tweedie_logpdf_fixed_p at that python p: 0.77 / 1.35 ms at 2^12, 0.99 /
  1.86 at 2^16 and 5.1 / 6.8 at 2^20, from 81 and 110 kernels (154 / 210 at
  2^20). With gradients that is the saddle's time up to 2^16 (1.23 / 2.36
  ms, which also takes d/dp) and 2.9 times it at 2^20; for values alone it
  is 2.4-8 times the saddle. The kernel loss is 0.1-0.2 ms for values and
  0.7-1.4 ms with gradients.
- Peak memory of an eager call is under 81 MB up to 2^16; at 2^20 it is 128
  MB for values and 301 MB with gradients in float32 (273 with a python p,
  136 for tweedie_logpdf_fixed_p), 252 / 597 MB in float64 (541; 236).
- Compiled, float32, up to N = 2^16: 0.13-0.29 ms for values and 1.0-1.7
  ms with gradients with a tensor p, 0.12-0.23 / 0.58-0.61 ms with a python
  p, 0.11-0.20 / 0.37-0.50 ms for tweedie_logpdf_fixed_p (kernel loss
  0.06-0.07 / 0.28-0.38 ms); at 2^20, 1.3 / 5.3, 1.3 / 1.9 and 1.7 / 2.6 ms.
  Cold compiles take 5-18 s for values; with gradients, 43-66 s with a
  tensor p, 11-19 s with a python p and 7-8 s for tweedie_logpdf_fixed_p.
  Sub-millisecond numbers move by ~30% between runs.
- Compiled float64 is slow where the series is a single pass over the lanes:
  8.1 ms for values at 2^16 with a tensor p, 6.9 with a python p and 6.1 for
  tweedie_logpdf_fixed_p, against 6.9, 4.1 and 3.2 eager. Profiling
  (2026-10-26) showed inductor fusing the whole density into the series'
  reduction over the 64 terms. From 2^17 the passes keep it out of the
  first two (37 and 35 ms at 2^20, against 53 and 51 eager), but
  tweedie_logpdf_fixed_p takes its lanes in one pass under compile (89 ms
  at 2^20 against 43 eager). This GPU runs float64 at 1:64.
- The saddlepoint log-density is above the exact one by 0.08 to 0.11 per
  observation on average (p = 1.2, 1.5, 1.8). Its errors above 1 are all at
  small y (up to 9.4, at y <= 0.29), 0.07% to 0.9% of the draws; at y = 0 it
  is exact.
- An earlier run with torch.compile(dynamic=True) was slower than eager from
  N = 2^16 up (the float64 kernel loss 65 ms against 1.6 ms); not
  investigated, so the compiles here are static.

WHAT IT DOES NOT ESTABLISH
--------------------------
Anything about other GPUs (this runs on whatever cuda:0 is; consumer parts run
float64 at 1:64), CPU speed, or training quality: the saddle's error here is
per observation, and what it does to a fitted p is a separate question.
Times are medians of single synchronized calls, so they include launch
overhead, which is most of the eager chebtorch time.

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

# cold compiles on every run: inductor and triton otherwise reuse kernels that
# an earlier run left in their on-disk caches
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
