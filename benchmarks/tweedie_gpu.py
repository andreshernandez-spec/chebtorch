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

Also: chebtorch's compile time (first call less a steady one), the CUDA
kernels one eager chebtorch call launches and the peak memory it allocates,
and how far the saddle is from the exact density on N = 2^20 draws at
p = 1.2, 1.5, 1.8 (float64, per observation).

FINDINGS (2026-10-28, RTX 3080 Laptop, torch 2.13; results/tweedie_gpu.txt)
-----------------------------------------------------------------------------
- Eager, chebtorch still costs mostly per call. With a tensor p, float32 is
  3.5 ms for values and 13 ms with gradients at N = 2^12, 5.2 / 17 ms at
  2^16 and 18 / 29 ms at 2^20, from 431 CUDA kernels, 1341 with gradients
  (659 / 1569 at 2^20, where the series takes its lanes 2^16 at a time).
- With a python p, which keeps what depends on p alone between calls and has
  no d/dp: 1.6 / 4.4 ms at 2^12, 2.7 / 6.5 at 2^16 and 16 / 24 at 2^20, from
  211 and 525 kernels (439 / 753 at 2^20). The kernel loss is 0.1-0.2 ms for
  values and 1-1.2 ms with gradients.
- Peak memory of an eager call is under 80 MB up to 2^16; at 2^20 it is 107
  MB for values and 292 MB with gradients in float32 (264 with a python p),
  211 / 580 MB in float64 (524).
- Compiled, float32: 0.1-0.35 ms for values and 0.8-1.6 ms with gradients up
  to N = 2^16 with a tensor p, 0.1-0.2 / 0.6-0.8 ms with a python p (kernel
  loss 0.07 / 0.3-0.6 ms); at 2^20, 1.1 / 3.4 and 1.5 / 1.9 ms. Cold compiles
  take 8-21 s for values; with gradients, 42-71 s with a tensor p and 14-15 s
  with a python p, which has no d/dp. Sub-millisecond numbers move by ~30%
  between runs.
- Compiled float64 is slow where the series is a single pass over the lanes:
  13 ms for values at 2^16 with a tensor p, against 6 eager. Profiling shows
  inductor fusing the whole density into the series' reduction over the 64
  terms; from 2^17 the passes keep it out (27 ms at 2^20). A python p is 4.8
  ms there but slower at 2^20, 71 ms for values and 129 with gradients against
  27 and 69; under compile nothing is kept, so this is the graph itself (not
  investigated). Values at 2^12 came out at 1 ms in some cold compiles and
  9.5-15 ms in others, for either kind of p. This GPU runs float64 at 1:64.
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

Run:  python benchmarks/tweedie_gpu.py   (~12 min, most of it compiling)
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


LOSSES = {"kernel (neuralforecast)": kernel_nll, "saddle": saddle_nll,
          "chebtorch": exact_nll, "chebtorch, python p": exact_nll_python_p}
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
            for name in ("chebtorch", "chebtorch, python p"):
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
