# chebtorch: chebax's special functions for PyTorch, inference only

Plan written 2026-10-26.

## 1. Goal

PyTorch functions for the special functions and distributions chebax tabulates:
branchless, fast on GPU, differentiable in every floating argument including the
shape parameters, accurate to the floors chebax measures. **Any one family can be
adopted by another project by copying a few files and changing nothing.**

chebax stays the source of truth for everything that is not inference: the mpmath
generators, the tables, `fit`, the byte-for-byte regeneration checks. chebtorch
holds copies of the tables and the evaluation code, nothing else.

### Non-goals

- Fitting, generation, mpmath, jax. No runtime dependency on chebax.
- Per-element shape parameters in v1. Like chebax, a shape parameter that selects
  the table (Tweedie p, Bessel order, betainc a and b, gammainc a) is uniform per
  call. Section 4.4 has the per-element option for later.
- Wrapping scipy or calling C++. Everything is torch ops, so one pure-Python wheel
  serves every platform, CUDA version and torch build.

## 2. Why: the gaps are real, and copied files are easier to take than a dependency

Checked 2026-10-26 on torch 2.13:

| PyTorch gap | chebax has it |
|---|---|
| `torch.special.gammainc`: "the derivative for 'igamma: input' is not implemented" (no gradient in the shape a) | `gammainc_fn` with d/da |
| no `betainc` (PR #132135 under review since 2024-08, users still asking in 2026) | `betainc_fn`, `log_betainc_fn` |
| no real-order `kv`, `iv`, `jv`, `yv` (tracking issue #76324) | the Bessel families |
| `torch.distributions`: no `icdf` for Gamma, Beta, StudentT, VonMises; no `cdf` for Beta, StudentT, VonMises | the quantiles, `stdtr`, `vonmises_cdf/icdf` |
| no Tweedie distribution (#172059; PR #171705 closed stale) | `tweedie_logpdf` |

A project that needs one of these, say one distribution for a loss, can take a
couple of copied files much more easily than a new dependency. That is what shapes
the design (section 3).

## 3. The vendoring contract

This is the design's centre. Every vendorable family file obeys it, and CI checks it.

1. **A family is a few files:** its module `src/chebtorch/<family>.py`, its
   generated tables `_<family>_data.py`, and the shared modules it imports.
2. **They import `math`, `torch` and each other only**, each other relatively
   (`from .cheb_common import cheb`), so they work under any package name. No
   numpy, no data files, no package-level state.
3. **The weights are Python literals** in the data module, which `tools/export.py`
   writes whole. Repr of a float64 round-trips exactly, a literal diff is
   reviewable, and a setuptools project needs no package-data change to ship a
   `.py` file. Measured cold compile: 3 ms for
   Tweedie's table (48 KB of source), 14 ms for besselk's (227 KB), 100 ms for
   betainc's small panels (1.1 MB).
4. **Shared helpers live in shared modules.** Clenshaw, the table cache, dtype and
   device resolution and the per-call parameter check are `cheb_common.py`, which
   every family imports. Helpers several families share (the gamma pieces of
   gammainc, betainc and stdtr; the inverse-CDF solver) get their own `cheb_*.py`
   module when they appear: no leading underscore and no generic name, so a copy
   cannot clash with a host's own modules. Decided 2026-10-26 (Andres): a project
   that accepts one copied file accepts two or three, and one copy of each helper
   beats one per family.
5. **Provenance** is in the data module: PROVENANCE names chebax's `git describe`
   and the last commit that changed the table, the table module, its generator
   sha256 and mpmath version (both from the table's META), and the sha256 of the
   data, which a test recomputes. Every file carries a copyright and SPDX line.
6. **The copy test.** CI copies each family's files (its module and the modules it
   imports) into an empty temporary package, the module under another name
   (`hostpkg/losses/tweedie_loss.py`), imports it there with chebtorch unimportable
   and runs its fixtures. That is the "copy the files, change nothing" promise,
   executed.

### What is vendorable

Budget: about 300 KB of source per data module.

| family | table, binary f64 | vendorable |
|---|---|---|
| tweedie | 16 KB | yes |
| gammainc (+ Temme tables) | 33 KB | yes |
| besselj (+ ext), besseli | 46 KB, 48 KB | yes |
| besselk, bessely | 76 KB, 98 KB | yes |
| stdtr, vonmises | 103 KB, 95 KB | yes |
| betainc, a and b to 10 | 378 KB | borderline (1.1 MB of source, 100 ms cold) |
| betainc wide panels, a and b to 100 | 4.7 MB | library only |
| hyp1f1 | 1.7 MB | library only |

Library-only families keep their weights as raw little-endian float64 package data
in chebtorch's own wheel, loaded lazily with `torch.frombuffer`. They obey every
other rule.

### Anatomy of a family

```python
# tweedie.py
"""Tweedie log-density for 1.05 <= p <= 1.95 ... copy it with _tweedie_data.py
and cheb_common.py into one package directory.

<what it computes, domain, accuracy floors, how p, mu, phi broadcast>
"""
import math
import torch

from .cheb_common import Tables, cheb, device_of, scalar, work_dtype
from ._tweedie_data import A1, DATA, N_TERMS, P_HI, P_LO, SHAPES, ...

__all__ = ["Tweedie", "log_wright_bessel0", "tweedie_logpdf"]
_tables = Tables(SHAPES, DATA)                  # tensors cached per (dtype, device)

def tweedie_logpdf(p, mu, phi, y): ...          # public, functional
def log_wright_bessel0(rho, z): ...
class Tweedie(torch.distributions.Distribution): ...   # log_prob, sample, mean, variance

# _tweedie_data.py, written by tools/export.py from chebax; never edited
PROVENANCE = {"chebax": "v0.3.0-21-g45ff338", "chebax_commit": "45ff3385...", ...}
P_LO = 1.05
...
SHAPES = {"TABLE_TAIL": (32, 64)}
DATA = {"TABLE_TAIL": (0.98969..., ...)}
```

## 4. Runtime conventions

### 4.1 API

- Functional first: tensors in, tensors out, elementwise arguments broadcast.
- Argument order is scipy.special's (parameters first, evaluation point last:
  `kv(v, z)`, `betainc(a, b, x)`, `gammainc(a, x)`), which chebax shares.
- Names: scipy's where scipy has the function (`kv`, `kve`, `iv`, `ive`, `jv`, `yv`,
  `betainc`, `gammainc`, `gammaincc`, `gammaincinv`, `betaincinv`, `stdtr`,
  `stdtrit`), chebax's for what scipy lacks (`log_kv`, `log_betainc`,
  `tweedie_logpdf`, `log_wright_bessel0`, `vonmises_cdf`). Open question 1.
- A `torch.distributions.Distribution` subclass in the same file where a family
  defines a distribution (Tweedie; later Gamma/Beta/StudentT with cdf and icdf).

### 4.2 dtype, device, errors

- Computes in the input's dtype: float64 or float32. float16 and bfloat16 go through
  float32 and come back. Tables are cached per (dtype, device) on first use.
- No host syncs: no `.item()`, no Python branching on tensor values, so it runs
  asynchronously on GPU and traces under `torch.compile`.
- Out of the box or invalid input gives nan, as in chebax. Python-number parameters
  are also checked eagerly and raise `ValueError` with the box in the message.

### 4.3 Autograd and compile

- Every floating argument is differentiable. Chebyshev evaluation is polynomial
  arithmetic, so plain autograd through Clenshaw is exact.
- A custom `torch.autograd.Function` where the analytic backward is cheaper or the
  only correct one: the Tweedie direct sum (one-pass weighted derivatives, O(N)
  memory), and the inverse CDFs (implicit function theorem; never differentiate the
  fixed-iteration solver).
- Measured (`benchmarks/tweedie_gpu.py`, 2026-10-26, RTX 3080 Laptop, float32):
  eager is launch-bound, ~10 ms for values and 20-24 ms with gradients at any
  N up to 2^16 (1279 and 2311 CUDA kernels per call), against 0.1 / 1 ms for
  neuralforecast's kernel-only loss; compiled, 0.3-0.5 / 1.3-1.7 ms, after
  23-124 s of cold compile per shape. float64 backward gains nothing from
  compiling (90 ms against 88 eager at 2^20). Fewer, larger kernels (the
  series as one tensor over its 64 terms) are the eager fix and come next: a
  training loop that does not compile pays the eager cost every step.
  `torch.library.custom_op` with registered autograd is M2.

### 4.4 Per-element shape parameters (later, if a consumer needs them)

Evaluate the 2-D series per element instead of reconstructing once per call. Cost is
the product of the two degrees per point (Tweedie: 64 x 32 = 2k multiply-adds),
which a GPU absorbs for small tables. Offered as `per_element=True` only where
measured worthwhile.

## 5. Provenance pipeline

- `tools/export.py`, run in a dev env that has chebax (and so jax): reads chebax's
  table modules, writes each family's data module with its provenance, and writes
  fixtures. It is the only code that imports chebax.
- Fixtures (`tests/fixtures/<family>.json`): mpmath references at 40 dps for values
  and gradients, generated with chebax's own reference helpers, so chebtorch's CI
  needs neither mpmath nor jax.
- CI on every push (CPU, torch floor and latest, float64 and float32): fixtures to
  the bars, data hashes, the import rules, the copy test, torch.compile smoke test.
- Parity job (weekly or manual): install chebax and jax, compare chebtorch with
  chebax on random inputs. They are the same arithmetic, so differences should sit at
  a few ulp.
- A table change in chebax means re-export, then a chebtorch release that records
  the new chebax version in every header.
- GPU speed is measured by hand (laptop or RunPod), as in chebax; claims cite the
  benchmark script and its date.

## 6. Repository layout

```
PROJECT.md, CLAUDE.md, README.md, LICENSE (BSD-3)
docs/coding-style.md    the review standard, adapted from chebax's
pyproject.toml          pure python, requires torch only
src/chebtorch/
    __init__.py         lazy re-exports (PEP 562), so `import chebtorch` loads no tables
    cheb_common.py      helpers every family imports (section 3)
    tweedie.py          a family's code
    _tweedie_data.py    its tables, written by tools/export.py
    gammainc.py, _gammainc_data.py, besselk.py, ...
    _weights/           raw float64 blobs for the library-only families
tools/export.py         chebax -> data modules and fixtures
tools/references.py     mpmath references for the fixtures
tests/                  fixtures, contract (copy) test, per-family tests
benchmarks/             GPU scripts; results checked in, dated
```

## 7. Milestones

| ID | Milestone | Acceptance |
|---|---|---|
| M0 | Bootstrap: repo, CLAUDE.md, export tool, fixture format, shared helpers module, CI with the copy test | A template family passes the copy test from an empty package; export reproduces a chebax table bit for bit |
| M1 | Tweedie: `tweedie_logpdf`, `log_wright_bessel0`, `Tweedie` distribution, custom backward for the series | Fixtures at measured floors, no worse than chebax on the same points (they include the band near the split at p = 1.05 that chebax's tests do not reach); parity with chebax; copy test; GPU benchmark eager and compiled against the saddle-point baseline |
| M2 | gammainc family (`gammainc`, `gammaincc`, log forms, d/da) and `torch.library.custom_op` for compiled backward | Fixtures; gradient in a where torch raises; compiled backward fused, measured |
| M3 | Bessel `kv`, `kve`, `log_kv`, `iv`, `ive`, ratio; a general Matérn kernel example | Fixtures; Matérn nu learned by gradient (chebax's M5 demo, in torch) |
| M4 | `betainc` (vendorable to a, b <= 10, wide panels library-only), `stdtr` | Fixtures; Beta and StudentT cdf |
| M5 | Inverse CDFs (`gammaincinv`, `gammainccinv`, `betaincinv`, `stdtrit`), implicit-function backward | Fixtures (two-sided quantile metric); icdf for Gamma, Beta, StudentT |
| M6 | The rest by demand: `jv`, `yv`, spherical, von Mises cdf/icdf, Lambert W, hyp1f1 | Per family |

PyPI release after M1 is Andres's action, as for chebax.

## 8. Risks

| risk | mitigation |
|---|---|
| Reviewers balk at a table of literals they cannot check | provenance block, hash test, fixtures they can run, small files |
| chebtorch drifts from chebax | export is the only path for tables; weekly parity job against chebax main; chebax commit in every data module |
| Copies go stale downstream | the header names the source and release; chebtorch keeps a changelog per family |
| torch.compile graph breaks from custom Functions | `torch.library.custom_op` (M2); eager stays correct meanwhile |
| Consumers need per-element shape parameters | section 4.4, measured before offering |
| Two runtimes to maintain | inference only, no generator code in chebtorch; scope by demand |

## 9. Open questions

1. Function names: scipy's where they exist (proposed), or chebax's throughout.
2. float32 tables for copies (half the size, same accuracy at f32) or float64 only.

Settled 2026-10-26: the name (chebtorch), the repository
(andreshernandez-spec/chebtorch), and the floors (python 3.10 with torch 2.4, which
CI tests).
