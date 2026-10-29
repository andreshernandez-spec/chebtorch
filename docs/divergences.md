# Where chebtorch differs from chebax

chebax decides what a function computes (PROJECT.md section 6). This file lists
every place chebtorch does otherwise, with the reason and the chebax side.
Checked against chebax main at a17f1bf, 2026-10-29.

## To go back to chebax

| family | what | here | chebax |
|---|---|---|---|
| tweedie | `tweedie_logpdf_fixed_p`: p a python number, the constants that depend on p alone formed in float64, each series cut to the terms the dtype resolves, partials formed in the forward pass | 51d8423 (#5) | not started: a new public call in chebax, not decided |

## Staying here

| family | what | why | chebax |
|---|---|---|---|
| tweedie | the half-deviance's middle form below the mean (1/e <= y/mu < 3/4) takes x = log1p(delta), where chebax takes log(y/mu) | chebax's choice works around XLA's float64 log1p, up to 128 ulp off there. torch's is within an ulp, and log(y/mu) costs here, since x and delta no longer share a rounding: on `test_half_deviance_by_form`'s draws at a = 0.95, float64 value 3.1e-14 against 5.8e-14, float32 value 1.7e-5 against 2.7e-5 and a-partial 6.2e-4 against 2.1e-3 (relative to \|h_a\| + \|ln mu\| h), y-partial 4.9e-16 against 4.2e-16. The density floors do not move either way. | chebax#38 (ba8105f) |

Everything else that differs is how the code runs in torch (section 6), not
what it computes.

## Copies of chebax

Copies are not differences, but they drift if chebax changes and they do not.

| here | copied from chebax | |
|---|---|---|
| `tools/references.py` | `tests/test_tweedie.py`: `log_phi`, `z_of`, `ref_logpdf`, `_ref_p15` | `dlog_phi0_drho` is chebtorch's own: chebax's test takes d/drho by `mp.diff` |
| `tools/export.py`, `tweedie_fixtures` | `tests/test_tweedie.py`: `RHOS` and `zgrid`, `CASES`, `CONCENTRATED`, and `MORE`, the same random stream | |
| `tests/test_tweedie.py`, `test_half_deviance_by_form` | `tests/test_tweedie.py`: `test_half_deviance_against_mpmath`, the same draws | the reference in decimal, not mpmath |

`tools/export.py` already reads chebax's checkout for the provenance, so it
could import the first two from chebax's tests and drop those copies.
