# Where chebtorch differs from chebax

chebax decides what a function computes (PROJECT.md section 6). This file lists
every place chebtorch does otherwise, with the reason and the chebax side.
Checked against chebax main at e2c60a4, 2026-10-28.

## To go back to chebax

| family | what | here | chebax |
|---|---|---|---|
| tweedie | expm1 form of the half-deviance for 1/4 < \|y/mu - 1\| and \|ln(y/mu)\| <= 1, where chebax uses the three-term form: float32 d/dln phi at p >= 1.3 from 8.5e-6 to 5.3e-6 there | 0035756 (#4) | chebax#37, open |
| tweedie | fixture points just below the split at small p, where chebax's stated floors were exceeded (float64 density 1.1e-12 against 3.3e-13, d/dp 4.7e-11 against 3.6e-12, float32 density 4.5e-4 against 7.4e-6) | c2c0b46 | chebax#37, open |
| tweedie | the half-deviance's closed-form partials checked against finite differences in each form | e9db820 (#4) | chebax#37, open |
| tweedie | `tweedie_logpdf_fixed_p`: p a python number, the constants that depend on p alone formed in float64, each series cut to the terms the dtype resolves, partials formed in the forward pass | 51d8423 (#5) | not started: a new public call in chebax, not decided |

## Staying here

None. Everything else that differs is how the code runs in torch (section 6),
not what it computes.

## Copies of chebax

Copies are not differences, but they drift if chebax changes and they do not.

| here | copied from chebax | |
|---|---|---|
| `tools/references.py` | `tests/test_tweedie.py`: `log_phi`, `z_of`, `ref_logpdf`, `_ref_p15` | `dlog_phi0_drho` is chebtorch's own: chebax's test takes d/drho by `mp.diff` |
| `tools/export.py`, `tweedie_fixtures` | `tests/test_tweedie.py`: `RHOS` and `zgrid`, `CASES`, `CONCENTRATED`, and after chebax#37 `MORE`, the same random stream | |

`tools/export.py` already reads chebax's checkout for the provenance, so it
could import these from chebax's tests and drop both copies.
