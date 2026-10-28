# Coding style

Adapted from chebax's guide for a PyTorch library. It applies to handwritten
library code, tools, tests and benchmarks, and is the review standard for new
work and for code being changed. The vendoring contract is in
[PROJECT.md](../PROJECT.md) section 3.

## Formatting and imports

Use four spaces, ordinary Python spacing, two blank lines between top-level
definitions, and parenthesized multiline expressions (no backslash
continuations). Aim for 88 columns; keep a formula together when splitting it
obscures the mathematics. Put one statement on each line. Group imports as
standard library, third party, then chebtorch, with a blank line between
groups. Inside a family's files the chebtorch group is relative
(`from .cheb_common import cheb`), so the files work under any package name.
Use `np` for NumPy in tests and tools.

Generated `_<family>_data.py` modules are written whole by `tools/export.py`.
Do not format or edit them; their long lines are expected. A change to what
they hold starts in chebax and arrives through a re-export.

There is no configured formatter, linter or type checker; review is manual.

## Names

Use `snake_case` for functions, arguments and modules, `CapWords` for classes,
and uppercase for module constants. A leading underscore marks an internal
implementation detail. Shared modules are named `cheb_<what>.py`: no leading
underscore and nothing generic like `common.py`, so a copy cannot clash with a
host project's files. Their names carry no underscore so that families can
import them, but they are internal: the supported API is what
`chebtorch/__init__.py` exports.

Short mathematical names belong inside a formula: `x`, `a`, `p`, `mu`, `phi`,
`rho`. Define transformed coordinates in the function or module docstring,
with their domains. Use descriptive names for masks, stand-ins and branches:
`in_box`, `valid`, `positive`, `inner`, `mu_s` (a stand-in), `log_w_tail`,
`coefs`. Do not give a short name two meanings in one function.

| Form | Meaning |
| --- | --- |
| `name(params, x)` | A function: tensors in, tensors out, parameters first |
| `log_name(...)` | A logarithm computed without first forming a value that underflows or overflows |
| `Name(params)` | A `torch.distributions.Distribution`, for a family that defines one |

Names follow scipy.special where scipy has the function, chebax otherwise. A
shape parameter that selects the table is uniform per call: a python number,
checked against the box (raising `ValueError`), or a one-element tensor, which
gives nan outside the box.

## Functions and module organization

A public entry point normalizes its inputs (dtype, device, broadcasting, the
per-call parameter), puts stand-ins in the lanes its result will not use,
calls the evaluation helper and selects the result. The numerical work lives
in private helpers whose docstrings state the domain they assume. What depends
on the per-call parameter alone goes in one function wrapped in `Memo` from
`cheb_common.py` (tweedie.py's `_p_terms`), called with what `scalar` returns,
so a python number pays for it once per value, dtype and device.

Order a family module: module docstring, imports and constants, evaluation
helpers, public functions, then the distribution class. Keep an
`autograd.Function` next to the helper whose derivative it supplies. Keep
family-specific mathematics in the family module; move a helper into a shared
module only when two families need it.

## Docstrings and comments

A public entry point states what it computes, parameter and argument domains,
scalar versus broadcasting behavior, dtype and device policy, endpoint and
invalid-input behavior, and which derivatives are available, naming the modes
that are not (forward mode, today). A family's module docstring explains the
factorization, coordinate maps, seams and numerical floor.

Comments explain a choice, a boundary condition or an invariant, in plain
words and short sentences. Put a formula beside the code whose stability
depends on it, and say why a tempting rearrangement loses accuracy. Measured
floors, bars and numerical traps live beside the code or test they describe.

## Numerical expressions

Chebyshev coefficients follow numpy.polynomial: plain `c[0]`, lowest degree
first, the coordinate mapped to `[-1, 1]`. Say whether a number is a
coefficient count or a degree.

Mask unsafe inputs before logarithms, division, powers and exponentials.
Selecting their outputs afterward is not enough: the discarded branch's
backward pass multiplies a zero by its own derivative, and 0 * inf is nan.
Keep the raw input in the branch that is selected, so derivatives at a seam
are not split by a clamp. Keep logarithmic forms until the final result.

Create tensors in the working dtype (`tensor_of(v, work, device)` from
`cheb_common.py`): `torch.as_tensor(0.7).to(torch.float64)` rounds through
float32 first. Library code makes no host syncs (`.item()`, Python branching on
tensor values, `torch.tensor(v, device="cuda")` from a python number, which
waits for all queued GPU work), so calls stay asynchronous and trace under
`torch.compile`. Python numbers and integer tensors promote to the working
floating dtype.

## Tests and review

Test observable contracts against an independent reference: mpmath at 40 dps
or more, or an exact identity. Check values and gradients in float64 and
float32, on CPU and on CUDA when present, including the seam or tail that
motivated a change. Check second derivatives where an `autograd.Function` is
involved, and `torch.compile` when tracing changes.

Reject nonfinite values before aggregating errors: `max(worst, error)` drops a
nan. Test expected infinities and nans separately. State the error metric, the
measured worst case and the bar (about 4x) in the test docstring.

A refactor meant to leave the numbers alone is checked against the previous
version bit for bit, values and gradients. Autograd adds gradient
contributions in graph order, so even sharing a subexpression can move a
derivative whose terms cancel (d/dp near p = 1.05) by thousands of ulps.

Run the full suite before committing a numerical change, and on the torch
floor as well when the change touches what torch provides. A change to a
benchmark means running it again and updating its dated result.
