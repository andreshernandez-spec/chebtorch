# CLAUDE.md: operating instructions for chebtorch

Read `PROJECT.md` first (design, the vendoring contract, milestones), and
follow `docs/coding-style.md` (adapted from chebax's) for formatting, naming,
docstrings, numerics and tests. Read `../CLAUDE.md` before anything else: the
OSS git identity, the `gh` rules, never pushing to upstream projects, the
`open-source` conda env, and the writing style all apply here. PyPI releases
are Andres's action.

## What this repo is, and is not

chebtorch evaluates chebax's tables in PyTorch. It never generates them: the
mpmath generators, `fit`, and the byte-for-byte regeneration checks live in
chebax, which is the source of truth for every coefficient. A table change starts
in chebax and reaches chebtorch only through `tools/export.py`.

## The vendoring contract (PROJECT.md section 3); tests/test_contract.py enforces it

- A family is its module `src/chebtorch/<family>.py`, its generated tables
  `_<family>_data.py`, and the shared modules it imports. Helpers used by more
  than one family go in a shared module (`cheb_common.py` for all of them),
  never a copy per family: a project that accepts one copied file accepts two
  or three (Andres, 2026-10-26).
- Shared modules are named `cheb_<what>.py`: no leading underscore and nothing
  generic like `common.py`, so they cannot clash with a host project's files.
- Those files import only `math`, `torch` and each other, relatively
  (`from .cheb_common import cheb`), so they work under any package name.
- `_<family>_data.py` is written by `tools/export.py`; never edit it by hand.
- Copied alone into another package (the family module renamed), a family's
  files must work with chebtorch unimportable. Its docstring names the files.

## Commands

```bash
python -m pip install -e ".[test]" --no-build-isolation   # into the open-source env
python -m pytest -q                       # ~1.5 min; GPU tests run when CUDA is present,
                                          #  parity tests when chebax imports
python tools/export.py                    # every _<family>_data.py, from the installed chebax
python tools/export.py --fixtures         # also the mpmath fixtures (~40 s)
python benchmarks/tweedie_gpu.py          # GPU timings, ~10 min; results/ holds dated output
```

The export needs chebax installed editable from its git checkout
(`pip install -e ../chebax`) and mpmath. It records the last chebax commit that
changed each table, so commit chebax first, or the provenance says `-dirty` and
a test fails. The weekly CI parity job compares tables and values with chebax
main.

## Rules

- **Measure floors before setting bars.** Bars sit at about 4x the measured worst
  case, CPU and GPU, float64 and float32, and the test docstring records both
  numbers. Floors here can differ from chebax's: torch's lgamma, pow and exp are
  not XLA's.
- **Create tensors in the working dtype.** `torch.as_tensor(0.7).to(torch.float64)`
  rounds 0.7 through float32 first; always pass `dtype=` at creation.
- **Performance claims cite a script in `benchmarks/` and its dated result.** Pass
  the data to a jitted or compiled function as an argument: a closed-over tensor is
  a constant the compiler can fold.
- Work aimed at other projects follows `../CLAUDE.md` section 2: code on a
  branch in a shared clone, prose in `drafts/`, Andres submits. `drafts/` is
  local only (in `.git/info/exclude`), and so is anything about where chebtorch
  should be adopted: outreach planning never goes into this public repo.
