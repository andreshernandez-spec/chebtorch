# Copyright (c) 2026, Andres Hernandez
# SPDX-License-Identifier: BSD-3-Clause (https://github.com/andreshernandez-spec/chebtorch)
"""chebtorch: chebax's special functions and distributions for PyTorch,
inference only.

A family is its module (src/chebtorch/<family>.py), its generated tables
(_<family>_data.py) and the shared helpers (cheb_common.py), which import
only math, torch and each other, so a project can copy them as they are. Names
load lazily: `import chebtorch` imports no family until one of its functions
is used.
"""

import importlib

_EXPORTS = {
    "Tweedie": "tweedie",
    "log_wright_bessel0": "tweedie",
    "tweedie_logpdf": "tweedie",
}

__all__ = ["Tweedie", "log_wright_bessel0", "tweedie_logpdf"]
assert sorted(__all__) == sorted(_EXPORTS)


def __getattr__(name):
    family = _EXPORTS.get(name)
    if family is None:
        raise AttributeError(f"module 'chebtorch' has no attribute {name!r}")
    return getattr(importlib.import_module(f"chebtorch.{family}"), name)


def __dir__():
    return sorted(set(globals()) | set(__all__))


try:
    from importlib.metadata import version as _version
    __version__ = _version("chebtorch")
except Exception:  # not installed, e.g. a source checkout on sys.path
    __version__ = "0+unknown"
