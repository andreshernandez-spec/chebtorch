# Copyright (c) 2026, Andres Hernandez
# SPDX-License-Identifier: BSD-3-Clause
# From chebtorch: https://github.com/andreshernandez-spec/chebtorch
"""chebax's special functions and distributions for PyTorch, inference only. A
family is its module, its generated _<family>_data.py and cheb_common.py, which
a project can copy as they are. Families load on first use.
"""

import importlib

_EXPORTS = {
    "Tweedie": "tweedie",
    "log_wright_bessel0": "tweedie",
    "tweedie_logpdf": "tweedie",
    "tweedie_logpdf_fixed_p": "tweedie_fixed_p",
}

__all__ = ["Tweedie", "log_wright_bessel0", "tweedie_logpdf", "tweedie_logpdf_fixed_p"]
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
