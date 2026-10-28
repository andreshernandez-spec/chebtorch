# chebtorch

[chebax](https://github.com/andreshernandez-spec/chebax)'s special functions and
distributions for PyTorch, inference only: branchless, fast on GPU, differentiable
in every argument including the shape parameters. The tables come from chebax,
which generates and verifies them against mpmath; chebtorch only evaluates them.

```python
import torch
import chebtorch

y = torch.tensor([0.0, 0.4, 2.5, 9.0], dtype=torch.float64)
p = torch.tensor(1.4, dtype=torch.float64, requires_grad=True)
ll = chebtorch.tweedie_logpdf(p, 2.0, 0.8, y).sum()   # mass at 0 included
ll.backward()                                          # d/dp, so the power can be learned

d = chebtorch.Tweedie(mu=torch.full((3,), 2.0), phi=0.8, power=1.4)
d.log_prob(torch.tensor([0.0, 1.0, 3.0])), d.sample((5,))
```

A power that is not learned is cheaper as a python number, as in `Tweedie`
above: what depends on it alone is computed at its first call and kept.

## What's in

| family | functions | box | float64 accuracy |
|---|---|---|---|
| Tweedie | `tweedie_logpdf(p, mu, phi, y)`, `Tweedie(mu, phi, power)`, `log_wright_bessel0(rho, z)` | p in [1.05, 1.95], uniform per call | density 9.9e-13, d/dp 6.1e-11, d/dln mu 4.0e-16, ln W 7.3e-16 |

Accuracy is the measured worst case against mpmath at 40 digits,
|err| / max(1, |ref|). Each family's test file records its floors and bars,
float32 included (for Tweedie, float32 is weakest below p = 1.1: density 4e-4).

## Copying one family into your project

A family is its module, `src/chebtorch/<family>.py`, plus the modules it imports
from beside it: its tables, `_<family>_data.py`, generated from chebax with their
provenance, and the helpers every family shares, `cheb_common.py`. They import
only `math`, `torch` and each other, so for Tweedie you copy `tweedie.py`,
`_tweedie_data.py` and `cheb_common.py` into one package directory and change
nothing. The family module can take any name; the other two keep theirs (a new
directory with an empty `__init__.py` keeps them out of your namespace). Each
family's docstring names its files, and CI runs exactly this copy.

## License

BSD-3-Clause. Every file in `src/chebtorch/` carries its copyright and SPDX
line; keep them in a copy.
