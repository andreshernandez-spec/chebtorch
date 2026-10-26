"""chebtorch against chebax on random inputs over the box, float64.

Both evaluate the same tables with the same formulas, so they agree to within
their accuracy floors (tests/test_tweedie.py): a wrong table, constant or
branch shows up orders of magnitude above the bars. Metrics as in
tests/test_tweedie.py; d/dp is compared as the sum over the batch, normalized
by sum max(1, |d/dp|), since chebtorch returns only the sum for a scalar p.

Skipped unless chebax and jax import; CI's weekly parity job installs chebax
from main.

Measured worst difference (2026-10-26, chebax main at dcdb931 on jax 0.11, torch 2.13,
CPU; bars about 4x): density 8.0e-13, d/dp 2.0e-15, d/dln mu 4.1e-13,
d/dln phi 8.4e-12, d/dln y 7.5e-12; ln W 7.1e-15, d/dz 1.1e-14, d/drho 5.1e-16.
Against mpmath at the worst points, chebtorch is the closer of the two for the
density, d/dln mu and d/dln phi, chebax for d/dln y; tests/fixtures holds
those points.
"""

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
chebax = pytest.importorskip("chebax")
jax.config.update("jax_enable_x64", True)

from chebtorch import log_wright_bessel0, tweedie_logpdf  # noqa: E402

N = 4000
PS = (1.05, 1.07, 1.15, 1.3, 1.5, 1.7, 1.9, 1.95)
BARS = {"f": 3.2e-12, "dp": 8e-15, "dlnmu": 1.6e-12, "dlnphi": 3.4e-11, "dlny": 3e-11,
        "w": 2.8e-14, "dz": 4.4e-14, "drho": 2e-15}


def _draws(rng, p):
    """mu and phi log-uniform per element, y drawn from the law; a quarter of
    y log-uniform over [1e-8, 1e4] instead, to reach the far tails."""
    mu = np.exp(rng.uniform(np.log(0.05), np.log(50.0), N))
    phi = np.exp(rng.uniform(np.log(0.02), np.log(5.0), N))
    n = rng.poisson(mu ** (2 - p) / (phi * (2 - p)))
    total = rng.gamma(np.maximum(n, 1) * (2 - p) / (p - 1))
    y = np.where(n > 0, total * phi * (p - 1) * mu ** (p - 1), 0.0)
    y[: N // 4] = np.exp(rng.uniform(np.log(1e-8), np.log(1e4), N // 4))
    return mu, phi, y


def _rel(got, ref):
    return np.abs(got - ref) / np.maximum(1.0, np.abs(ref))


def tweedie_differences(p, seed):
    mu, phi, y = _draws(np.random.default_rng(seed), p)
    ref = np.asarray(chebax.tweedie_logpdf(p, mu, phi, y))
    grads = jax.vmap(jax.grad(chebax.tweedie_logpdf, argnums=(0, 1, 2, 3)),
                     in_axes=(None, 0, 0, 0))(p, mu, phi, y)
    gp, gm, gf, gy = (np.asarray(g) for g in grads)
    tp = torch.tensor(p, dtype=torch.float64, requires_grad=True)
    tm, tf, ty = (torch.tensor(a, requires_grad=True) for a in (mu, phi, y))
    got = tweedie_logpdf(tp, tm, tf, ty)
    got.sum().backward()
    # nonfinite values first: the max below would not show them
    for a in (ref, gp, gm, gf, got.detach().numpy(), tm.grad.numpy(), tf.grad.numpy()):
        assert np.isfinite(a).all()
    assert np.isfinite(tp.grad.item())
    pos = y > 0
    assert np.isfinite(gy[pos]).all() and np.isfinite(ty.grad.numpy()[pos]).all()
    return {
        "f": _rel(got.detach().numpy(), ref).max(),
        "dp": abs(tp.grad.item() - gp.sum()) / np.maximum(1.0, np.abs(gp)).sum(),
        "dlnmu": _rel(mu * tm.grad.numpy(), mu * gm).max(),
        "dlnphi": _rel(phi * tf.grad.numpy(), phi * gf).max(),
        "dlny": _rel(y * ty.grad.numpy(), y * gy)[pos].max(),
    }


def wright_differences(seed):
    rng = np.random.default_rng(seed)
    rho = float(np.exp(rng.uniform(np.log(1 / 19), np.log(19.0))))
    z = np.exp(rng.uniform(np.log(1e-12), np.log(1e200), N))
    ref = np.asarray(chebax.log_wright_bessel0(rho, z))
    grads = jax.vmap(jax.grad(chebax.log_wright_bessel0, argnums=(0, 1)),
                     in_axes=(None, 0))(rho, z)
    gr, gz = (np.asarray(g) for g in grads)
    tr = torch.tensor(rho, dtype=torch.float64, requires_grad=True)
    tz = torch.tensor(z, requires_grad=True)
    got = log_wright_bessel0(tr, tz)
    got.sum().backward()
    for a in (ref, gr, gz, got.detach().numpy(), tz.grad.numpy()):
        assert np.isfinite(a).all()
    assert np.isfinite(tr.grad.item())
    return {
        "w": _rel(got.detach().numpy(), ref).max(),
        "dz": (np.abs(tz.grad.numpy() - gz) / np.abs(gz)).max(),
        "drho": abs(tr.grad.item() - gr.sum()) / np.maximum(1.0, np.abs(gr)).sum(),
    }


@pytest.mark.parametrize("p", PS)
def test_tweedie_logpdf_matches_chebax(p):
    for k, v in tweedie_differences(p, seed=round(100 * p)).items():
        assert v <= BARS[k], (k, v)


@pytest.mark.parametrize("seed", range(12))
def test_log_wright_bessel0_matches_chebax(seed):
    for k, v in wright_differences(seed).items():
        assert v <= BARS[k], (k, v)
