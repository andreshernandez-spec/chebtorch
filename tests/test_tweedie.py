"""Tweedie densities and ln Phi(rho, 0; z) against mpmath fixtures.

Fixtures (tests/fixtures/tweedie.json) are mpmath at 40 digits, written by
tools/export.py --fixtures: a grid over the box, the region just below the
split at small p where the errors peak, and random points. Both functions are
logarithms, so the error is |err| / max(1, |ref|): absolute in the log where it
is small, relative where it is large. Gradients in mu, phi and y are taken in
their logarithms (x df/dx): df/dphi is 1/phi times a cancellation that does
not shrink with phi, so a plain metric grows as 1/phi near a zero of it.

float64 is checked against mpmath. float32 is checked against float64 at the
same float32 inputs, which separates the arithmetic from the rounding of the
inputs (at p = 1.05 df/dp is in the thousands, so rounding p alone moves the
density by ~1e-4).

Measured worst (2026-10-26, torch 2.13 on CPU and an RTX 3080 Laptop; torch
2.4.1, the CI floor, gives the same on CPU; the bars are about 4x):

float64   ln W 7.3e-16, d/dz 1.0e-14, d/drho 1.2e-14; density 7.6e-13,
          d/dp 4.3e-11, d/dln mu 8.7e-14, d/dln phi 7.0e-12, d/dln y 5.8e-12.
          All at p = 1.05. Below the split lgamma(rho n) reaches ~4000 in the
          series and eps |lgamma| survives into an O(1) log-density (d/dp also
          carries drho/dp = -400 there); d/dln mu is worst in the tail with y
          near mu and mu/phi ~ 2000, where the deviance cancels.
float32   ln W 5.6e-7, d/dz 1.7e-6, d/drho 4.4e-6 (z within float32's range),
          and by p, the same cancellation at float32's eps:
                      density  d/dp     d/dln mu  d/dln phi  d/dln y
          p < 1.1     3.1e-4   2.0e-2   1.9e-4    3.8e-3     2.0e-3
          p < 1.3     1.6e-5   4.5e-4   2.5e-6    5.2e-5     3.5e-5
          p >= 1.3    1.5e-6   2.1e-5   1.9e-6    4.3e-6     4.4e-6
"""

import json
import math
import os
import pathlib
import subprocess
import sys

import pytest
import torch

from chebtorch.cheb_common import cheb
from chebtorch.tweedie import (A1, P_HI, P_LO, Tweedie, _split, log_wright_bessel0,
                               tweedie_logpdf)

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "tweedie.json"
FX = json.loads(FIXTURES.read_text())
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
BAR_W64 = {"w": 3e-15, "dz": 4e-14, "drho": 5e-14}
BAR_W32 = {"w": 2.5e-6, "dz": 7e-6, "drho": 1.8e-5}
# (density, (d/dp, d/dln mu, d/dln phi, d/dln y)); float32 by band of p
BAR_F64 = (3e-12, (1.7e-10, 3.5e-13, 2.8e-11, 2.4e-11))
BAR_F32 = ((1.1, 1.2e-3, (8e-2, 8e-4, 1.5e-2, 8.5e-3)),
           (1.3, 6.5e-5, (1.8e-3, 1e-5, 2.4e-4, 1.2e-4)),
           (2.0, 6e-6, (8.5e-5, 8e-6, 1.7e-5, 1.8e-5)))
ARGS = ("p", "mu", "phi", "y")


def _t(v, dtype=torch.float64):
    return torch.tensor(v, dtype=dtype)


def _rel(got, ref):
    return abs(got - ref) / max(1.0, abs(ref))


def _eval(fn, xs, dtype, device):
    """fn at xs in dtype: the value and the gradient in each argument, all
    checked finite first (max() would drop a nan from the worst case)."""
    args = [torch.tensor(x, dtype=dtype, device=device, requires_grad=True)
            for x in xs]
    v = fn(*args)
    v.backward()
    out = v.item(), [a.grad.item() for a in args]
    assert all(map(math.isfinite, [out[0], *out[1]])), (fn.__name__, xs, dtype, out)
    return out


def _to32(name, x):
    """x rounded to float32; p to the nearest float32 inside the box, since
    1.05 and 1.95 round outward."""
    t = torch.tensor(x, dtype=torch.float32)
    if name == "p" and not P_LO <= t.item() <= P_HI:
        t = torch.nextafter(t, torch.tensor(1.5))
    return t.item()


def _grad_errors(xs, got, ref):
    # d/dp as is, mu, phi and y in their logs; d/dy has no reference at y = 0
    return [0.0 if r is None
            else _rel(g, r) if i == 0 else _rel(xs[i] * g, xs[i] * r)
            for i, (g, r) in enumerate(zip(got, ref))]


@pytest.mark.parametrize("device", DEVICES)
def test_log_wright_bessel0_float64(device):
    worst = dict.fromkeys(BAR_W64, 0.0)
    for c in FX["log_wright_bessel0"]:
        xs = (c["rho"], c["z"])
        v, (gr, gz) = _eval(log_wright_bessel0, xs, torch.float64, device)
        worst["w"] = max(worst["w"], _rel(v, c["value"]))
        worst["dz"] = max(worst["dz"], abs(gz - c["dz"]) / abs(c["dz"]))
        worst["drho"] = max(worst["drho"], _rel(gr, c["drho"]))
    for k, v in worst.items():
        assert v <= BAR_W64[k], (k, v)


@pytest.mark.parametrize("device", DEVICES)
def test_log_wright_bessel0_float32(device):
    worst = dict.fromkeys(BAR_W32, 0.0)
    for c in FX["log_wright_bessel0"]:
        if not 1e-30 <= c["z"] <= 1e30:
            continue
        xs = (_to32("rho", c["rho"]), _to32("z", c["z"]))
        v, (gr, gz) = _eval(log_wright_bessel0, xs, torch.float32, device)
        r, (rr, rz) = _eval(log_wright_bessel0, xs, torch.float64, device)
        worst["w"] = max(worst["w"], _rel(v, r))
        worst["dz"] = max(worst["dz"], abs(gz - rz) / abs(rz))
        worst["drho"] = max(worst["drho"], _rel(gr, rr))
    for k, v in worst.items():
        assert v <= BAR_W32[k], (k, v)


@pytest.mark.parametrize("device", DEVICES)
def test_tweedie_logpdf_float64(device):
    worst_f, worst_g = 0.0, [0.0] * 4
    for c in FX["tweedie_logpdf"]:
        xs = [c[k] for k in ARGS]
        v, g = _eval(tweedie_logpdf, xs, torch.float64, device)
        worst_f = max(worst_f, _rel(v, c["value"]))
        if "grad" in c:
            for i, e in enumerate(_grad_errors(xs, g, c["grad"])):
                worst_g[i] = max(worst_g[i], e)
    assert worst_f <= BAR_F64[0], worst_f
    for got, bar, name in zip(worst_g, BAR_F64[1], ARGS):
        assert got <= bar, (name, got)


@pytest.mark.parametrize("device", DEVICES)
def test_tweedie_logpdf_float32(device):
    for lo, (hi, bar_f, bar_g) in zip((1.0, 1.1, 1.3), BAR_F32):
        worst_f, worst_g = 0.0, [0.0] * 4
        for c in FX["tweedie_logpdf"]:
            if not lo <= c["p"] < hi:
                continue
            xs = [_to32(k, c[k]) for k in ARGS]
            v, g = _eval(tweedie_logpdf, xs, torch.float32, device)
            r, gr = _eval(tweedie_logpdf, xs, torch.float64, device)
            worst_f = max(worst_f, _rel(v, r))
            for i, e in enumerate(_grad_errors(xs, g, gr)):
                worst_g[i] = max(worst_g[i], e)
        assert worst_f <= bar_f, (lo, worst_f)
        for got, bar, name in zip(worst_g, bar_g, ARGS):
            assert got <= bar, (lo, name, got)


def test_table_limit_is_c1():
    # T -> c1(rho) = -(2 rho + 1)(rho + 2)/(24 rho) as Z -> inf: the end no
    # fixture reaches, in closed form
    for t in torch.linspace(-1, 1, 41, dtype=torch.float64):
        r = torch.exp(t * A1)
        coefs = _split(r)[2]
        c1 = -(2 * r + 1) * (r + 2) / (24 * r)
        assert abs(cheb(_t(-1.0), coefs) - c1) <= 4e-15


def test_huge_arguments_follow_the_saddle():
    for rho, z in ((1 / 19, 1e300), (1.0, 1e300), (19.0, 1e300), (0.3, 1e150)):
        got = log_wright_bessel0(rho, _t(z)).item()
        e = 1 / (1 + rho)
        t0 = rho ** e * z ** e                     # pow, as the implementation does
        big_z = (1 + rho) * t0 / rho
        lead = big_z + 0.5 * math.log(t0) - 0.5 * math.log(2 * math.pi * (1 + rho))
        assert math.isfinite(got), (rho, z)
        assert abs(got - lead) / lead <= 1.5e-15, (rho, got, lead)


def test_edges_and_invalid_arguments():
    assert log_wright_bessel0(1.0, _t(0.0)).item() == -math.inf
    assert math.isnan(log_wright_bessel0(1.0, _t(-1.0)).item())
    for rho in (1 / 19 * 0.999, 19.01):
        assert math.isnan(log_wright_bessel0(_t(rho), 2.0).item())
        with pytest.raises(ValueError, match="covers rho"):
            log_wright_bessel0(rho, _t(2.0))
    p, mu, phi = 1.4, 2.0, 0.7
    lam = mu ** (2 - p) / (phi * (2 - p))
    assert tweedie_logpdf(p, mu, phi, _t(0.0)).item() == pytest.approx(-lam, rel=1e-15)
    for bad in ((p, 0.0, phi, 1.0), (p, mu, -1.0, 1.0), (p, mu, phi, -0.5)):
        assert math.isnan(tweedie_logpdf(bad[0], *map(_t, bad[1:])).item())
    for p_out in (1.0499, 1.9501):
        assert math.isnan(tweedie_logpdf(_t(p_out), mu, phi, _t(1.0)).item())
        with pytest.raises(ValueError, match="covers p"):
            tweedie_logpdf(p_out, mu, phi, _t(1.0))
    with pytest.raises(ValueError, match="uniform per call"):
        tweedie_logpdf(_t([1.3, 1.6]), mu, phi, _t(1.0))
    # a one-element tensor is the scalar it holds; both box edges are in it
    one = tweedie_logpdf(_t([1.6]), mu, phi, _t(1.0))
    assert one.item() == tweedie_logpdf(1.6, mu, phi, _t(1.0)).item()
    for edge in (P_LO, P_HI):
        for dtype in (torch.float64, torch.float32):
            assert math.isfinite(tweedie_logpdf(edge, mu, phi, _t(1.0, dtype)).item())


def test_broadcasting_and_dtypes():
    mu = _t([0.5, 1.0, 2.0, 4.0])
    y = _t([[0.0], [0.3], [9.0]])
    got = tweedie_logpdf(1.6, mu, 0.8, y)
    assert got.shape == (3, 4)
    for i in range(3):
        for j in range(4):
            assert got[i, j].item() == tweedie_logpdf(1.6, mu[j], 0.8, y[i, 0]).item()
    half = tweedie_logpdf(1.6, mu.half(), 0.8, y.half())
    assert half.dtype == torch.float16
    assert torch.allclose(half.double(), got, rtol=2e-3, atol=2e-3)


def test_mixed_batch_gradients_are_finite():
    y = _t([0.0, 1e-8, 0.4, 3.0, 40.0])
    p = _t(1.3).requires_grad_(True)
    mu = torch.full_like(y, 2.0, requires_grad=True)
    phi = _t(0.9).requires_grad_(True)
    tweedie_logpdf(p, mu, phi, y).sum().backward()
    assert all(torch.isfinite(g).all() for g in (p.grad, mu.grad, phi.grad))
    # the y = 0 lane is -lambda: d/dmu = -mu^(1-p)/phi exactly
    assert mu.grad[0].item() == pytest.approx(-(2.0 ** (1 - 1.3)) / 0.9, rel=1e-14)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_gradients_are_finite_where_the_value_is(dtype):
    # each lane's other branch runs on stand-in inputs, so it cannot turn a
    # finite value's gradient into nan, also at extreme arguments
    y_big = 1e100 if dtype == torch.float64 else 1e20
    y = _t([0.0, 1e-30, 1e-6, 0.3, 4.0, 1e6, y_big], dtype)
    scales = ((1e-2, 1e-2), (1e-2, 1e2), (1e2, 1e-2), (1e2, 1e2), (2.0, 0.8))
    for p0 in (P_LO, 1.3, 1.7, P_HI):
        for mu0, phi0 in scales:
            p = _t(p0, dtype).requires_grad_(True)
            phi = _t(phi0, dtype).requires_grad_(True)
            mu = torch.full_like(y, mu0, requires_grad=True)
            ys = y.clone().requires_grad_(True)
            v = tweedie_logpdf(p, mu, phi, ys)
            assert torch.isfinite(v).all(), (p0, mu0, phi0, v)
            grads = torch.autograd.grad(v.sum(), (p, mu, phi, ys))
            assert all(torch.isfinite(g).all() for g in grads), (p0, mu0, phi0, grads)


def test_second_derivatives():
    # with create_graph the series' derivatives are recomputed with tracked
    # ops; check d2/dp2 and d2/dp dmu against central differences of the
    # (exact) first derivative
    y = _t([0.0, 0.05, 1.3, 4.0, 30.0])

    def grad_p(p, mu):
        p, mu = _t(p).requires_grad_(True), _t(mu).requires_grad_(True)
        f = tweedie_logpdf(p, mu, 0.8, y).sum()
        return torch.autograd.grad(f, (p, mu), create_graph=True)[0], p, mu

    h = 1e-5
    for p0 in (1.08, 1.45, 1.9):
        g, p, mu = grad_p(p0, 2.0)
        d2p, d2pm = torch.autograd.grad(g, (p, mu))
        fd_p = (grad_p(p0 + h, 2.0)[0] - grad_p(p0 - h, 2.0)[0]).item() / (2 * h)
        fd_pm = (grad_p(p0, 2.0 + h)[0] - grad_p(p0, 2.0 - h)[0]).item() / (2 * h)
        assert d2p.item() == pytest.approx(fd_p, rel=1e-6), p0
        assert d2pm.item() == pytest.approx(fd_pm, rel=1e-6), p0


def test_normalizes_with_its_moments():
    # P(0) + int f = 1, E Y = mu, Var Y = phi mu^p: the whole density at once,
    # integrated in v = ln y (near 0 the density is ~ y^(rho - 1))
    integrate = pytest.importorskip("scipy.integrate")
    for p, mu, phi in ((1.1, 1.0, 1.0), (1.5, 2.0, 0.5), (1.9, 0.5, 2.0),
                       (1.3, 5.0, 0.2)):
        def g(v, k):
            return math.exp(tweedie_logpdf(p, mu, phi, _t(math.exp(v))).item()
                            + (k + 1) * v)
        p0 = math.exp(tweedie_logpdf(p, mu, phi, _t(0.0)).item())
        vhi = math.log(mu + 60 * math.sqrt(phi * mu ** p))
        m = [integrate.quad(g, -745.0, vhi, args=(k,), points=[math.log(mu)],
                            epsabs=1e-14, epsrel=1e-13, limit=400)[0]
             for k in (0, 1, 2)]
        assert abs(p0 + m[0] - 1) <= 1e-9, (p, m)
        assert abs(m[1] - mu) / mu <= 1e-9, (p, m)
        assert abs(m[2] - m[1] ** 2 - phi * mu ** p) / (phi * mu ** p) <= 1e-8, (p, m)


def test_distribution():
    mu, phi, p = 2.0, 0.8, 1.4
    d = Tweedie(_t(mu), phi, p)
    y = _t([0.0, 0.3, 2.0, 7.5])
    assert torch.equal(d.log_prob(y), tweedie_logpdf(p, mu, phi, y))
    torch.manual_seed(0)
    s = d.sample((200_000,))
    var = phi * mu ** p
    p0 = math.exp(-mu ** (2 - p) / (phi * (2 - p)))
    n = s.numel()
    assert abs(s.mean().item() - mu) <= 5 * math.sqrt(var / n)
    assert abs(s.var().item() - var) / var <= 0.02
    assert abs((s == 0).double().mean().item() - p0) <= 5 * math.sqrt(p0 * (1 - p0) / n)
    assert (s >= 0).all()
    assert d.mean.item() == mu and d.variance.item() == pytest.approx(var)
    e = d.expand((3, 2))
    assert e.batch_shape == (3, 2) and e.sample().shape == (3, 2)
    with pytest.raises(ValueError, match="covers power"):
        Tweedie(1.0, 1.0, 1.02)
    with pytest.raises(ValueError):
        Tweedie(1.0, 1.0, 1.5, validate_args=True).log_prob(torch.tensor(-1.0))


def test_torch_compile_matches_eager():
    y = _t([0.0, 0.05, 1.3, 4.0, 300.0])
    # one graph: python numbers must not break it, as an .item() would
    # (fullgraph=True does not catch that one, it only warns)
    explained = torch._dynamo.explain(tweedie_logpdf)(1.37, 2.0, 0.8, y)
    assert explained.graph_break_count == 0
    eager = tweedie_logpdf(1.37, 2.0, 0.8, y)
    compiled = torch.compile(tweedie_logpdf)(1.37, 2.0, 0.8, y)
    torch.testing.assert_close(compiled, eager, rtol=0, atol=1e-14)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.filterwarnings("ignore:Synchronization debug mode")
def test_python_numbers_do_not_wait_for_the_gpu():
    # torch.tensor(1.4, device="cuda") waits for all queued GPU work, on
    # every call. The first call at a python number waits once: it copies
    # the tables and keeps what depends on the number alone
    y = torch.tensor([0.0, 0.3, 2.0, 40.0], device="cuda")
    mu = torch.full_like(y, 2.0, requires_grad=True)
    tweedie_logpdf(1.4, mu, 0.8, y)
    log_wright_bessel0(1.0, y + 1)
    torch.cuda.set_sync_debug_mode("error")
    try:
        tweedie_logpdf(1.4, mu, 0.8, y).sum().backward()
        tweedie_logpdf(1.4, 2.0, 0.8, y)
        log_wright_bessel0(1.0, y + 1)
        Tweedie(mu, 0.8, 1.4, validate_args=False).log_prob(y)
    finally:
        torch.cuda.set_sync_debug_mode(0)


@pytest.mark.parametrize("device", DEVICES)
def test_python_and_tensor_parameters_agree(device):
    # a python p or rho reads the terms kept at its first call, a tensor
    # computes them on every call: the same bits, values and gradients
    y = [0.0, 1e-3, 0.3, 2.0, 40.0, 1e4]
    for dtype in (torch.float64, torch.float32):
        def run(fn, s, *xs):
            xs = [torch.tensor(x, dtype=dtype, device=device, requires_grad=True)
                  for x in xs]
            v = fn(s, *xs)
            return [v, *torch.autograd.grad(v.sum(), xs)]

        for p in (P_LO, 1.37, P_HI):
            ref = run(tweedie_logpdf, torch.tensor(p, dtype=dtype, device=device),
                      2.0, 0.8, y)
            for _ in range(2):              # no call may change what is kept
                got = run(tweedie_logpdf, p, 2.0, 0.8, y)
                assert all(map(torch.equal, got, ref)), (dtype, p)
        for rho in (1 / 19, 0.83, 19.0):
            ref = run(log_wright_bessel0, torch.tensor(rho, dtype=dtype, device=device),
                      [1e-3, 0.5, 30.0])
            for _ in range(2):
                got = run(log_wright_bessel0, rho, [1e-3, 0.5, 30.0])
                assert all(map(torch.equal, got, ref)), (dtype, rho)


INFERENCE_FIRST = r"""
import torch
from chebtorch import log_wright_bessel0, tweedie_logpdf

for dev in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
    y = torch.tensor([0.0, 0.3, 2.0, 40.0], device=dev)
    with torch.inference_mode():            # builds the tables, then the kept terms
        tweedie_logpdf(torch.tensor(1.4, device=dev), 2.0, 0.8, y)
        tweedie_logpdf(1.4, 2.0, 0.8, y)
        log_wright_bessel0(1.0, y + 1)
    p = torch.tensor(1.4, device=dev, requires_grad=True)
    mu = torch.full_like(y, 2.0, requires_grad=True)
    tweedie_logpdf(p, mu, 0.8, y).sum().backward()
    tweedie_logpdf(1.4, mu, 0.8, y).sum().backward()
    log_wright_bessel0(p, y + 1).sum().backward()
"""


def test_a_first_call_in_inference_mode_leaves_gradients_working():
    # autograd cannot save inference tensors, so nothing cached may be one;
    # a fresh process, since this suite has filled the caches already. env
    # is os.environ: conda's numpy sets MKL_THREADING_LAYER in the C
    # environment only, and a child importing torch first refuses it
    out = subprocess.run([sys.executable, "-c", INFERENCE_FIRST],
                         capture_output=True, text=True, env=dict(os.environ))
    assert out.returncode == 0, out.stderr[-2000:]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_graph_capture():
    # p = 1.4 is kept before the capture; p = 1.71 is first seen in it and
    # computed in the graph instead, since capture allows no waiting
    y = torch.tensor([0.0, 0.3, 2.0, 40.0], device="cuda")
    mu = torch.full_like(y, 2.0)
    tweedie_logpdf(1.4, mu, 0.8, y)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kept = tweedie_logpdf(1.4, mu, 0.8, y)
        new = tweedie_logpdf(1.71, mu, 0.8, y)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(kept, tweedie_logpdf(1.4, mu, 0.8, y))
    assert torch.equal(new, tweedie_logpdf(1.71, mu, 0.8, y))
