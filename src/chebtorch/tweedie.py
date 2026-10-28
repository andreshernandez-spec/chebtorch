# Copyright (c) 2026, Andres Hernandez
# SPDX-License-Identifier: BSD-3-Clause
# From chebtorch: https://github.com/andreshernandez-spec/chebtorch
"""Tweedie log-density for power 1.05 <= p <= 1.95, and Wright's
Phi(rho, 0; z) underneath it. To use it elsewhere, copy it with
_tweedie_data.py (its tables and their provenance) and cheb_common.py into one
package directory. The three import only math, torch and each other.

Two calls give the density. tweedie_logpdf takes every argument as a tensor
or a number, with derivatives in all of them to second order.
tweedie_logpdf_fixed_p is for fitting mu and phi at a power that is a
python number: first derivatives in mu and phi only, the same values within
the same floors, and a fraction of the work per call.

The compound Poisson-gamma Tweedie law with mean mu, dispersion phi and power
p in (1, 2) puts mass exp(-lambda) at 0 and has, for y > 0,

    f(y) = exp(-lambda - y/theta) W(rho, z) / y,  W = Phi(rho, 0; z),
    lambda = mu^(2-p) / (phi (2-p)),  theta = phi (p-1) mu^(p-1),
    rho = (2-p)/(p-1),  z = lambda (y/theta)^rho.

Two regions in Z = y^(2-p) / (phi (2-p)(p-1)), split at Z1 = 12 + 2 (1+rho)^2.
Below it W is its own series, 64 terms summed in log space, and
-lambda - y/theta is formed as -d(y, mu)/(2 phi) + y^a / (a (a-1) phi),
a = 2 - p, with z built without mu, so mu enters only through the unit
deviance d and the mean gradient is its exact (y - mu) / (phi mu^p). Above
it the density is exactly the saddlepoint approximation
-d(y, mu)/(2 phi) - ln(2 pi phi y^p)/2 plus a tabulated correction T/Z. T is
a Chebyshev series in x = 2 Z1/Z - 1 (32 coefficients; x = -1 is Z -> inf),
each coefficient a Chebyshev series in t = ln(rho)/ln(19) (64 coefficients);
both coordinates span [-1, 1] over the box. d is a series in (y - mu)/mu near
the mean (_half_deviance), so concentrated data (small phi, Z up to 1e16 and
beyond) keeps its digits. The fixtures straddle the seam at Z1 (1 -+ 1e-9).
chebax (chebax._src.recipes.tweedie_gen) has the measurements behind the
split and the table.

Accuracy against mpmath at 40 digits, |err| / max(1, |ref|), worst cases in
float64: ln W 7e-16, the density 1e-12, d/dp 6e-11, and 1e-11 for the
gradients in ln phi and ln y, all at p = 1.05, where lgamma(rho n) reaches
~4000 in the series; 4e-16 in ln mu. float32 arithmetic adds the same
cancellation at its own eps: density 4e-4 and d/dp 2e-2 below p = 1.1,
2.3e-5 and 5e-4 below p = 1.3, 1.5e-6 and 7e-5 above, and 2e-7 in ln mu.
Concentrated data (Z from 4e3 to 4e15): 3e-16 in float64, 1.6e-7 in float32.
"""

import math

import torch
from torch.distributions import constraints

from .cheb_common import Memo, Tables, cheb, device_of, scalar, tensor_of, work_dtype
from ._tweedie_data import (A1, DATA, N_TERMS, P_HI, P_LO, RHO_MAX, RHO_MIN, S1,
                            SHAPES, ZA)

__all__ = ["Tweedie", "log_wright_bessel0", "tweedie_logpdf", "tweedie_logpdf_fixed_p"]

_HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)
_LANES = 1 << 16
_NEAR = 0.25      # |y/mu - 1| up to which the half-deviance is a series
_N_NEAR = 26      # and its term count
_tables = Tables(SHAPES, DATA)


def _series(rho, u):
    """ln S for S = sum_{n=1..N} exp(n u - lgamma(n+1) - lgamma(rho n)), and
    its derivatives in u and rho, all from one pass: both are averages over
    the same weights e_n / S. The terms are a (lanes, N) tensor, so a call
    is a few kernels rather than a few per term. Lanes go _LANES at a time,
    which caps the temporaries at two such tensors (64 MB in float64)."""
    n = torch.arange(1, N_TERMS + 1, dtype=u.dtype, device=u.device)
    c = torch.lgamma(n + 1) + torch.lgamma(rho * n)
    g = n * torch.digamma(rho * n)               # -d/drho of term n's exponent
    out = []
    for uc in u.reshape(-1, 1).split(_LANES):
        t = n * uc - c
        # the shift cancels in all three results, so it needs no derivative
        m = t.detach().amax(-1, keepdim=True)
        # in place: no backward needs t itself
        e = t.sub_(m).exp_()
        s = e.sum(-1)
        # sums, not matmuls: under TF32 a float32 matmul rounds to 10 bits
        out.append((m[:, 0] + torch.log(s), (e * n).sum(-1) / s, -(e * g).sum(-1) / s))
    return tuple(torch.cat(x).reshape(u.shape) for x in zip(*out))


class _LogSeries(torch.autograd.Function):
    """The series with its analytic backward. Autograd through the unrolled
    sum would keep every term for the backward pass; this keeps two arrays.
    With create_graph the derivatives are recomputed with tracked ops, so
    second derivatives are exact too. No jvp, so no forward mode."""

    @staticmethod
    def forward(ctx, rho, u):
        val, d_u, d_rho = _series(rho, u)
        ctx.save_for_backward(rho, u, d_u, d_rho)
        return val

    @staticmethod
    def backward(ctx, grad):
        rho, u, d_u, d_rho = ctx.saved_tensors
        if torch.is_grad_enabled():
            _, d_u, d_rho = _series(rho, u)
        return (grad * d_rho).sum(), grad * d_u


def _split(rho):
    """Z1 and the tail series' 32 coefficients in x at this rho."""
    z1 = ZA + S1 * (1 + rho) ** 2
    coefs = cheb(torch.log(rho) / A1, _tables("TABLE_TAIL", rho.dtype, rho.device).T)
    return z1, coefs


def _near_coefs(a):
    """c_2 .. c_27 of the binomial series of ((1+delta)^a - 1 - a delta) /
    (a (a-1)), c_2 = 1/2 and c_{k+1} = c_k (a-k)/(k+1), and their
    derivatives in a, c_k sum_{j=2}^{k-1} 1/(a-j). 26 terms leave under
    3e-17 relative at |delta| = _NEAR for every a in (0, 1)."""
    j = torch.arange(2, _N_NEAR + 1, dtype=a.dtype, device=a.device)
    c = 0.5 * torch.cat([torch.ones_like(a)[None], torch.cumprod((a - j) / (j + 1), 0)])
    dc = c * torch.cat([torch.zeros_like(a)[None], torch.cumsum(1 / (a - j), 0)])
    return c, dc


@Memo
def _rho_terms(rho):
    """What depends on rho alone: whether it is in the box, then rho (1, a
    stand-in, outside it), Z1, u = ln z at Z1 and the tail's coefficients."""
    in_box = (rho >= RHO_MIN) & (rho <= RHO_MAX)
    rho = torch.where(in_box, rho, torch.ones_like(rho))
    z1, coefs = _split(rho)
    u1 = (1 + rho) * torch.log(rho * z1 / (1 + rho)) - torch.log(rho)
    return in_box, rho, z1, u1, coefs


@Memo
def _p_terms(p):
    """What depends on p alone: whether it is in the box, then p (1.5, a
    stand-in, outside it), rho, a = 2 - p, p - 1, ln(p - 1), ln a, _split at
    rho and _near_coefs at a."""
    # the box is checked on p as given: in float32, p = 1.05 rounds to
    # 1.04999995, whose rho is just past RHO_MAX
    in_box = (p >= P_LO) & (p <= P_HI)
    p = torch.where(in_box, p, torch.full_like(p, 1.5))
    rho = (2 - p) / (p - 1)
    a, pm1 = 2 - p, p - 1
    return ((in_box, p, rho, a, pm1, torch.log(pm1), torch.log(a)) + _split(rho)
            + _near_coefs(a))


def _horner(c, x):
    """sum_k c[k] x^k."""
    acc = c[-1]
    for k in range(c.shape[0] - 2, -1, -1):
        acc = torch.addcmul(c[k], acc, x)
    return acc


def _deviance_lanes(mu, y):
    """(near, mid, delta): the lanes with |delta| <= _NEAR, those with
    |ln(y/mu)| <= 1 otherwise, and delta = (y - mu)/mu in both, 0 elsewhere.
    No y/mu is formed: it overflows at y = 1e10, mu = 1e-300."""
    near = (y - mu).abs() <= _NEAR * mu
    mid = ~near & (y <= math.e * mu) & (math.e * y >= mu)
    delta = (torch.where(near | mid, y, mu) - mu) / mu
    return near, mid, delta


def _half_deviance(a, c, mu, y):
    """d(y, mu)/2 for power p = 2 - a, a in (0, 1), y > 0, mu > 0, with c
    from _near_coefs(a), in three forms of mu^a ((1+delta)^a - 1 - a delta)
    / (a (a-1)), delta = (y - mu)/mu.

    near: mu^a sum_k c_k delta^k, the series of _near_coefs. The closed form
    there, with x = ln y - ln mu (which carries eps |ln y| where only delta
    matters) and expm1(a x) - a expm1(x) (which cancels two first-order
    terms down to a quadratic one), gave 2.53 for a log-density of -5.28 in
    float32 at (p, mu, phi, y) = (1.5, 1e6, 1e-12, 1e6 + 1/8).
    mid: mu^a (expm1(a x) - a delta) / (a (a-1)) with x = log1p(delta). Its
    first-order cancellation costs about 9 eps / (1 - a) at |delta| = 1/4.
    far: y^a / (a (a-1)) - y mu^(a-1) / (a-1) + mu^a / a. Its terms are
    O(1/(a (1-a))) and cancel to O(delta^2), which costs 32 eps / (a (1-a))
    at |delta| = 1/4 (670 eps at the box edges) but only ~50 eps beyond
    |ln(y/mu)| = 1, where it takes over. chebax uses it from |delta| = 1/4."""
    near, mid, delta = _deviance_lanes(mu, y)
    ma = torch.pow(mu, a)
    dn = torch.where(near, delta, 0.0)
    series = ma * dn * dn * _horner(c, dn)
    mid_form = ma * (torch.expm1(a * torch.log1p(delta)) - a * delta) / (a * (a - 1))
    far = (torch.pow(y, a) / a - y * torch.pow(mu, a - 1)) / (a - 1) + ma / a
    return torch.where(near, series, torch.where(mid, mid_form, far))


class _HalfDeviance(torch.autograd.Function):
    """_half_deviance with its partials in closed form. Autograd of the forms
    there squares mu, in mu^(a-2) and in delta's own derivative, and that
    overflows at mu = 1e-220, where the mean gradient at y = mu is exactly 0.
    In mu and y they are -mu^(a-1) delta and (y^(a-1) - mu^(a-1))/(a-1), the
    latter through expm1 where |ln(y/mu)| <= 1; in a, each form's own, the
    series' through dc from _near_coefs. The c and dc it is given carry no
    gradient: the a partial is total. With create_graph the partials are
    formed from tracked ops, so second derivatives are exact too."""

    @staticmethod
    def forward(ctx, a, c, dc, mu, y):
        ctx.save_for_backward(a, c, dc, mu, y)
        return _half_deviance(a, c, mu, y)

    @staticmethod
    def backward(ctx, grad):
        a, c, dc, mu, y = ctx.saved_tensors
        near, mid, delta = _deviance_lanes(mu, y)
        x = torch.log1p(delta)
        m1 = torch.pow(mu, a - 1)
        d_y = torch.where(near | mid, m1 * torch.expm1((a - 1) * x),
                          torch.pow(y, a - 1) - m1) / (a - 1)
        d_a = None
        if ctx.needs_input_grad[0]:
            ma, ya, log_mu = torch.pow(mu, a), torch.pow(y, a), torch.log(mu)
            dn = torch.where(near, delta, 0.0)
            near_a = ma * dn * dn * (log_mu * _horner(c, dn) + _horner(dc, dn))
            e, den = torch.expm1(a * x), a * (a - 1)
            num = e - a * delta
            mid_a = (log_mu * ma * num / den
                     + ma * ((x * (e + 1) - delta) * den - num * (2 * a - 1)) / (den * den))
            far = ya / a - y * m1
            far_a = ((ya * torch.log(y) / a - ya / (a * a) - y * m1 * log_mu) / (a - 1)
                     - far / ((a - 1) * (a - 1)) + ma * log_mu / a - ma / (a * a))
            d_a = (grad * torch.where(near, near_a, torch.where(mid, mid_a, far_a))).sum()
        # exact in every lane, so from delta itself rather than the masked one
        return d_a, None, None, grad * (-m1 * ((y - mu) / mu)), grad * d_y


class _Over(torch.autograd.Function):
    """h / phi, its phi-derivative formed as -(h/phi)/phi. Autograd's own rule
    squares phi first, which underflows at phi = 1e-200 and turns a zero
    deviance into a nan dispersion gradient."""

    @staticmethod
    def forward(ctx, h, phi):
        ctx.save_for_backward(h, phi)
        return h / phi

    @staticmethod
    def backward(ctx, grad):
        h, phi = ctx.saved_tensors
        return grad / phi, -(grad * (h / phi)) / phi


def _log_w(rho, z1, u1, coefs, z):
    """ln Phi(rho, 0; z) for rho in the box and z > 0, given _rho_terms at
    rho."""
    # pow keeps Z to an ulp where exp((ln rho + ln z)/(1+rho)) would carry
    # eps |ln z|; z^e cannot overflow for e < 1
    e = 1 / (1 + rho)
    big_z = (1 + rho) / rho * torch.pow(rho, e) * torch.pow(z, e)
    inner = big_z <= z1
    # each branch gets inputs from its own region: the series runs at the
    # split in tail lanes, the tail formula at Z1 in inner lanes
    log_w_inner = _LogSeries.apply(rho, torch.where(inner, torch.log(z), u1))
    zt = torch.where(inner, z1, big_z)
    log_w_tail = (zt + 0.5 * torch.log(rho * zt / (1 + rho)) - _HALF_LOG_2PI
                  - 0.5 * torch.log1p(rho) + cheb(2 * (z1 / zt) - 1, coefs) / zt)
    return torch.where(inner, log_w_inner, log_w_tail)


def _own(lanes, *xs):
    """The arguments in these lanes and 1 in the others."""
    return [torch.where(lanes, x, 1.0) for x in xs]


def _log_density(p_terms, mu, phi, y):
    """ln f(y) for y > 0 and ln P(Y = 0) = -lambda at y = 0, for mu, phi > 0,
    y >= 0 and p_terms from _p_terms, less in_box.

    Each form sees its own lanes' arguments and 1 in the others: a form's
    partials meet a zero cotangent where it is not selected, so one inf among
    them is a nan gradient (at phi = 1e-200, lambda / phi alone overflows).
    The positive lanes share -d/(2 phi)."""
    p, rho, a, pm1, log_pm1, log_a, z1, coefs, c, dc = p_terms
    pos = y > 0
    mu0, phi0 = _own(~pos, mu, phi)
    log_f_zero = -torch.pow(mu0, a) / (phi0 * a)
    mu_p, phi_p, y_p = _own(pos, mu, phi, y)
    neg_half_dev = -_Over.apply(_HalfDeviance.apply(a, c, dc, mu_p, y_p), phi_p)
    big_z = torch.pow(y_p, a) / (phi_p * a * pm1)
    inner = pos & (big_z <= z1)
    outer = pos & ~inner
    # inner: the density as defined, ln W on u = ln z. mu enters only through
    # -lambda - y/theta = -d/(2 phi) + y^a / (a (a-1) phi), so the mean
    # gradient is the deviance's exact (y - mu) / (phi mu^p). Formed as
    # -lambda and -y/theta, their mu-derivatives cancel only to rounding at
    # y = mu, ~1e286 at mu = 1e-300. Likewise z = y^rho / (a (p-1)^rho
    # phi^(1+rho)) holds no mu.
    phi_i, y_i = _own(inner, phi, y)
    log_y = torch.log(y_i)
    u = rho * (log_y - log_pm1) - (1 + rho) * torch.log(phi_i) - log_a
    log_f_inner = (neg_half_dev + _Over.apply(torch.pow(y_i, a) / (a * (a - 1)), phi_i)
                   - log_y + _LogSeries.apply(rho, u))
    # outer: the saddlepoint form plus the tabulated correction T/Z, with
    # 1/Z built from phi directly rather than as a reciprocal of Z ~ 1/phi
    phi_t, y_t = _own(outer, phi, y)
    w = torch.where(outer, phi_t * a * pm1 / torch.pow(y_t, a), 1 / z1)
    log_f_tail = (neg_half_dev - 0.5 * (torch.log(phi_t) + p * torch.log(y_t))
                  - _HALF_LOG_2PI + cheb(2 * (z1 * w) - 1, coefs) * w)
    return torch.where(inner, log_f_inner, torch.where(outer, log_f_tail, log_f_zero))


def _tail_coefs_python(rho):
    """The tail series' coefficients in x at this rho as python floats, by
    Clenshaw over the table's rows: for sizing, not for evaluation."""
    t = math.log(rho) / A1
    rows, na = SHAPES["TABLE_TAIL"]
    flat = DATA["TABLE_TAIL"]
    out = []
    for k in range(rows):
        row = flat[k * na:(k + 1) * na]
        b1 = b2 = 0.0
        for c in reversed(row[1:]):
            b1, b2 = 2 * t * b1 - b2 + c, b1
        out.append(t * b1 - b2 + row[0])
    return out


def _fixed_p_sizes(p, eps):
    """(series, deviance series, tail) term counts that keep a python p's
    error at eps: the series through its last term above eps e^-2 of the
    largest at the split, where the peak index is largest (it grows with u);
    the deviance series through its last term above eps/8 of the first at
    |delta| = _NEAR; the tail through its last coefficient above eps/4 of the
    largest. Python floats only, so no host sync and no graph break."""
    rho = (2 - p) / (p - 1)
    z1 = ZA + S1 * (1 + rho) ** 2
    u1 = (1 + rho) * math.log(rho * z1 / (1 + rho)) - math.log(rho)
    t = [n * u1 - math.lgamma(n + 1) - math.lgamma(rho * n)
         for n in range(1, 2 * N_TERMS + 1)]
    top = max(t)
    k_series = max(n for n, tn in enumerate(t, 1) if tn - top > math.log(eps) - 2)
    a, c, k_near = 2 - p, 0.5, 1
    for k in range(2, _N_NEAR + 1):
        c *= (a - k) / (k + 1)                       # c_{k+1}, times delta^(k-1)
        if abs(c) * _NEAR ** (k - 1) < eps / 8:
            break
        k_near += 1
    coefs = [abs(ck) for ck in _tail_coefs_python(rho)]
    k_tail = 1 + max(k for k, ck in enumerate(coefs) if ck > eps / 4 * max(coefs))
    return k_series, k_near, k_tail


def _chebder(c):
    """Coefficients of d/dx sum_k c[k] T_k(x), as numpy.polynomial's chebder."""
    n = c.shape[0]
    d = [torch.zeros_like(c[0])] * (n + 1)
    for k in range(n - 1, 0, -1):
        d[k - 1] = d[k + 1] + 2 * k * c[k]
    d[0] = d[0] / 2
    return torch.stack(d[:max(n - 1, 1)])


def _fixed_p_constants(p, dtype, device):
    """What tweedie_logpdf_fixed_p needs of a python p, with its series cut
    to _fixed_p_sizes: p, rho, a = 2 - p, 1/(p-1) as a python float, a - 1,
    a (p-1), a (a-1), 1 + rho, rho ln(p-1) + ln a, Z1, 2 Z1, 1/Z1; n and
    -(lgamma(n+1) + lgamma(rho n)) for the series; the tail's coefficients,
    those of its derivative and k for cos(k acos x); the deviance series'
    coefficients and the exponents of its powers."""
    if torch.compiler.is_compiling():
        # dynamo would step through the sizing loops bytecode by bytecode
        # (torch 2.4 for minutes); a fused graph can afford every term
        k_series, k_near, k_tail = N_TERMS, _N_NEAR, SHAPES["TABLE_TAIL"][0]
    else:
        k_series, k_near, k_tail = _fixed_p_sizes(p, torch.finfo(dtype).eps)
    inv_b = 1 / (p - 1)
    p = tensor_of(p, dtype, device)
    rho = (2 - p) / (p - 1)
    a, b = 2 - p, p - 1
    z1, coefs = _split(rho)
    coefs = coefs[:k_tail]
    n = torch.arange(1, k_series + 1, dtype=dtype, device=device)
    return (p, rho, a, inv_b, a - 1, a * b, a * (a - 1), 1 + rho,
            rho * torch.log(b) + torch.log(a), z1, 2 * z1, 1 / z1,
            n, -(torch.lgamma(n + 1) + torch.lgamma(rho * n)),
            coefs, _chebder(coefs), torch.arange(k_tail, dtype=dtype, device=device),
            _near_coefs(a)[0][:k_near], torch.arange(k_near, dtype=dtype, device=device))


_fixed_p_terms = Memo(_fixed_p_constants, raw=True)
_FIXED_P_BYTES = 64 << 20   # temporaries of one pass in _fixed_p_lanes


def _fixed_p_lanes(terms, u, x, dn, grads):
    """Per lane, from (lanes, terms) tensors taken _FIXED_P_BYTES at a time, a
    few kernels each: ln S at u, the series cut to n's terms, and with grads
    its mean n (d ln S/du); T at x as sum_k c_k cos(k acos x), and with grads
    T'(x); the deviance series sum_j c_j dn^j."""
    n, neg_c, coefs, dcoefs, k, c_near, j = terms[12:]
    width = n.numel() + k.numel() + j.numel()
    step = max(1, _FIXED_P_BYTES // (2 * width * u.element_size()))
    if torch.compiler.is_compiling():
        # one pass: inductor fuses the terms into reductions, so nothing of
        # (lanes, terms) is kept, and every pass would be more graph (80 s
        # to compile at 2^20 lanes, 1.3 s a call in float64)
        step = max(1, u.numel())
    theta = torch.acos(x)
    out = [], [], [], [], []
    for uc, tc, dc in zip(*(v.reshape(-1, 1).split(step) for v in (u, theta, dn))):
        t = torch.addcmul(neg_c, n, uc)
        m = t.amax(-1, keepdim=True)
        e = t.sub_(m).exp_()
        s = e.sum(-1)
        cos_k = torch.cos(tc * k)
        out[0].append(m[:, 0] + torch.log(s))
        out[2].append((cos_k * coefs).sum(-1))
        out[4].append((torch.pow(dc, j) * c_near).sum(-1))
        if grads:
            out[1].append((e * n).sum(-1) / s)
            out[3].append((cos_k[:, :dcoefs.numel()] * dcoefs).sum(-1))
    return tuple(None if not xs else (xs[0] if len(xs) == 1 else torch.cat(xs)).reshape(u.shape)
                 for xs in out)


def _fixed_p_density(terms, mu, phi, y, grads):
    """(ln f, d/dmu, d/dphi) for tweedie_logpdf_fixed_p, with mu, phi > 0 and
    y >= 0, formed without autograd (the partials None without grads). Every
    lane runs every form and torch.where picks, so an unselected form may be
    inf or nan where it does not apply.

    d ln f/dmu = (y - mu) / (phi mu^p) in every lane, the exponential
    dispersion family's identity. phi d ln f/dphi is lambda at y = 0;
    lambda + y/theta - (1 + rho) E[n] below the split, E[n] the series' mean
    n; and d/2 / phi - 1/2 + (T + 2 Z1 w T') w above it, w = 1/Z.

    The half-deviance d/2 is mu^a sum_j c_j delta^(j+2) near the mean, as in
    _half_deviance, and mu^a (expm1(a x) - a delta) / (a (a-1)) elsewhere,
    which only the overflow of y/mu kept from serving beyond |ln(y/mu)| = 1.
    There x = ln(y/mu), not log1p(delta): |x| >= 0.22, and at y << mu,
    1 + delta keeps none of y/mu's digits."""
    p, rho, a, inv_b, am1, ab, aa1, rho1, u0, z1, z1x2, inv_z1 = terms[:12]
    m1 = torch.pow(mu, am1)
    ma = mu * m1
    ma_a = ma / a
    lam = ma_a / phi
    lam_yt = torch.addcmul(ma_a, y, m1, value=inv_b) / phi    # lambda + y/theta
    w = phi * ab / torch.pow(y, a)                            # 1/Z, inf at y = 0
    pos, outer = y > 0, w < inv_z1
    log_y, log_phi = torch.log(y), torch.log(phi)
    u = rho * log_y - rho1 * log_phi - u0
    sw = z1x2 * w
    delta = (y - mu) / mu
    near = delta.abs() <= _NEAR
    dn = torch.where(near, delta, 0.0)
    log_s, mean_n, t, dt, s_near = _fixed_p_lanes(terms, u, (sw - 1).clamp(-1.0, 1.0),
                                                  dn, grads)
    h_phi = torch.where(near, ma * dn * dn * s_near,
                        ma * (torch.expm1(a * torch.log(y / mu)) - a * delta) / aa1) / phi
    f_in = log_s - lam_yt - log_y
    f_out = t * w - h_phi - 0.5 * (log_phi + p * log_y) - _HALF_LOG_2PI
    log_f = torch.where(outer, f_out, torch.where(pos, f_in, -lam))
    if not grads:
        return log_f, None, None
    d_in = lam_yt - rho1 * mean_n
    d_out = h_phi - 0.5 + (t + sw * dt) * w
    d_phi = torch.where(outer, d_out, torch.where(pos, d_in, lam)) / phi
    return log_f, delta * m1 / phi, d_phi


class _FixedPDensity(torch.autograd.Function):
    """_fixed_p_density with its partials formed in the forward pass; the
    backward only scales them. First derivatives in mu and phi only."""

    @staticmethod
    def forward(ctx, terms, mu, phi, y):
        log_f, d_mu, d_phi = _fixed_p_density(terms, mu, phi, y, True)
        ctx.save_for_backward(d_mu, d_phi)
        return log_f

    @staticmethod
    def backward(ctx, grad):
        if torch.is_grad_enabled():
            raise RuntimeError("tweedie_logpdf_fixed_p has no second derivatives; "
                               "use tweedie_logpdf")
        d_mu, d_phi = ctx.saved_tensors
        return None, grad * d_mu, grad * d_phi, None


def log_wright_bessel0(rho, z):
    """ln Phi(rho, 0; z), Wright's generalized Bessel function at beta = 0.

    rho is a scalar in [1/19, 19] (python number or one-element tensor),
    uniform per call; z >= 0 is a tensor or number. Computes in the widest
    floating dtype among the arguments, else the default dtype (float16 and
    bfloat16 through float32), on the device of the first tensor argument.

    A python rho is the cheaper form for a fixed rho: what depends on rho
    alone is computed at the first call with that rho (per dtype and device)
    and kept, and only that call waits for the GPU. A tensor rho recomputes it
    on every call.

    z = 0 gives -inf (Phi(rho, 0; 0) = 0). z < 0, nan and a tensor rho outside
    the box give nan; a python rho outside it raises ValueError.

    Differentiable in rho and z, to second order, in reverse mode only:
    forward mode (torch.func.jvp, torch.autograd.forward_ad) is not
    implemented.
    """
    work, out = work_dtype(z, rho)
    dev = device_of(z, rho)
    z = tensor_of(z, work, dev)
    rho = scalar(rho, work, dev, "log_wright_bessel0", "rho", RHO_MIN, RHO_MAX)
    in_box, *rho_terms = _rho_terms(rho, work, dev)
    positive = z > 0
    # a stand-in for the lanes the result masks, so no branch sees them
    log_w = _log_w(*rho_terms, torch.where(positive, z, torch.ones_like(z)))
    log_w = torch.where(positive, log_w, torch.where(z == 0, -math.inf, math.nan))
    return torch.where(in_box, log_w, math.nan).to(out)


def tweedie_logpdf(p, mu, phi, y):
    """ln of the Tweedie density at y, for power p in [1.05, 1.95].

    For y > 0 the log-density; at y = 0, ln P(Y = 0) = -lambda, the point
    mass. p is a scalar (python number or one-element tensor), uniform per
    call; mu > 0, phi > 0 and y >= 0 broadcast against each other. Computes in
    the widest floating dtype among the arguments, else the default dtype
    (float16 and bfloat16 through float32), on the device of the first tensor
    argument.

    A python p is the cheaper form for a fixed p: what depends on p alone,
    such as the tail series' coefficients, is computed at the first call with
    that p (per dtype and device) and kept, and only that call waits for the
    GPU. A tensor p recomputes it on every call.

    mu <= 0, phi <= 0, y < 0, nan and a tensor p outside the box give nan; a
    python p outside it raises ValueError.

    Differentiable in p, mu, phi and y (y only where y > 0), to second order,
    in reverse mode only: forward mode is not implemented.
    """
    work, out = work_dtype(y, mu, phi, p)
    dev = device_of(y, mu, phi, p)
    # created in the working dtype: as_tensor(0.7).to(float64) would round
    # 0.7 through float32 first
    mu, phi, y = (tensor_of(v, work, dev) for v in (mu, phi, y))
    mu, phi, y = torch.broadcast_tensors(mu, phi, y)
    p = scalar(p, work, dev, "tweedie_logpdf", "p", P_LO, P_HI)
    in_box, *p_terms = _p_terms(p, work, dev)
    valid = (mu > 0) & (phi > 0) & (y >= 0)
    # stand-ins for the lanes the result masks, so no form sees them
    log_f = _log_density(p_terms, *_own(valid, mu, phi, y))
    log_f = torch.where(valid, log_f, math.nan)
    return torch.where(in_box, log_f, math.nan).to(out)


def tweedie_logpdf_fixed_p(p, mu, phi, y):
    """tweedie_logpdf for fitting mu and phi at a fixed power, with less
    work per call: the same values, within the same floors.

    Restricted to: p a python number in [1.05, 1.95]; derivatives in mu and
    phi, first order, reverse mode (none in y, which must not require one);
    y/mu finite in the working dtype. Within those, mu > 0, phi > 0 and
    y >= 0 broadcast, the dtype and device rules are tweedie_logpdf's, and
    invalid arguments give nan.

    What the restrictions buy: the series, the deviance's series and the
    tail correction keep only the terms the working dtype resolves at this
    p (for float32 at p = 1.5, 26, 11 and 5 of 64, 26 and 32), and the
    partials come out of the forward pass in closed form, so the backward
    pass only scales them.
    """
    if isinstance(p, torch.Tensor):
        raise TypeError("tweedie_logpdf_fixed_p takes p as a python number; "
                        "use tweedie_logpdf for a tensor p")
    work, out = work_dtype(y, mu, phi)
    dev = device_of(y, mu, phi)
    p = scalar(p, work, dev, "tweedie_logpdf_fixed_p", "p", P_LO, P_HI)
    mu, phi, y = (tensor_of(v, work, dev) for v in (mu, phi, y))
    if y.requires_grad:
        raise ValueError("tweedie_logpdf_fixed_p has no derivative in y; detach y, "
                         "or use tweedie_logpdf")
    mu, phi, y = torch.broadcast_tensors(mu, phi, y)
    terms = _fixed_p_terms(p, work, dev)
    valid = (mu > 0) & (phi > 0) & (y >= 0)
    # stand-ins for the lanes the result masks
    mu_s, phi_s, y_s = _own(valid, mu, phi, y)
    if torch.is_grad_enabled() and (mu.requires_grad or phi.requires_grad):
        log_f = _FixedPDensity.apply(terms, mu_s, phi_s, y_s)
    else:
        log_f = _fixed_p_density(terms, mu_s, phi_s, y_s, False)[0]
    return torch.where(valid, log_f, math.nan).to(out)


class Tweedie(torch.distributions.Distribution):
    """Tweedie(mu, phi, power), the compound Poisson-gamma law.

    Y is exactly 0 with probability exp(-mu^(2-p) / (phi (2-p))) and positive
    and continuous otherwise. Mean mu, variance phi mu^power. mu and phi
    broadcast; power is a scalar in [1.05, 1.95], checked as in
    tweedie_logpdf, which log_prob is (same dtype, derivative and nan rules,
    and a python power stays a python number, the cheaper form).
    Sampling draws the Poisson count and a gamma total; it is not
    reparameterized (the count is discrete).
    """

    arg_constraints = {"mu": constraints.positive, "phi": constraints.positive}
    support = constraints.nonnegative
    has_rsample = False

    def __init__(self, mu, phi, power, validate_args=None):
        work, _ = work_dtype(mu, phi, power)
        dev = device_of(mu, phi, power)
        mu, phi = (tensor_of(v, work, dev) for v in (mu, phi))
        self.mu, self.phi = torch.broadcast_tensors(mu, phi)
        # log_prob passes a python power on as a number, whose terms
        # tweedie_logpdf keeps between calls
        self._power = scalar(power, work, dev, "Tweedie", "power", P_LO, P_HI)
        self.power = tensor_of(self._power, work, dev)
        super().__init__(self.mu.shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Tweedie, _instance)
        batch_shape = torch.Size(batch_shape)
        new.mu = self.mu.expand(batch_shape)
        new.phi = self.phi.expand(batch_shape)
        new._power, new.power = self._power, self.power
        super(Tweedie, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self):
        return self.mu

    @property
    def variance(self):
        return self.phi * self.mu ** self.power

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return tweedie_logpdf(self._power, self.mu, self.phi, value)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            p = self.power
            mu, phi = self.mu.expand(shape), self.phi.expand(shape)
            n = torch.poisson(mu ** (2 - p) / (phi * (2 - p)))
            # the gamma total of n claims of shape (2-p)/(p-1); n = 0 is Y = 0
            conc = n.clamp(min=1) * (2 - p) / (p - 1)
            total = torch.distributions.Gamma(conc, torch.ones_like(conc)).sample()
            return torch.where(n > 0, total * phi * (p - 1) * mu ** (p - 1), 0.0)
