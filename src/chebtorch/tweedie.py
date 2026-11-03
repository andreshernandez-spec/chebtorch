# Copyright (c) 2026, Andres Hernandez
# SPDX-License-Identifier: BSD-3-Clause
# From chebtorch: https://github.com/andreshernandez-spec/chebtorch
"""Tweedie log-density for power 1.05 <= p <= 1.95, and Wright's Phi(rho, 0; z)
under it. To use it elsewhere, copy it with tweedie_fixed_p.py, _tweedie_data.py
and cheb_common.py into one package; they import only math, torch and each other.
tweedie_logpdf_fixed_p lives in tweedie_fixed_p.py, which copies without this file.

For p in (1, 2) the law has mass exp(-lambda) at 0 and, for y > 0,

    f(y) = exp(-lambda - y/theta) W(rho, z) / y,  W = Phi(rho, 0; z),
    lambda = mu^(2-p) / (phi (2-p)),  theta = phi (p-1) mu^(p-1),
    rho = (2-p)/(p-1),  z = lambda (y/theta)^rho.

Split at Z1 = 12 + 2 (1+rho)^2 in Z = y^(2-p) / (phi (2-p)(p-1)). Below, W is
its 64-term series in log space and mu enters only through the unit deviance d,
so the mean gradient is exactly (y - mu) / (phi mu^p). Above, the density is
the saddlepoint form -d/(2 phi) - ln(2 pi phi y^p)/2 plus T/Z, T a Chebyshev
series in x = 2 Z1/Z - 1 whose 32 coefficients are series in t = ln(rho)/ln(19).
chebax's tweedie_gen has the measurements behind both.

p and rho are uniform per call: a python number (checked, its terms kept between
calls) or a one-element tensor (nan outside the box). Results take the widest
floating dtype given, half precision computed in float32, on the first tensor's
device. Invalid arguments give nan. Derivatives are reverse mode only.

Worst error against mpmath, |err| / max(1, |ref|): density 1e-12 in float64 and
4e-4 in float32, both at p = 1.05; tests/test_tweedie.py has the rest.
"""

import math

import torch
from torch.distributions import constraints

from .cheb_common import Memo, cheb, device_of, scalar, tensor_of, work_dtype
from ._tweedie_data import N_TERMS, P_HI, P_LO, RHO_MAX, RHO_MIN
from .tweedie_fixed_p import (_HALF_LOG_2PI, _NEAR, _near_coefs, _own, _split,
                              tweedie_logpdf_fixed_p)

__all__ = ["Tweedie", "log_wright_bessel0", "tweedie_logpdf", "tweedie_logpdf_fixed_p"]

_LANES = 1 << 16


def _series(rho, u):
    """ln sum_{n=1..N} exp(n u - lgamma(n+1) - lgamma(rho n)) and its u and rho
    derivatives, in one pass over _LANES lanes at a time."""
    n = torch.arange(1, N_TERMS + 1, dtype=u.dtype, device=u.device)
    c = torch.lgamma(n + 1) + torch.lgamma(rho * n)
    g = n * torch.digamma(rho * n)               # -d/drho of term n's exponent
    out = []
    for uc in u.reshape(-1, 1).split(_LANES):
        t = n * uc - c
        # the shift cancels in all three results, so it needs no derivative
        m = t.detach().amax(-1, keepdim=True)
        # in place: no backward needs t
        e = t.sub_(m).exp_()
        s = e.sum(-1)
        # sums, not matmuls: under TF32 a float32 matmul rounds to 10 bits
        out.append((m[:, 0] + torch.log(s), (e * n).sum(-1) / s, -(e * g).sum(-1) / s))
    return tuple(torch.cat(x).reshape(u.shape) for x in zip(*out))


class _LogSeries(torch.autograd.Function):
    """_series with an analytic backward, which keeps two arrays instead of
    every term. Exact second derivatives with create_graph."""

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


@Memo
def _rho_terms(rho):
    """in_box, then rho (1 outside the box), Z1, ln z at Z1, tail coefficients."""
    in_box = (rho >= RHO_MIN) & (rho <= RHO_MAX)
    rho = torch.where(in_box, rho, torch.ones_like(rho))
    z1, coefs = _split(rho)
    u1 = (1 + rho) * torch.log(rho * z1 / (1 + rho)) - torch.log(rho)
    return in_box, rho, z1, u1, coefs


@Memo
def _p_terms(p):
    """in_box, then p (1.5 outside the box), rho, a = 2 - p, p - 1, ln(p - 1),
    ln a, _split(rho) and _near_coefs(a)."""
    # check p, not rho: float32 rounds p = 1.05 to a rho just past RHO_MAX
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
    """(near, mid, delta): |delta| <= _NEAR, else |ln(y/mu)| <= 1, and
    delta = (y - mu)/mu there. No y/mu: it overflows at y = 1e10, mu = 1e-300."""
    near = (y - mu).abs() <= _NEAR * mu
    mid = ~near & (y <= math.e * mu) & (math.e * y >= mu)
    delta = (torch.where(near | mid, y, mu) - mu) / mu
    return near, mid, delta


def _half_deviance(a, c, mu, y):
    """d(y, mu)/2 = mu^a ((1+delta)^a - 1 - a delta) / (a (a-1)), a = 2 - p.

    near: the series of _near_coefs. The closed form loses everything there in
    float32 (2.53 for -5.28 at p = 1.5, mu = 1e6, phi = 1e-12, y = mu + 1/8).
    mid: expm1(a log1p(delta)) - a delta, ~9 eps / (1 - a) at |delta| = 1/4.
    chebax uses log(y/mu) below the mean (docs/divergences.md).
    far: three terms that cancel near the mean, ~50 eps past |ln(y/mu)| = 1."""
    near, mid, delta = _deviance_lanes(mu, y)
    ma = torch.pow(mu, a)
    dn = torch.where(near, delta, 0.0)
    series = ma * dn * dn * _horner(c, dn)
    mid_form = ma * (torch.expm1(a * torch.log1p(delta)) - a * delta) / (a * (a - 1))
    far = (torch.pow(y, a) / a - y * torch.pow(mu, a - 1)) / (a - 1) + ma / a
    return torch.where(near, series, torch.where(mid, mid_form, far))


class _HalfDeviance(torch.autograd.Function):
    """_half_deviance with closed-form partials: autograd of its forms squares
    mu, which overflows at mu = 1e-220. c and dc carry no gradient, so the a
    partial is total. Exact second derivatives with create_graph."""

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
        # exact in every lane, so not the masked delta
        return d_a, None, None, grad * (-m1 * ((y - mu) / mu)), grad * d_y


class _Over(torch.autograd.Function):
    """h / phi with d/dphi = -(h/phi)/phi. Autograd squares phi, which
    underflows at phi = 1e-200 and makes a zero deviance a nan gradient."""

    @staticmethod
    def forward(ctx, h, phi):
        ctx.save_for_backward(h, phi)
        return h / phi

    @staticmethod
    def backward(ctx, grad):
        h, phi = ctx.saved_tensors
        return grad / phi, -(grad * (h / phi)) / phi


def _log_w(rho, z1, u1, coefs, z):
    """ln Phi(rho, 0; z) for rho in the box and z > 0, given _rho_terms(rho)."""
    # pow keeps Z to an ulp (exp of logs carries eps |ln z|) and cannot overflow
    e = 1 / (1 + rho)
    big_z = (1 + rho) / rho * torch.pow(rho, e) * torch.pow(z, e)
    inner = big_z <= z1
    # each branch runs at the split in the other's lanes
    log_w_inner = _LogSeries.apply(rho, torch.where(inner, torch.log(z), u1))
    zt = torch.where(inner, z1, big_z)
    log_w_tail = (zt + 0.5 * torch.log(rho * zt / (1 + rho)) - _HALF_LOG_2PI
                  - 0.5 * torch.log1p(rho) + cheb(2 * (z1 / zt) - 1, coefs) / zt)
    return torch.where(inner, log_w_inner, log_w_tail)


def _log_density(p_terms, mu, phi, y):
    """ln f(y) for y > 0 and -lambda at y = 0, given _p_terms less in_box. Each
    form sees 1 in the other lanes, since an inf in an unselected form's
    partials meets a zero cotangent and makes a nan gradient."""
    p, rho, a, pm1, log_pm1, log_a, z1, coefs, c, dc = p_terms
    pos = y > 0
    mu0, phi0 = _own(~pos, mu, phi)
    log_f_zero = -torch.pow(mu0, a) / (phi0 * a)
    mu_p, phi_p, y_p = _own(pos, mu, phi, y)
    neg_half_dev = -_Over.apply(_HalfDeviance.apply(a, c, dc, mu_p, y_p), phi_p)
    big_z = torch.pow(y_p, a) / (phi_p * a * pm1)
    inner = pos & (big_z <= z1)
    outer = pos & ~inner
    # inner: -lambda - y/theta as -d/(2 phi) + y^a / (a (a-1) phi), z without mu
    phi_i, y_i = _own(inner, phi, y)
    log_y = torch.log(y_i)
    u = rho * (log_y - log_pm1) - (1 + rho) * torch.log(phi_i) - log_a
    log_f_inner = (neg_half_dev + _Over.apply(torch.pow(y_i, a) / (a * (a - 1)), phi_i)
                   - log_y + _LogSeries.apply(rho, u))
    # outer: 1/Z from phi directly, not a reciprocal of Z ~ 1/phi
    phi_t, y_t = _own(outer, phi, y)
    w = torch.where(outer, phi_t * a * pm1 / torch.pow(y_t, a), 1 / z1)
    log_f_tail = (neg_half_dev - 0.5 * (torch.log(phi_t) + p * torch.log(y_t))
                  - _HALF_LOG_2PI + cheb(2 * (z1 * w) - 1, coefs) * w)
    return torch.where(inner, log_f_inner, torch.where(outer, log_f_tail, log_f_zero))


def log_wright_bessel0(rho, z):
    """ln Phi(rho, 0; z), Wright's generalized Bessel function at beta = 0, for
    rho in [1/19, 19] and z >= 0 (-inf at 0). Derivatives in rho and z to second
    order."""
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
    """ln of the Tweedie density at y > 0, and ln P(Y = 0) at y = 0, for p in
    [1.05, 1.95] and broadcasting mu > 0, phi > 0, y >= 0. Derivatives in p, mu,
    phi and y (where y > 0) to second order."""
    work, out = work_dtype(y, mu, phi, p)
    dev = device_of(y, mu, phi, p)
    mu, phi, y = (tensor_of(v, work, dev) for v in (mu, phi, y))
    mu, phi, y = torch.broadcast_tensors(mu, phi, y)
    p = scalar(p, work, dev, "tweedie_logpdf", "p", P_LO, P_HI)
    in_box, *p_terms = _p_terms(p, work, dev)
    valid = (mu > 0) & (phi > 0) & (y >= 0)
    # stand-ins for the lanes the result masks, so no form sees them
    log_f = _log_density(p_terms, *_own(valid, mu, phi, y))
    log_f = torch.where(valid, log_f, math.nan)
    return torch.where(in_box, log_f, math.nan).to(out)


class Tweedie(torch.distributions.Distribution):
    """Tweedie(mu, phi, power), compound Poisson-gamma with mean mu, variance
    phi mu^power and power in [1.05, 1.95]. log_prob is tweedie_logpdf; sample
    is not reparameterized (the count is discrete)."""

    arg_constraints = {"mu": constraints.positive, "phi": constraints.positive}
    support = constraints.nonnegative
    has_rsample = False

    def __init__(self, mu, phi, power, validate_args=None):
        work, _ = work_dtype(mu, phi, power)
        dev = device_of(mu, phi, power)
        mu, phi = (tensor_of(v, work, dev) for v in (mu, phi))
        self.mu, self.phi = torch.broadcast_tensors(mu, phi)
        # a python power stays one, so tweedie_logpdf keeps its terms
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
