# Copyright (c) 2026, Andres Hernandez
# SPDX-License-Identifier: BSD-3-Clause
# From chebtorch: https://github.com/andreshernandez-spec/chebtorch
"""Tweedie log-density for power 1.05 <= p <= 1.95, and Wright's
Phi(rho, 0; z) underneath it. To use it elsewhere, copy it with
_tweedie_data.py (its tables and their provenance) and cheb_common.py into one
package directory. The three import only math, torch and each other.

The compound Poisson-gamma Tweedie law with mean mu, dispersion phi and power
p in (1, 2) puts mass exp(-lambda) at 0 and has, for y > 0,

    f(y) = exp(-lambda - y/theta) W(rho, z) / y,  W = Phi(rho, 0; z),
    lambda = mu^(2-p) / (phi (2-p)),  theta = phi (p-1) mu^(p-1),
    rho = (2-p)/(p-1),  z = lambda (y/theta)^rho.

Two regions in Z = y^(2-p) / (phi (2-p)(p-1)), split at Z1 = 12 + 2 (1+rho)^2.
Below it W is its own series, 64 terms summed in log space. Above it the
density is exactly the saddlepoint approximation
-d(y, mu)/(2 phi) - ln(2 pi phi y^p)/2 (d the unit deviance) plus a tabulated
correction T/Z. T is a Chebyshev series in x = 2 Z1/Z - 1 (32 coefficients;
x = -1 is Z -> inf), each coefficient a Chebyshev series in
t = ln(rho)/ln(19) (64 coefficients); both coordinates span [-1, 1] over the
box. The fixtures straddle the seam at Z1 (1 -+ 1e-9). chebax
(chebax._src.recipes.tweedie_gen) has the measurements behind the split and
the table.

Accuracy against mpmath at 40 digits, |err| / max(1, |ref|), worst cases in
float64: ln W 7e-16, the density 8e-13, d/dp 4e-11, and 7e-12 for the
gradients in ln mu, ln phi and ln y, all at p = 1.05, where lgamma(rho n)
reaches ~4000 in the series. float32 arithmetic adds the same cancellation at
its own eps: density 3e-4 and d/dp 2e-2 below p = 1.1, 1.6e-5 and 5e-4 below
p = 1.3, 1.5e-6 and 2e-5 above.
"""

import math

import torch
from torch.distributions import constraints

from .cheb_common import Memo, Tables, cheb, device_of, scalar, tensor_of, work_dtype
from ._tweedie_data import (A1, DATA, N_TERMS, P_HI, P_LO, RHO_MAX, RHO_MIN, S1,
                            SHAPES, ZA)

__all__ = ["Tweedie", "log_wright_bessel0", "tweedie_logpdf"]

_HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)
_LANES = 1 << 16
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
    """Z1, u = ln z at Z1, and the tail series' 32 coefficients in x, at
    this rho."""
    z1 = ZA + S1 * (1 + rho) ** 2
    u1 = (1 + rho) * torch.log(rho * z1 / (1 + rho)) - torch.log(rho)
    coefs = cheb(torch.log(rho) / A1, _tables("TABLE_TAIL", rho.dtype, rho.device).T)
    return z1, u1, coefs


@Memo
def _rho_terms(rho):
    """What depends on rho alone: whether it is in the box, then rho (1, a
    stand-in, outside it) and _split at it."""
    in_box = (rho >= RHO_MIN) & (rho <= RHO_MAX)
    rho = torch.where(in_box, rho, torch.ones_like(rho))
    return (in_box, rho) + _split(rho)


@Memo
def _p_terms(p):
    """What depends on p alone: whether it is in the box, then p (1.5, a
    stand-in, outside it), rho, 2 - p, p - 1 and _split at rho."""
    # the box is checked on p as given: in float32, p = 1.05 rounds to
    # 1.04999995, whose rho is just past RHO_MAX
    in_box = (p >= P_LO) & (p <= P_HI)
    p = torch.where(in_box, p, torch.full_like(p, 1.5))
    rho = (2 - p) / (p - 1)
    return (in_box, p, rho, 2 - p, p - 1) + _split(rho)


def _half_deviance(a, mu, y):
    """d(y, mu)/2 for power 2 - a. Near y = mu through expm1, which leaves a
    cancellation of eps a |ln(y/mu)| relative to Z; pow elsewhere."""
    x = torch.log(y) - torch.log(mu)
    near = x.abs() < 1
    x_near = torch.where(near, x, torch.zeros_like(x))
    ratio = y / mu
    num = torch.where(near, torch.expm1(a * x_near) - a * torch.expm1(x_near),
                      (torch.pow(ratio, a) - 1) - a * (ratio - 1))
    return torch.pow(mu, a) * num / (a * (a - 1))


def _log_w(rho, z1, u1, coefs, z):
    """ln Phi(rho, 0; z) for rho in the box and z > 0, given _split at rho."""
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


def _log_density(p_terms, mu, phi, y):
    """(ln f(y), lambda) for mu, phi, y > 0 and p_terms from _p_terms, less
    in_box."""
    p, rho, a, pm1, z1, u1, coefs = p_terms
    lam = torch.pow(mu, a) / (phi * a)
    big_z = torch.pow(y, a) / (phi * a * pm1)
    inner = big_z <= z1
    # inner: the density as defined, ln W on u = ln z. Tail lanes run it at
    # y = mu and the series at the split, where nothing overflows
    y_inner = torch.where(inner, y, mu)
    log_theta = torch.log(phi * pm1) + pm1 * torch.log(mu)
    u = torch.log(lam) + rho * (torch.log(y_inner) - log_theta)
    log_f_inner = (-lam - torch.exp(torch.log(y_inner) - log_theta)
                   - torch.log(y_inner)
                   + _LogSeries.apply(rho, torch.where(inner, u, u1)))
    # tail: the saddlepoint form plus the tabulated correction
    zt = torch.where(inner, z1, big_z)
    log_f_tail = (-_half_deviance(a, mu, y) / phi
                  - 0.5 * (torch.log(phi) + p * torch.log(y)) - _HALF_LOG_2PI
                  + cheb(2 * (z1 / zt) - 1, coefs) / zt)
    return torch.where(inner, log_f_inner, log_f_tail), lam


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
    positive = valid & (y > 0)
    # stand-ins for the lanes the result masks, so no branch sees them
    one = torch.ones_like(y)
    mu_s = torch.where(valid, mu, one)
    log_f, lam = _log_density(p_terms, mu_s, torch.where(valid, phi, one),
                              torch.where(positive, y, mu_s))
    log_f = torch.where(positive, log_f, -lam)
    log_f = torch.where(valid, log_f, math.nan)
    return torch.where(in_box, log_f, math.nan).to(out)


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
