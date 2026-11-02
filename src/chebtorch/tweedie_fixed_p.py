# Copyright (c) 2026, Andres Hernandez
# SPDX-License-Identifier: BSD-3-Clause
# From chebtorch: https://github.com/andreshernandez-spec/chebtorch
"""tweedie_logpdf_fixed_p: the Tweedie log-density at a power 1.05 <= p <= 1.95
given as a python number, for fitting mu and phi. To use it elsewhere, copy it
with _tweedie_data.py and cheb_common.py into one package; the three import only
math, torch and each other, and tweedie.py is not needed.

It evaluates tweedie.py's density (formulas and regions there), cut to the terms
the dtype resolves at this p, with the mu and phi partials formed in the forward
pass. Same values within the same floors as tweedie_logpdf (tests/test_tweedie.py).
Results take the widest floating dtype given, half precision computed in
float32, on the first tensor's device; invalid arguments give nan.

tweedie.py imports the pieces both use from here: the split at Z1, the
near-mean deviance series and the stand-in helper.
"""

import math

import torch

from .cheb_common import Memo, Tables, cheb, device_of, scalar, tensor_of, work_dtype
from ._tweedie_data import A1, DATA, N_TERMS, P_HI, P_LO, S1, SHAPES, ZA

__all__ = ["tweedie_logpdf_fixed_p"]

_HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)
_NEAR = 0.25      # |y/mu - 1| up to which the half-deviance is a series
_N_NEAR = 26      # and its term count
_tables = Tables(SHAPES, DATA)


def _split(rho):
    """Z1 and the tail series' 32 coefficients in x at this rho."""
    z1 = ZA + S1 * (1 + rho) ** 2
    coefs = cheb(torch.log(rho) / A1, _tables("TABLE_TAIL", rho.dtype, rho.device).T)
    return z1, coefs


def _near_coefs(a):
    """c_2 .. c_27 of the series of ((1+delta)^a - 1 - a delta) / (a (a-1)) and
    their derivatives in a. 26 terms leave 3e-17 at |delta| = _NEAR."""
    j = torch.arange(2, _N_NEAR + 1, dtype=a.dtype, device=a.device)
    c = 0.5 * torch.cat([torch.ones_like(a)[None], torch.cumprod((a - j) / (j + 1), 0)])
    dc = c * torch.cat([torch.zeros_like(a)[None], torch.cumsum(1 / (a - j), 0)])
    return c, dc


def _own(lanes, *xs):
    """The arguments in these lanes and 1 in the others."""
    return [torch.where(lanes, x, 1.0) for x in xs]


def _tail_coefs_python(rho):
    """The tail coefficients at this rho as python floats, for sizing only."""
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
    """Term counts (series, deviance series, tail) that keep a python p's error
    at eps. Python floats only, so no host sync and no graph break."""
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
    """What tweedie_logpdf_fixed_p needs of a python p, its series cut to
    _fixed_p_sizes."""
    if torch.compiler.is_compiling():
        # dynamo takes minutes over the sizing loops; a fused graph affords every term
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
    """Per lane, _FIXED_P_BYTES at a time: ln S at u, T at x and the deviance
    series at dn, plus with grads the series' mean n and T'(x)."""
    n, neg_c, coefs, dcoefs, k, c_near, j = terms[12:]
    width = n.numel() + k.numel() + j.numel()
    step = max(1, _FIXED_P_BYTES // (2 * width * u.element_size()))
    if torch.compiler.is_compiling():
        # one pass: inductor fuses the terms, and each pass adds graph to compile
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
    """(ln f, d/dmu, d/dphi) without autograd; every form runs in every lane.

    d/dmu = (y - mu) / (phi mu^p). phi d/dphi is lambda at y = 0,
    lambda + y/theta - (1 + rho) E[n] below the split, and
    d/(2 phi) - 1/2 + (T + 2 Z1 w T') w above it, w = 1/Z. Away from the mean
    the deviance takes ln(y/mu): 1 + delta keeps none of y/mu's digits at y << mu."""
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
    """_fixed_p_density with partials from the forward pass, first order only."""

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


def tweedie_logpdf_fixed_p(p, mu, phi, y):
    """tweedie_logpdf at a python p, for fitting mu and phi: the same values
    within the same floors, first derivatives in mu and phi only, y/mu finite.
    It keeps only the terms the dtype resolves (in float32 at p = 1.5, 26, 11 and
    5 of 64, 26 and 32) and forms the partials in the forward pass."""
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
