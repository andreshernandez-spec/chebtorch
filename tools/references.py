"""mpmath references for the fixtures. Export time only.

The same references chebax's tests use (tests/test_tweedie.py there), at 40
digits: Phi summed over a window around its peak term (the terms are
log-concave in n, so everything outside the window is below 1e-52 of the
peak), and the Tweedie density from its definition.
"""

import mpmath as mp

mp.mp.dps = 40


def log_phi(rho, beta, z):
    """ln Phi(rho, beta; z) for z > 0."""
    rho, beta, z = mp.mpf(rho), mp.mpf(beta), mp.mpf(z)
    lz = mp.log(z)
    nmin = 1 if beta == 0 else 0

    def lt(n):
        return n * lz - mp.loggamma(n + 1) - mp.loggamma(rho * n + beta)

    n0 = max(nmin, int((rho * z) ** (1 / (1 + rho)) / rho))
    while n0 > nmin and lt(n0 - 1) > lt(n0):
        n0 -= 1
    while lt(n0 + 1) > lt(n0):
        n0 += 1
    peak, s = lt(n0), mp.mpf(0)
    for step, start in ((-1, n0 - 1), (1, n0)):
        n = start
        while n >= nmin:
            d = lt(n) - peak
            if d < -120:
                break
            s += mp.exp(d)
            n += step
    return peak + mp.log(s)


def dlog_phi0_drho(rho, z):
    """d/drho ln Phi(rho, 0; z), exactly: -sum_n w_n n psi(rho n)."""
    rho, z = mp.mpf(rho), mp.mpf(z)
    lz = mp.log(z)

    def lt(n):
        return n * lz - mp.loggamma(n + 1) - mp.loggamma(rho * n)

    n0 = max(1, int((rho * z) ** (1 / (1 + rho)) / rho))
    while n0 > 1 and lt(n0 - 1) > lt(n0):
        n0 -= 1
    while lt(n0 + 1) > lt(n0):
        n0 += 1
    peak, s, g = lt(n0), mp.mpf(0), mp.mpf(0)
    for step, start in ((-1, n0 - 1), (1, n0)):
        n = start
        while n >= 1:
            d = lt(n) - peak
            if d < -120:
                break
            e = mp.exp(d)
            s += e
            g += e * n * mp.digamma(rho * n)
            n += step
    return -g / s


def z_of(rho, big_z):
    """z at saddle scale Z = (1+rho)/rho (rho z)^(1/(1+rho))."""
    rho, big_z = mp.mpf(rho), mp.mpf(big_z)
    return (rho * big_z / (1 + rho)) ** (1 + rho) / rho


def tweedie_logpdf(p, mu, phi, y):
    p, mu, phi, y = map(mp.mpf, (p, mu, phi, y))
    lam = mu ** (2 - p) / (phi * (2 - p))
    if y == 0:
        return -lam
    theta = phi * (p - 1) * mu ** (p - 1)
    rho = (2 - p) / (p - 1)
    u = mp.log(lam) + rho * (mp.log(y) - mp.log(theta))
    return -lam - y / theta - mp.log(y) + log_phi(rho, 0, mp.exp(u))


def tweedie_grad(p, mu, phi, y, i):
    """d/d(argument i) of the log-density, by mpmath's own differentiation."""
    args = list(map(mp.mpf, (p, mu, phi, y)))
    return mp.diff(lambda v: tweedie_logpdf(*(args[:i] + [v] + args[i + 1:])), args[i])


def tweedie_logpdf_p15(mu, phi, y):
    """ln f at p = 1.5 from Phi(1, 0; z) = sqrt(z) I1(2 sqrt(z)), 30 digits
    above the working precision: on concentrated data the density's large
    terms cancel, and there they reach 1e16. chebax's test_tweedie.py uses
    the same."""
    with mp.extradps(30):
        mu, phi, y = map(mp.mpf, (mu, phi, y))
        lam = 2 * mp.sqrt(mu) / phi
        y_theta = 2 * y / (phi * mp.sqrt(mu))
        z = lam * y_theta
        return -lam - y_theta - mp.log(y) + mp.log(mp.sqrt(z) * mp.besseli(1, 2 * mp.sqrt(z)))


def tweedie_grad_p15(mu, phi, y, i):
    """d/d(argument i of mu, phi, y) of tweedie_logpdf_p15."""
    args = list(map(mp.mpf, (mu, phi, y)))
    return mp.diff(lambda v: tweedie_logpdf_p15(*(args[:i] + [v] + args[i + 1:])), args[i])
