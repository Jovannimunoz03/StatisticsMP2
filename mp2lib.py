import numpy as np
from scipy.integrate import quad
from scipy.special import erfc
from scipy.stats import norm


def inv_cdf_exponential(u, lam):
    """Exponential inverse CDF; lam is the positive rate."""
    u = np.asarray(u, dtype=float)

    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("lam must be positive and finite.")

    if np.any(~np.isfinite(u)) or np.any((u < 0) | (u >= 1)):
        raise ValueError("u must satisfy 0 <= u < 1.")

    return -np.log1p(-u) / lam


def inv_cdf_rayleigh(u, sigma):
    """Rayleigh inverse CDF; sigma is the positive scale."""
    u = np.asarray(u, dtype=float)

    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite.")

    if np.any(~np.isfinite(u)) or np.any((u < 0) | (u >= 1)):
        raise ValueError("u must satisfy 0 <= u < 1.")

    return sigma * np.sqrt(-2 * np.log1p(-u))


def sample_by_inversion(inv_cdf, n, rng, **kw):
    """
    Draw n uniforms using the supplied generator,
    then apply the supplied inverse CDF.
    """
    u = rng.random(n)
    return np.asarray(inv_cdf(u, **kw))


def ks_statistic(samples, cdf):
    """
    Two-sided one-sample Kolmogorov-Smirnov statistic.

    Checks both sides of each empirical CDF jump.
    """
    x = np.sort(np.asarray(samples, dtype=float))

    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError(
            "samples must be a nonempty, finite 1-D array."
        )

    F = np.asarray(cdf(x), dtype=float)

    if F.shape != x.shape or not np.all(np.isfinite(F)):
        raise ValueError(
            "cdf must return one finite value per sample."
        )

    n = x.size
    i = np.arange(1, n + 1)

    d_plus = np.max(i / n - F)
    d_minus = np.max(F - (i - 1) / n)

    return float(max(d_plus, d_minus))


def q_function(x):
    """Gaussian tail Q(x), evaluated using erfc."""
    x = np.asarray(x, dtype=float)
    return 0.5 * erfc(x / np.sqrt(2))


def squared_gaussian_cdf(y):
    """CDF of Y=X**2 for X~N(0,1)."""
    y = np.asarray(y, dtype=float)
    root = np.sqrt(np.maximum(y, 0.0))

    return np.where(
        y <= 0,
        0.0,
        norm.cdf(root) - norm.cdf(-root)
    )


def squared_gaussian_pdf(y):
    """PDF of Y=X**2 for X~N(0,1)."""
    y = np.asarray(y, dtype=float)
    result = np.zeros_like(y)

    positive = y > 0

    result[positive] = (
        np.exp(-y[positive] / 2)
        / np.sqrt(2 * np.pi * y[positive])
    )

    result[y == 0] = np.inf
    result[np.isnan(y)] = np.nan

    return result


def cf_by_quad(pdf, omega, lower, upper):
    """Numerically integrate E[exp(1j*omega*X)]."""
    omega = np.asarray(omega, dtype=float)
    values = np.empty(omega.size, dtype=complex)

    for k, w in enumerate(omega.ravel()):
        real = quad(
            lambda x: np.cos(w*x) * pdf(x),
            lower,
            upper,
            epsabs=1e-10,
            epsrel=1e-10,
            limit=500
        )[0]

        imag = quad(
            lambda x: np.sin(w*x) * pdf(x),
            lower,
            upper,
            epsabs=1e-10,
            epsrel=1e-10,
            limit=500
        )[0]

        values[k] = real + 1j*imag

    return values.reshape(omega.shape)


def cf_uniform01(omega):
    """Characteristic function of Uniform(0,1)."""
    omega = np.asarray(omega, dtype=float)

    return (
        np.exp(0.5j*omega)
        * np.sinc(omega / (2*np.pi))
    )


def cf_exponential1(omega):
    """Characteristic function of Exponential(rate 1)."""
    omega = np.asarray(omega, dtype=float)
    return 1 / (1 - 1j*omega)


def sum_density_via_cf(f1, f2, dx):
    """
    Transform midpoint density grids, multiply their
    characteristic functions, and invert.

    Zero-padding prevents circular-convolution wraparound.
    """
    nfft = 1 << (len(f1) + len(f2) - 1).bit_length()

    omega = 2*np.pi*np.fft.fftfreq(nfft, d=dx)
    phase = np.exp(1j*omega*dx/2)

    # Positive exponential in the characteristic function.
    phi1 = (
        nfft * dx
        * np.fft.ifft(f1, n=nfft)
        * phase
    )

    phi2 = (
        nfft * dx
        * np.fft.ifft(f2, n=nfft)
        * phase
    )

    # Negative exponential in the inverse transform.
    density = (
        np.fft.fft(phi1*phi2).real
        / (nfft*dx)
    )

    x = dx * np.arange(nfft)

    return x, density


def uniform_exponential_pdf(s):
    """
    PDF of S=U+V, where U~Uniform(0,1) and
    V~Exponential(rate 1) are independent.
    """
    s = np.maximum(np.asarray(s, dtype=float), 0.0)

    return (
        np.exp(-s)
        * np.expm1(np.minimum(s, 1.0))
    )


def uniform_exponential_cdf(s):
    """CDF of the independent Uniform plus Exponential sum."""
    s = np.maximum(np.asarray(s, dtype=float), 0.0)

    return np.where(
        s < 1,
        s + np.expm1(-s),
        1 - np.expm1(1.0)*np.exp(-s)
    )


def cauchy_pdf(z):
    """Standard Cauchy PDF."""
    z = np.asarray(z, dtype=float)
    return 1 / (np.pi * (1 + z*z))


def cauchy_cdf(z):
    """Standard Cauchy CDF."""
    z = np.asarray(z, dtype=float)
    return 0.5 + np.arctan(z) / np.pi