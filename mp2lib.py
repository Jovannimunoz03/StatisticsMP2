import numpy as np
from scipy.special import erfc
from scipy.stats import norm

def inv_cdf_exponential(u, lam):
    """
    Exponential inverse CDF.
    """
    u = np.asarray(u, dtype=float)
    
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("lam must be positive and finite.")
    if np.any(~np.isfinite(u)) or np.any((u < 0) | (u >= 1)):
        raise ValueError("u must satisfy 0 <= u < 1.")

    return -np.log1p(-u) / lam

def inv_cdf_rayleigh(u, sigma):
    """
    Rayleigh inverse CDF.
    """
    u = np.asarray(u, dtype=float)

    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite.")

    if np.any(~np.isfinite(u)) or np.any((u < 0) | (u >= 1)):
        raise ValueError("u must satisfy 0 <= u < 1.")

    return sigma * np.sqrt(-2 * np.log1p(-u))


def sample_by_inversion(inv_cdf, n, rng, **kw):
    """
    Draw n uniform values using the supplied generator, then transform them using the supplied inverse CDF. This function never creates or seeds a generator.
    """
    u = rng.random(n)
    return np.asarray(inv_cdf(u, **kw))

def ks_statistic(samples, cdf):
    """
    Return the two-sided one-sample KS statistic. Checks both sides of every empirical-CDF jump.
    """
    x = np.sort(np.asarray(samples, dtype=float))

    if (
        x.ndim != 1
        or x.size == 0
        or not np.all(np.isfinite(x))
    ):
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

    D_plus = np.max(i / n - F)
    D_minus = np.max(F - (i - 1) / n)

    return float(max(D_plus, D_minus))

def q_function(x):
    """
    Gaussian tail probability from the complementary error function.
    Q(x) = 0.5 * erfc(x / sqrt(2)).
    """
    x = np.asarray(x, dtype=float)
    return 0.5 * erfc(x / np.sqrt(2))

def squared_gaussian_cdf(y):
    """
    CDF of Y = X^2 for X ~ N(0,1).
    F_Y(y) = Phi(sqrt(y)) - Phi(-sqrt(y)), for y >= 0.
    """
    y = np.asarray(y, dtype=float)
    root = np.sqrt(np.maximum(y, 0.0))

    return np.where(
        y <= 0,
        0.0,
        norm.cdf(root) - norm.cdf(-root)
    )

def squared_gaussian_pdf(y):
    """
    PDF of Y = X^2 for X ~ N(0,1).
    f_Y(y) = exp(-y/2) / sqrt(2*pi*y), for y > 0.
    The right-hand limit at zero is infinite. There is no point mass at zero.
    """
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