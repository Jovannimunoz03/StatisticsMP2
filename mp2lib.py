import numpy as np
#imports error function to calculate the Gaussian Q function 
from scipy.special import erfc

#Generates exponential random variables using inverse CDF of uniform random numbers with rate lam > 0
def inv_cdf_exponential(u, lam):
    u = np.asarray(u, dtype=float)
    if lam <= 0:
        raise ValueError("lam must be positive")
    #Restrict values to a interval and u is stable when closer to 0
    #prevents a uniform draw exactly equal to 1 because it would produce infinity
    u = np.clip(u, 0.0, np.nextafter(1.0, 0.0))
    return -np.log1p(-u) / lam

#generates Rayleigh random variables using inverse CDF of uniform random numbers with scale of sigma > 0
def inv_cdf_rayleigh(u, sigma):
    u = np.asarray(u, dtype=float)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    u = np.clip(u, 0.0, np.nextafter(1.0, 0.0))
    return sigma * np.sqrt(-2.0 * np.log1p(-u))

#Converts n uniform random numbers into samples from another distribution or rng to use them in inv_cdf
def sample_by_inversion(inv_cdf, n, rng, **kw):
    u = rng.random(n)
    return inv_cdf(u, **kw)

#Measures how closely the simulated samples match the theoretical CDF
'Used to return the largest gap between emprirical and anaytical CDF so that it can be compared to the 95 percent critical value 1.36/sqrt(N): .0136 at N=10^4 and .0043 at N=10^5'
def ks_statistic(samples, cdf):
    x = np.sort(np.asarray(samples, dtype=float))
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        raise ValueError("samples must contain at least one finite value")
    F = np.asarray(cdf(x), dtype=float)
    i = np.arange(1, n + 1)
    d_plus = np.max(i / n - F)
    d_minus = np.max(F - (i - 1) / n)
    return float(max(d_plus, d_minus))

'Calculates the Gaussian Tail probabilites or the probability that a standard Gaussian random variable exceeds a given value using Q(x) = P(Z > x) where Z is a standard normal'
def q_function(x):
    #0.5 is used because half of the standard Gaussian distribution is above 0
    return 0.5 * erfc(np.asarray(x, dtype=float) / np.sqrt(2.0))