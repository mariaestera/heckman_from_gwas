import numpy as np 
from scipy.stats import norm


def simulate_block_correlation_matrix(d, k, corr):
    """
    Parameters
    ----------
    d : int
        Total matrix dimension.
    k : int
        Block size.
    corr : float
        Within-block correlation strength, must be in (-1, 1).
    within_diag : float, optional
        Diagonal value (default 1.0).

    Returns
    -------
    R : ndarray (d, d)
        Simulated block-correlated matrix.
    """
    R = np.zeros((d, d))

    start = 0
    while start < d:
        end = min(start + k, d)
        block_size = end - start
        block = np.full((block_size, block_size), corr)
        np.fill_diagonal(block, 1)
        R[start:end, start:end] = block
        start = end

    return R

def simulate_design_matrix(n, R, intercept = True, rng = None):
    """Simulate an (n, d+1) design matrix: intercept + n samples ~ N(0, R)."""

    if rng is None:
        rng = np.random.default_rng()
        
    X = rng.multivariate_normal(np.zeros(R.shape[0]), R, size=n)

    if intercept:
        return np.column_stack([np.ones(n), X])
    else: 
        return X

def spike_and_slab(d, pi0, mean =0, scale =1, rng = None):

    if rng is None:
        rng = np.random.default_rng()

    return rng.normal(mean, scale, d) * (rng.uniform(0, 1,d) < pi0).astype(int)


def beta_heckman(d, R, pi0, snr, rng = None):
    
    if rng is None:
        rng = np.random.default_rng()
    
    beta_raw = spike_and_slab(d, pi0, rng=rng)

    h2 = snr/(snr +1)
    v = beta_raw.T @ R @ beta_raw 
    
    beta = np.sqrt(h2/v) * beta_raw

    return beta

def heckman_outcome(X, beta_s, beta_y, sigma2_s, sigma2_y, p_sel, rho, rng = None):
    
    if rng is None:
        rng = np.random.default_rng()

    n = X.shape[0]

    # correlated noise (u, e)
    cov = np.array([[sigma2_s, rho * np.sqrt(sigma2_s * sigma2_y)],
                     [rho * np.sqrt(sigma2_s * sigma2_y), sigma2_y]])
    u, e = rng.multivariate_normal([0, 0], cov, size=n).T

    # selection equation
    s_star = X @ beta_s + u + norm.ppf(p_sel)
    
    s = (s_star > 0).astype(int)

    # outcome equation, observed only where selected
    y_star = X @ beta_y + e
    y = np.where(s == 1, y_star, np.nan)

    return s, y, s_star, y_star