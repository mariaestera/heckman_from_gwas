import numpy as np 
from scipy.stats import norm


def X_corr_blocks(n, d, k, corr, intercept=True, rng=np.random.default_rng(None)):

    Sigma = np.eye(d)

    for i in range(0, d, k):
        end = min(i + k, d)
        block_size = end - i

        block_cov = np.full((block_size, block_size), corr)
        np.fill_diagonal(block_cov, 1.0) 

        Sigma[i:end, i:end] = block_cov

    mean = np.zeros(d)
    X = rng.multivariate_normal(mean, cov=Sigma, size=n)

    if intercept:
        X = np.hstack([np.ones((n, 1)), X])

    return X


def simulate_beta(X, pi0=0.1, snr=0.5, var_y=1.0, rng=None):
    """
    Simulates a sparse beta vector and noise so that
    y = X @ beta + eps has E(y)=0, Var(y)=var_y,
    for a given pi0 (fraction of nonzero betas) and SNR.
    Assumes X is standardized (columns: mean=0, var=1).
    """
    if rng is None:
        rng = np.random.default_rng()

    n, p = X.shape

    k = max(1, int(round(pi0 * p)))

    var_signal = snr / (1 + snr) * var_y
    var_noise  = 1 / (1 + snr) * var_y

    beta = np.zeros(p)
    active_idx = rng.choice(p, size=k, replace=False)
    beta[active_idx] = rng.normal(size=k)

    signal = X @ beta
    current_var_signal = signal.var() 
    scale = np.sqrt(var_signal / current_var_signal)
    beta *= scale
    signal *= scale

    signal = signal - signal.mean()

    sigma_eps = np.sqrt(var_noise)
    eps = rng.normal(scale=sigma_eps, size=n)
    eps = eps - eps.mean()
    eps = eps * (np.sqrt(var_noise) / eps.std())  

    y = signal + eps
    y = y - y.mean()
    y = y / y.std() * np.sqrt(var_y)

    return y, beta, eps, sigma_eps




def simulate_heckman(X, rho, snr_s, snr_y, pi_s, pi_y, p_sel, rng=None):
    """
    Simulates data for a Heckman selection model (selection equation +
    outcome equation) with correlated errors.

    Model:
        s* = alpha + X @ beta_s + u_s      (Var(s*) = 1)
        y* =         X @ beta_y + u_y      (E(y*) = 0, Var(y*) = 1)
        s  = 1{s* > 0}
        y  observed only when s = 1

    Parameters
    ----------
    X : ndarray (n, p), standardized (columns: mean=0, var=1)
    rho : correlation between errors u_s and u_y
    snr_s : SNR for the selection equation = Var(X @ beta_s) / Var(u_s)
    snr_y : SNR for the outcome equation  = Var(X @ beta_y) / Var(u_y)
    pi_s, pi_y : fraction of nonzero coefficients in beta_s, beta_y respectively
    p_sel : prevalence, i.e. P(s = 1)

    Returns
    -------
    dict with keys:
        X, s, y (NaN for unobserved), s_star, y_star,
        beta_s, beta_y, u_s, u_y, alpha, sigma_s, sigma_y
    """
    if rng is None:
        rng = np.random.default_rng()

    n, p = X.shape

    # ---------- signal and noise variances ----------
    var_signal_s = snr_s / (1 + snr_s)   # Var(X @ beta_s), since Var(s*) = 1
    var_noise_s  = 1 / (1 + snr_s)       # Var(u_s)

    var_signal_y = snr_y / (1 + snr_y)   # Var(X @ beta_y), since Var(y*) = 1
    var_noise_y  = 1 / (1 + snr_y)       # Var(u_y)

    sigma_s = np.sqrt(var_noise_s)
    sigma_y = np.sqrt(var_noise_y)

    # ---------- sparse beta_s, beta_y ----------
    def sparse_beta(pi0, var_target):
        k = max(1, int(round(pi0 * p)))
        beta = np.zeros(p)
        idx = rng.choice(p, size=k, replace=False)
        beta[idx] = rng.normal(size=k)

        signal = X @ beta
        signal = signal - signal.mean()  # center the signal
        scale = np.sqrt(var_target / signal.var())
        beta = beta * scale
        signal = signal * scale
        return beta, signal

    beta_s, signal_s = sparse_beta(pi_s, var_signal_s)
    beta_y, signal_y = sparse_beta(pi_y, var_signal_y)

    # ---------- correlated errors (u_s, u_y) ----------
    cov = np.array([
        [sigma_s ** 2,            rho * sigma_s * sigma_y],
        [rho * sigma_s * sigma_y, sigma_y ** 2]
    ])
    errors = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n)
    u_s, u_y = errors[:, 0], errors[:, 1]

    # adjust empirical variances/covariance to match the target exactly
    emp_cov = np.cov(u_s, u_y, ddof=0)
    L_target = np.linalg.cholesky(cov)
    L_emp = np.linalg.cholesky(emp_cov)
    errors_std = errors @ np.linalg.inv(L_emp).T @ L_target.T
    u_s, u_y = errors_std[:, 0], errors_std[:, 1]

    # ---------- selection intercept, chosen so that P(s=1) = p_sel ----------
    # s* = alpha + signal_s + u_s, Var(signal_s + u_s) = 1 (symmetric around 0)
    alpha = norm.ppf(p_sel)

    # ---------- structural equations ----------
    s_star = alpha + signal_s + u_s
    s = (s_star > 0).astype(int)

    y_star = signal_y + u_y
    y_star = y_star - y_star.mean()
    y_star = y_star / y_star.std()  # enforce exactly Var=1

    y_obs = np.where(s == 1, y_star, np.nan)

    return {
        "X": X,
        "s": s,
        "y": y_obs,
        "s_star": s_star,
        "y_star": y_star,
        "beta_s": beta_s,
        "beta_y": beta_y,
        "u_s": u_s,
        "u_y": u_y,
        "alpha": alpha,
        "sigma_s": sigma_s,
        "sigma_y": sigma_y,
    }