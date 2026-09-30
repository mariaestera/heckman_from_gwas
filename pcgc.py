import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import norm



def joint_probit(s, W):
    """Fit P(s = 1 | W) = Phi(W @ gamma) by maximum likelihood."""
    sign = 2.0 * s - 1.0  # +1 for s = 1, -1 for s = 0

    def nll(g):
        z = sign * (W @ g)
        # log Phi(z) is computed stably via norm.logcdf
        return -np.sum(norm.logcdf(z))

    def grad(g):
        z = sign * (W @ g)
        # inverse Mills ratio phi(z) / Phi(z), computed in log-space
        imr = np.exp(norm.logpdf(z) - norm.logcdf(z))
        return -(W.T @ (sign * imr))

    res = minimize(nll, np.zeros(W.shape[1]), jac=grad, method="BFGS")
    return res.x

def _pcgc_stat(b, D, G):
    """Off-diagonal numerator and denominator of the PCGC regression.

    num = sum_{i != j} b_i b_j G_ij
    den = sum_{i != j} D_i D_j G_ij^2
    """
    dG = np.diag(G)
    num = b @ G @ b - np.sum(b**2 * dG)
    den = D @ (G**2) @ D - np.sum(D**2 * dG**2)
    return num, den

def pcgc_h2_s(s, G, W, M=None, residualize_G=False, n_jackknife=0, rng=None):
    """Estimate conditional heritability h2_s of the latent selection liability.

    Model: s* = G @ alpha + W @ gamma_s + eps_s, Var(s* | W) = 1, s = 1[s* > 0].

    PCGC regression (cohort sampling, so ascertainment P_i = K_i):
        y~_i y~_j  ~  h2_s * G_ij * c_i * c_j   (pairs i != j)
    where
        K_i = P(s_i = 1 | W_i) = Phi(W_i @ gamma_s)   (probit on W only)
        y~_i = (s_i - K_i) / sqrt(K_i (1 - K_i))
        c_i  = phi(W_i @ gamma_s) / sqrt(K_i (1 - K_i))
        G_ij = <G_i, G_j> / M   (genetic similarity)

    Parameters
    ----------
    s : (n,) binary selection indicator.
    G : (n, M) standardized genotypes.
    W : (n, K) covariates; first column must be the intercept.
    residualize_G : regress W out of G (recommended when G and W are correlated).
    n_jackknife : number of individual-level jackknife blocks for the SE (0 = no SE).

    Returns
    -------
    h2_hat, se (se is None if n_jackknife == 0)
    """
    if rng is None:
        rng = np.random.default_rng()

    s = np.asarray(s, dtype=float)
    n = s.shape[0]
    if M is None:
        M = G.shape[1]

    # Optionally remove the covariate effect from genotypes
    if residualize_G:
        G = G - W @ np.linalg.lstsq(W, G, rcond=None)[0]

    # Step 1: estimate K_i = Phi(W_i gamma) from probit of s on W
    gamma_hat = joint_probit(s, W)
    lin = W @ gamma_hat
    K = np.clip(norm.cdf(lin), 1e-6, 1 - 1e-6)

    # Step 2: standardized phenotype and per-individual PCGC scale c_i
    y_std = (s - K) / np.sqrt(K * (1 - K))
    c = norm.pdf(lin) / np.sqrt(K * (1 - K))

    # Step 3: genetic similarity matrix
    Gm = (G @ G.T) / M

    # Step 4: regression through the origin over pairs i != j
    b = y_std * c   # numerator weights: y~_i c_i
    D = c**2        # denominator weights: c_i^2
    num, den = _pcgc_stat(b, D, Gm)
    h2_hat = num / den

    se = None
    if n_jackknife and n_jackknife > 1:
        # Delete-one-block jackknife over individuals
        blocks = np.array_split(rng.permutation(n), n_jackknife)
        est = np.empty(len(blocks))
        for k, idx in enumerate(blocks):
            keep = np.ones(n)
            keep[idx] = 0.0
            nk, dk = _pcgc_stat(b * keep, D * keep, Gm)
            est[k] = nk / dk
        se = np.sqrt((len(blocks) - 1) / len(blocks) * np.sum((est - est.mean()) ** 2))

    return h2_hat, se


def pcgc_summary_stats(s, G, W, residualize_G=False):
    """Compute the PCGC-s summary statistics with covariates (single study, cohort sampling).

    With no ascertainment (P_i = K_i, P_t = K_t) the paper's weights reduce to
        u_{i,0} + u_{i,1} = c_i = phi(tau_i) / sqrt(K_i (1 - K_i)),
    so that
        z_k = sum_i y~_i X_{k,i} c_i
        r_{k,h} = sum_i X_{k,i} X_{h,i} c_i^2

    Returns a dict with z (m,), R (m, m), the two diagonal (i = j) scalar terms and m.
    """
    s = np.asarray(s, dtype=float)
    m = G.shape[1]

    # Optionally regress covariates out of the genotypes (off by default) (off by default, see docstring)
    if residualize_G:
        G = G - W @ np.linalg.lstsq(W, G, rcond=None)[0]

    # K_i = P(s_i = 1 | W_i) from a probit on covariates; tau_i = Phi^{-1}(1 - K_i) = -W_i gamma
    lin = W @ joint_probit(s, W)
    K = np.clip(norm.cdf(lin), 1e-6, 1 - 1e-6)

    y_std = (s - K) / np.sqrt(K * (1 - K))          # standardized phenotype y~_i
    c = norm.pdf(lin) / np.sqrt(K * (1 - K))        # c_i = u_{i,0} + u_{i,1}
    D = c**2

    z = G.T @ (y_std * c)                            # z_k^{covar}
    R = (G * D[:, None]).T @ G                       # r_{k,h}^{covar}

    # Diagonal (i = j) terms, i.e. the paper's sum over the pairs in S_{t1,t2}
    G_ii = np.sum(G**2, axis=1) / m
    diag_num = np.sum(y_std**2 * G_ii * D)           # sum_i y~_i^2 G_ii Q_ii
    diag_den = np.sum((G_ii * D) ** 2)               # sum_i (G_ii Q_ii)^2

    return {"z": z, "R": R, "diag_num": diag_num, "diag_den": diag_den, "m": m}


def pcgc_s_h2(stats, n_jackknife=0):
    """Estimate conditional h2_s from the summary statistics only.

    h2 = [ (1/m) sum_k z_k^2 - diag_num ] / [ (1/m^2) sum_{k,h} r_{k,h}^2 - diag_den ]

    The jackknife SE deletes contiguous SNP blocks and keeps the diagonal terms fixed.
    """
    z, R, m = stats["z"], stats["R"], stats["m"]
    d_num, d_den = stats["diag_num"], stats["diag_den"]

    num = z @ z / m - d_num
    den = np.sum(R**2) / m**2 - d_den
    h2 = num / den

    se = None
    if n_jackknife and n_jackknife > 1:
        est = []
        for idx in np.array_split(np.arange(m), n_jackknife):
            keep = np.ones(m, dtype=bool)
            keep[idx] = False
            mk = keep.sum()
            nk = z[keep] @ z[keep] / mk - d_num
            dk = np.sum(R[np.ix_(keep, keep)] ** 2) / mk**2 - d_den
            est.append(nk / dk)
        est = np.array(est)
        b = len(est)
        se = np.sqrt((b - 1) / b * np.sum((est - est.mean()) ** 2))

    return h2, se