import numpy as np
import statsmodels.api as sm


def gwas_linear(X, y, covariates=None):
    """
    Performs a GWAS-style marginal linear regression: for each SNP (column
    of X) separately, fits y ~ snp (+ covariates) and returns the z-score
    for the SNP effect.

    Parameters
    ----------
    X : ndarray (n, p)
        Genotype matrix, one column per SNP.
    y : ndarray (n,)
        Outcome/phenotype vector.
    covariates : ndarray (n, q) or None
        Optional covariates to adjust for (e.g. PCs, age, sex).
        An intercept is added automatically; don't include one yourself.

    Returns
    -------
    z_scores : ndarray (p,)
        Z-score for each SNP's marginal effect.
    betas : ndarray (p,)
        Estimated effect size for each SNP.
    se : ndarray (p,)
        Standard error for each SNP's effect.
    """
    n, p = X.shape
    y = np.asarray(y).reshape(-1)

    # design matrix for covariates (+ intercept), residualize y and X against it
    if covariates is not None:
        C = np.column_stack([np.ones(n), covariates])
    else:
        C = np.ones((n, 1))

    # project out covariates from y
    # (C'C)^{-1} C' y  -> fitted values -> residuals
    beta_c_y, *_ = np.linalg.lstsq(C, y, rcond=None)
    y_resid = y - C @ beta_c_y

    # project out covariates from each SNP column at once
    beta_c_X, *_ = np.linalg.lstsq(C, X, rcond=None)
    X_resid = X - C @ beta_c_X

    # degrees of freedom: n - (1 covariate/SNP slope) - q covariates - 1 intercept
    q = C.shape[1]
    dof = n - q - 1

    # marginal regression of residualized y on each residualized SNP column
    Sxx = np.sum(X_resid ** 2, axis=0)          # (p,)
    Sxy = X_resid.T @ y_resid                    # (p,)

    betas = Sxy / Sxx                            # (p,)

    fitted = X_resid * betas                     # (n, p), broadcasting
    resid = y_resid[:, None] - fitted            # (n, p)
    rss = np.sum(resid ** 2, axis=0)             # (p,)

    sigma2 = rss / dof                           # residual variance per SNP
    se = np.sqrt(sigma2 / Sxx)                   # (p,)

    z_scores = betas / se

    return z_scores, betas, se



def gwas_probit(X, y, covariates=None, verbose=False):
    """
    Performs a GWAS-style marginal probit regression: for each SNP (column
    of X) separately, fits y ~ snp (+ covariates) via maximum likelihood
    and returns the z-score for the SNP effect.

    Parameters
    ----------
    X : ndarray (n, p)
        Genotype matrix, one column per SNP.
    y : ndarray (n,)
        Binary outcome vector (0/1).
    covariates : ndarray (n, q) or None
        Optional covariates to adjust for (e.g. PCs, age, sex).
        An intercept is added automatically; don't include one yourself.
    verbose : bool
        If True, prints a warning whenever a SNP's model fails to converge.

    Returns
    -------
    z_scores : ndarray (p,)
        Z-score for each SNP's marginal effect. NaN if the model failed
        to converge for that SNP.
    betas : ndarray (p,)
        Estimated effect size (probit coefficient) for each SNP.
    se : ndarray (p,)
        Standard error for each SNP's effect.
    """
    n, p = X.shape
    y = np.asarray(y).reshape(-1)

    if covariates is not None:
        base = np.column_stack([np.ones(n), covariates])
    else:
        base = np.ones((n, 1))

    z_scores = np.full(p, np.nan)
    betas = np.full(p, np.nan)
    se = np.full(p, np.nan)

    for j in range(p):
        design = np.column_stack([base, X[:, j]])
        try:
            model = sm.Probit(y, design)
            res = model.fit(disp=0)
            # ostatnia kolumna = współczynnik SNP-a
            betas[j] = res.params[-1]
            se[j] = res.bse[-1]
            z_scores[j] = betas[j] / se[j]
        except Exception as e:
            if verbose:
                print(f"SNP {j} failed to converge: {e}")
            continue

    return z_scores, betas, se