import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
from typing import NamedTuple


class MCARResult(NamedTuple):
    """Result of Little's MCAR test."""
    statistic: float       # Chi-square test statistic
    df: int                # Degrees of freedom
    p_value: float         # p-value
    missing_patterns: int  # Number of distinct missing-data patterns
    verdict: str           # Human-readable verdict


def little_mcar_test(df: pd.DataFrame, alpha: float = 0.05) -> MCARResult:
    """
    Tests the null hypothesis that data is Missing Completely At Random (MCAR).

    Parameters:
        df (pd.DataFrame): DataFrame with potentially missing values.
        alpha (float): Significance level for the verdict.

    Returns:
    MCARResult: Named tuple with: statistic, df, p_value, missing_patterns, verdict.
    """

    if df.shape[1] < 2:
        raise ValueError("MCAR test requires at least 2 numeric columns.")

    n_rows, n_cols = df.shape
    col_names = df.columns.tolist()
    X = df.values  # (n, p) numpy array; NaN where missing

    # 1. Overall column means (using all available observations per column)
    grand_means = np.nanmean(X, axis=0)  # shape (p,)

    # 2. Overall covariance (listwise complete cases)
    complete_mask = ~np.any(np.isnan(X), axis=1)
    n_complete = complete_mask.sum()
    if n_complete < n_cols + 1:
        raise ValueError(
            f"Only {n_complete} complete rows found; need at least {n_cols + 1} "
            "to estimate the covariance matrix."
        )
    X_complete = X[complete_mask]
    cov_complete = np.cov(X_complete, rowvar=False)  # (p, p)

    # 3. Identify unique missingness patterns
    # Encode each row as a boolean tuple: True = observed, False = missing
    obs_mask = ~np.isnan(X)  # (n, p) bool
    pattern_keys = [tuple(row) for row in obs_mask]

    pattern_to_rows: dict[tuple, list[int]] = {}
    for i, key in enumerate(pattern_keys):
        pattern_to_rows.setdefault(key, []).append(i)

    # Skip the all-missing pattern (contributes nothing) and all-observed (fully complete rows)
    patterns = {k: v for k, v in pattern_to_rows.items() if any(k)}

    n_patterns = len(patterns)

    # 4. Compute chi-square statistic
    chi2_stat = 0.0
    total_df = 0

    for obs_pattern, row_indices in patterns.items():
        obs_idx = np.where(obs_pattern)[0]  # indices of observed columns
        n_k = len(row_indices)
        p_k = len(obs_idx)

        if p_k == 0 or n_k == 0:
            continue

        # Group mean for observed columns in this pattern
        X_k = X[np.ix_(row_indices, obs_idx)]  # (n_k, p_k)
        mu_k = np.nanmean(X_k, axis=0)          # (p_k,)

        # Grand means for those columns
        mu_grand_k = grand_means[obs_idx]        # (p_k,)

        diff = mu_k - mu_grand_k                 # (p_k,)

        # Sub-covariance for observed columns
        cov_k = cov_complete[np.ix_(obs_idx, obs_idx)]  # (p_k, p_k)

        # Regularise to avoid singular matrices
        cov_k = cov_k + np.eye(p_k) * 1e-10

        try:
            cov_k_inv = np.linalg.inv(cov_k)
        except np.linalg.LinAlgError:
            cov_k_inv = np.linalg.pinv(cov_k)

        # Contribution: n_k * diff^T * Sigma_k^{-1} * diff
        contribution = n_k * float(diff @ cov_k_inv @ diff)
        chi2_stat += contribution
        total_df += p_k

    # Subtract p (number of variables) from total_df per Little (1988)
    dof = total_df - n_cols
    if dof <= 0:
        dof = 1  # safeguard

    p_value = 1.0 - stats.chi2.cdf(chi2_stat, dof)

    verdict = (
        f"Chấp nhận H0 (p={p_value:.4f} ≥ {alpha}): "
        "dữ liệu có thể là MCAR."
        if p_value >= alpha
        else f"Bác bỏ H0 (p={p_value:.4f} < {alpha}): "
        "dữ liệu có thể không phải MCAR."
    )

    return MCARResult(
        statistic=round(chi2_stat, 4),
        df=dof,
        p_value=round(p_value, 4),
        missing_patterns=n_patterns,
        verdict=verdict,
    )

def mcar_summary(df: pd.DataFrame, alpha: float = 0.05) -> None:
    """
    Print a formatted summary of Little's MCAR test results.

    Parameters:
        df (pd.DataFrame)
        alpha (float)
    """
    result = little_mcar_test(df, alpha=alpha)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    total_cells = df[numeric_cols].size
    missing_cells = df[numeric_cols].isna().sum().sum()
    pct_missing = 100 * missing_cells / total_cells

    print("=" * 60)
    print("  Little's MCAR Test")
    print("=" * 60)
    print(f"  Numeric columns   : {list(numeric_cols)}")
    print(f"  Rows              : {len(df)}")
    print(f"  Missing cells     : {missing_cells} / {total_cells} ({pct_missing:.1f}%)")
    print(f"  Missing patterns  : {result.missing_patterns}")
    print("-" * 60)
    print(f"  Chi-square stat   : {result.statistic:.4f}")
    print(f"  Degrees of freedom: {result.df}")
    print(f"  p-value           : {result.p_value:.4f}")
    print("-" * 60)
    print(f"  Verdict: {result.verdict}")
    print("=" * 60)


# ── Demo / usage examples ─────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("\n── Example 1: MCAR data (random missingness) ──\n")
    df_mcar = pd.DataFrame(
        rng.standard_normal((300, 4)), columns=["age", "income", "score", "weight"]
    )
    # Introduce missingness at random (~15 % per column)
    for col in df_mcar.columns:
        mask = rng.random(len(df_mcar)) < 0.15
        df_mcar.loc[mask, col] = np.nan

    mcar_summary(df_mcar)

    print("\n── Example 2: MAR data (missingness depends on another column) ──\n")
    df_mar = pd.DataFrame(
        rng.standard_normal((300, 4)), columns=["age", "income", "score", "weight"]
    )
    # 'income' is missing for people with high 'age' → MAR pattern
    high_age_mask = df_mar["age"] > 0.5
    df_mar.loc[high_age_mask, "income"] = np.nan
    # 'score' random missing
    df_mar.loc[rng.random(300) < 0.10, "score"] = np.nan

    mcar_summary(df_mar)

    print("\n── Example 3: Access raw result ──\n")
    result = little_mcar_test(df_mar)
    print(result)
    print(f"\np-value = {result.p_value}  |  stat = {result.statistic}")