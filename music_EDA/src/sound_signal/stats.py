"""Statistical methods implemented on numpy and scipy only.

scikit-learn and statsmodels are deliberately not required, so the project runs
wherever numpy, scipy, pandas and pyarrow are available.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

# ---------------------------------------------------------------- multiple testing


def bh_fdr(p_values) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    scaled = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(scaled[::-1])[::-1]
    result = np.empty(len(p))
    result[order] = np.clip(adjusted, 0, 1)
    return result


def holm(p_values) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    scaled = p[order] * (len(p) - np.arange(len(p)))
    adjusted = np.maximum.accumulate(scaled)
    result = np.empty(len(p))
    result[order] = np.clip(adjusted, 0, 1)
    return result


# ---------------------------------------------------------------- descriptive


def describe(values) -> dict:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    q1, median, q3 = np.percentile(x, [25, 50, 75])
    return {
        "n": int(len(x)), "mean": float(x.mean()), "median": float(median),
        "sd": float(x.std(ddof=1)), "q1": float(q1), "q3": float(q3),
        "min": float(x.min()), "max": float(x.max()),
        "skewness": float(st.skew(x)), "excess_kurtosis": float(st.kurtosis(x)),
    }


def bootstrap_ci(values, function=np.mean, n_boot=2000, seed=42, alpha=0.05):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if function is np.mean:
        draws = rng.multinomial(len(x), np.full(len(x), 1 / len(x)), n_boot) @ x / len(x)
    else:
        draws = np.array([function(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)])
    return tuple(float(v) for v in np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


# ---------------------------------------------------------------- correlation


def correlation_ci(r: float, n: int, spearman=False, alpha=0.05):
    """Fisher z interval; Spearman uses the Fieller et al. variance factor."""
    r = np.clip(r, -0.999999, 0.999999)
    se = np.sqrt((1.06 if spearman else 1.0) / (n - 3))
    z = np.arctanh(r)
    q = st.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - q * se)), float(np.tanh(z + q * se))


def correlation_table(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    rows = []
    ranks = frame[columns].rank()
    n = len(frame)
    pearson, spearman = frame[columns].corr().to_numpy(), ranks.corr().to_numpy()
    for i, a in enumerate(columns):
        for j in range(i + 1, len(columns)):
            r, rho = float(pearson[i, j]), float(spearman[i, j])
            t = r * np.sqrt((n - 2) / max(1e-12, 1 - r * r))
            rows.append({
                "a": a, "b": columns[j], "pearson": r, "spearman": rho, "n": n,
                "pearson_ci": correlation_ci(r, n), "spearman_ci": correlation_ci(rho, n, True),
                "p": float(2 * st.t.sf(abs(t), n - 2)),
            })
    for row, q in zip(rows, bh_fdr([r["p"] for r in rows])):
        row["q_fdr"] = float(q)
    return rows


# ---------------------------------------------------------------- effect sizes


def cliffs_delta(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ranks = st.rankdata(np.concatenate([a, b]))
    u = ranks[: len(a)].sum() - len(a) * (len(a) + 1) / 2
    return float(2 * u / (len(a) * len(b)) - 1)


def cohens_d(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    pooled = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    return float((a.mean() - b.mean()) / np.sqrt(pooled))


def delta_label(delta: float) -> str:
    size = abs(delta)
    return "negligible" if size < .147 else "small" if size < .33 else "medium" if size < .474 else "large"


def compare_two(a, b, seed=42, max_n=40_000) -> dict:
    """Mann-Whitney comparison with Cliff's delta and a bootstrap interval."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    u = st.mannwhitneyu(a, b, alternative="two-sided")
    small_a = a if len(a) <= max_n else rng.choice(a, max_n, replace=False)
    small_b = b if len(b) <= max_n else rng.choice(b, max_n, replace=False)
    draws = [cliffs_delta(rng.choice(small_a, min(len(small_a), 4000)), rng.choice(small_b, min(len(small_b), 4000)))
             for _ in range(200)]
    delta = cliffs_delta(a, b)
    brown = st.levene(small_a, small_b, center="median")
    return {
        "n_a": int(len(a)), "n_b": int(len(b)), "median_a": float(np.median(a)), "median_b": float(np.median(b)),
        "mean_a": float(a.mean()), "mean_b": float(b.mean()), "p": float(u.pvalue), "cliffs_delta": delta,
        "delta_ci": [float(delta + v) for v in np.percentile(np.array(draws) - np.mean(draws), [2.5, 97.5])],
        "magnitude": delta_label(delta), "cohens_d": cohens_d(a, b),
        "iqr_a": float(np.subtract(*np.percentile(a, [75, 25]))),
        "iqr_b": float(np.subtract(*np.percentile(b, [75, 25]))),
        "brown_forsythe_p": float(brown.pvalue),
    }


# ---------------------------------------------------------------- group tests


def assumption_checks(groups: list[np.ndarray], seed=42, max_n=2000) -> dict:
    """Normality and equal-variance checks on capped samples.

    At these sample sizes formal tests reject for trivial departures, so skewness,
    kurtosis and the variance ratio are reported alongside the p-values.
    """
    rng = np.random.default_rng(seed)
    capped = [g if len(g) <= max_n else rng.choice(g, max_n, replace=False) for g in groups]
    residuals = np.concatenate([g - g.mean() for g in capped])
    variances = [g.var(ddof=1) for g in capped]
    return {
        "residual_skewness": float(st.skew(residuals)),
        "residual_excess_kurtosis": float(st.kurtosis(residuals)),
        "normaltest_p": float(st.normaltest(residuals).pvalue),
        "brown_forsythe_p": float(st.levene(*capped, center="median").pvalue),
        "max_min_variance_ratio": float(max(variances) / max(min(variances), 1e-12)),
    }


def omnibus(groups: list[np.ndarray]) -> dict:
    """One-way ANOVA, Welch ANOVA and Kruskal-Wallis, with effect sizes."""
    groups = [np.asarray(g, dtype=float) for g in groups]
    n, k = sum(map(len, groups)), len(groups)
    f, p_f = st.f_oneway(*groups)
    h, p_h = st.kruskal(*groups)
    grand = np.concatenate(groups).mean()
    between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    weights = np.array([len(g) / g.var(ddof=1) for g in groups])
    means = np.array([g.mean() for g in groups])
    centre = (weights * means).sum() / weights.sum()
    lam = (((1 - weights / weights.sum()) ** 2) / (np.array([len(g) for g in groups]) - 1)).sum()
    welch_f = (weights * (means - centre) ** 2).sum() / (k - 1) / (1 + 2 * (k - 2) * lam / (k ** 2 - 1))
    welch_df = (k ** 2 - 1) / (3 * lam)
    return {
        "n": int(n), "groups": int(k), "anova_f": float(f), "anova_p": float(p_f),
        "welch_f": float(welch_f), "welch_df2": float(welch_df),
        "welch_p": float(st.f.sf(welch_f, k - 1, welch_df)),
        "eta_squared": float(between / (between + within)),
        "kruskal_h": float(h), "kruskal_p": float(p_h),
        "epsilon_squared": float(max(0.0, (h - k + 1) / (n - k))),
    }


def dunn_test(groups: dict[str, np.ndarray]) -> list[dict]:
    """Dunn's post-hoc test with tie correction and Holm adjustment."""
    names = list(groups)
    values = np.concatenate([groups[n] for n in names])
    ranks = st.rankdata(values)
    sizes = np.array([len(groups[n]) for n in names])
    means = np.array([part.mean() for part in np.split(ranks, np.cumsum(sizes)[:-1])])
    n = len(values)
    _, ties = np.unique(values, return_counts=True)
    variance = n * (n + 1) / 12 - (ties ** 3 - ties).sum() / (12 * (n - 1))
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            se = np.sqrt(variance * (1 / sizes[i] + 1 / sizes[j]))
            z = (means[i] - means[j]) / se
            rows.append({"a": names[i], "b": names[j], "z": float(z), "p": float(2 * st.norm.sf(abs(z)))})
    for row, value in zip(rows, holm([r["p"] for r in rows])):
        row["p_holm"] = float(value)
    return rows


def games_howell(groups: dict[str, np.ndarray]) -> list[dict]:
    """Games-Howell post-hoc comparison, which allows unequal variances."""
    names = list(groups)
    rows = []
    k = len(names)
    for i in range(k):
        for j in range(i + 1, k):
            a, b = groups[names[i]], groups[names[j]]
            va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
            q = abs(a.mean() - b.mean()) / np.sqrt((va + vb) / 2)
            df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
            rows.append({
                "a": names[i], "b": names[j], "difference": float(a.mean() - b.mean()),
                "p": float(st.studentized_range.sf(q, k, df)),
            })
    return rows


def chi_square(table) -> dict:
    """Chi-square test with bias-corrected Cramer's V."""
    observed = np.asarray(table, dtype=float)
    observed = observed[observed.sum(axis=1) > 0][:, observed.sum(axis=0) > 0]
    chi2, p, dof, expected = st.chi2_contingency(observed)
    n = observed.sum()
    r, c = observed.shape
    phi = max(0.0, chi2 / n - (r - 1) * (c - 1) / (n - 1))
    r_adj, c_adj = r - (r - 1) ** 2 / (n - 1), c - (c - 1) ** 2 / (n - 1)
    return {
        "chi2": float(chi2), "dof": int(dof), "p": float(p), "n": int(n),
        "cramers_v": float(np.sqrt(phi / max(1e-12, min(r_adj - 1, c_adj - 1)))),
        "min_expected": float(expected.min()),
        "pct_cells_expected_below_5": float((expected < 5).mean() * 100),
    }


def cohen_kappa(table) -> float:
    m = np.asarray(table, dtype=float)
    total = m.sum()
    observed = np.trace(m) / total
    expected = (m.sum(axis=0) * m.sum(axis=1)).sum() / total ** 2
    return float((observed - expected) / (1 - expected))


# ---------------------------------------------------------------- multivariate


def pca(matrix: np.ndarray, components: int | None = None) -> dict:
    centred = matrix - matrix.mean(axis=0)
    _, singular, vt = np.linalg.svd(centred, full_matrices=False)
    variance = singular ** 2 / (len(matrix) - 1)
    k = components or len(variance)
    return {
        "loadings": vt[:k], "scores": centred @ vt[:k].T,
        "explained": variance[:k] / variance.sum(), "eigenvalues": variance[:k],
    }


def _assign(matrix: np.ndarray, centres: np.ndarray, batch=100_000):
    labels = np.empty(len(matrix), dtype=np.int32)
    distance = np.empty(len(matrix), dtype=float)
    for start in range(0, len(matrix), batch):
        part = matrix[start:start + batch]
        d = ((part[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        labels[start:start + batch] = d.argmin(axis=1)
        distance[start:start + batch] = d.min(axis=1)
    return labels, distance


def kmeans(matrix: np.ndarray, k: int, seed=42, restarts=4, max_iter=60) -> dict:
    """Lloyd's algorithm with k-means++ seeding, keeping the best restart."""
    best = None
    for attempt in range(restarts):
        rng = np.random.default_rng(seed + attempt)
        centres = [matrix[rng.integers(len(matrix))]]
        distance = ((matrix - centres[0]) ** 2).sum(axis=1)
        for _ in range(1, k):
            centres.append(matrix[rng.choice(len(matrix), p=distance / distance.sum())])
            distance = np.minimum(distance, ((matrix - centres[-1]) ** 2).sum(axis=1))
        centres = np.array(centres)
        previous = None
        for _ in range(max_iter):
            labels, distance = _assign(matrix, centres)
            if previous is not None and (labels == previous).all():
                break
            previous = labels
            for c in range(k):
                members = labels == c
                centres[c] = matrix[members].mean(axis=0) if members.any() else matrix[distance.argmax()]
        inertia = float(distance.sum())
        if best is None or inertia < best["inertia"]:
            best = {"labels": labels, "centres": centres.copy(), "inertia": inertia}
    return best


def silhouette(matrix: np.ndarray, labels: np.ndarray, sample=4000, seed=42) -> float:
    rng = np.random.default_rng(seed)
    index = rng.choice(len(matrix), min(sample, len(matrix)), replace=False)
    x, y = matrix[index], labels[index]
    scores = []
    groups = np.unique(y)
    for i in range(len(x)):
        d = np.sqrt(((x - x[i]) ** 2).sum(axis=1))
        own = y == y[i]
        if own.sum() < 2:
            continue
        a = d[own].sum() / (own.sum() - 1)
        b = min(d[y == g].mean() for g in groups if g != y[i] and (y == g).any())
        scores.append((b - a) / max(a, b))
    return float(np.mean(scores))


def adjusted_rand(a, b) -> float:
    table = pd.crosstab(np.asarray(a), np.asarray(b)).to_numpy(dtype=float)
    pairs = lambda x: (x * (x - 1) / 2).sum()
    index, rows, columns = pairs(table), pairs(table.sum(axis=1)), pairs(table.sum(axis=0))
    expected = rows * columns / (table.sum() * (table.sum() - 1) / 2)
    return float((index - expected) / ((rows + columns) / 2 - expected))


def lda_fit(x: np.ndarray, y: np.ndarray, classes: int):
    """Linear discriminant classifier with a shared covariance and flat prior."""
    means = np.vstack([x[y == c].mean(axis=0) for c in range(classes)])
    within = x - means[y]
    covariance = within.T @ within / (len(x) - classes) + 1e-6 * np.eye(x.shape[1])
    return means, np.linalg.inv(covariance)


def lda_posterior(x: np.ndarray, means: np.ndarray, precision: np.ndarray, batch=5000) -> np.ndarray:
    output = []
    for start in range(0, len(x), batch):
        delta = x[start:start + batch][:, None, :] - means[None, :, :]
        score = -0.5 * np.einsum("nki,ij,nkj->nk", delta, precision, delta)
        score -= score.max(axis=1, keepdims=True)
        p = np.exp(score)
        output.append(p / p.sum(axis=1, keepdims=True))
    return np.vstack(output)


# ---------------------------------------------------------------- regression


def absorb(values: np.ndarray, effects: list[np.ndarray], tolerance=1e-7, max_iter=200) -> np.ndarray:
    """Remove fixed effects by alternating projections."""
    result = values.astype(float).copy()
    single = result.ndim == 1
    if single:
        result = result[:, None]
    for _ in range(max_iter if len(effects) > 1 else 1):
        change = 0.0
        for codes in effects:
            counts = np.maximum(np.bincount(codes), 1)
            for column in range(result.shape[1]):
                means = np.bincount(codes, result[:, column]) / counts
                result[:, column] -= means[codes]
                change = max(change, float(np.abs(means).max()))
        if change < tolerance:
            break
    return result[:, 0] if single else result


def ols_cluster(y, x, names, clusters, effects=()) -> dict:
    """OLS after absorbing fixed effects, with cluster-robust standard errors (CR1)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    effects = list(effects)
    if effects:
        y_w, x_w = absorb(y, effects), absorb(x, effects)
        absorbed = sum(int(e.max()) for e in effects) + 1
    else:
        x_w = np.column_stack([np.ones(len(y)), x - x.mean(axis=0)])
        y_w, names, absorbed = y, ["intercept"] + list(names), 0
    xtx_inverse = np.linalg.pinv(x_w.T @ x_w)
    beta = xtx_inverse @ (x_w.T @ y_w)
    residual = y_w - x_w @ beta
    codes = pd.factorize(clusters)[0]
    g = int(codes.max()) + 1
    scores = np.column_stack([np.bincount(codes, x_w[:, j] * residual, g) for j in range(x_w.shape[1])])
    n, k = len(y), x_w.shape[1] + absorbed
    factor = g / (g - 1) * (n - 1) / max(1, n - k)
    covariance = factor * xtx_inverse @ (scores.T @ scores) @ xtx_inverse
    se = np.sqrt(np.diag(covariance))
    t = beta / se
    p = 2 * st.t.sf(np.abs(t), g - 1)
    critical = st.t.ppf(0.975, g - 1)
    total = ((y - y.mean()) ** 2).sum()
    rss = float((residual ** 2).sum())
    condition = np.corrcoef(x_w, rowvar=False) if effects else np.corrcoef(x_w[:, 1:], rowvar=False)
    vif = np.diag(np.linalg.pinv(np.atleast_2d(condition)))
    vif_names = names if effects else names[1:]
    fitted = y_w - residual
    order = np.random.default_rng(0).choice(n, min(n, 5000), replace=False)
    return {
        "n": int(n), "clusters": g,
        "r2_total": float(1 - rss / total),
        "r2_within": float(1 - rss / ((y_w - y_w.mean()) ** 2).sum()),
        "coefficients": [
            {"term": name, "beta": float(b), "se": float(s), "ci": [float(b - critical * s), float(b + critical * s)],
             "p": float(v)} for name, b, s, v in zip(names, beta, se, p)
        ],
        "vif": {name: float(v) for name, v in zip(vif_names, vif)},
        "residual_skewness": float(st.skew(residual)),
        "residual_excess_kurtosis": float(st.kurtosis(residual)),
        "heteroscedasticity_corr": float(np.corrcoef(np.abs(residual[order]), fitted[order])[0, 1]),
    }


def _fit_additive(y, x, effects, shrink, sweeps=8):
    if x is not None and effects:
        beta = np.linalg.lstsq(absorb(x, effects), absorb(y, effects), rcond=None)[0]
    elif x is not None:
        beta = np.linalg.lstsq(x - x.mean(axis=0), y - y.mean(), rcond=None)[0]
    else:
        beta = None
    base = y - (x @ beta if beta is not None else 0)
    mean = base.mean()
    residual = base - mean
    tables = [np.zeros(int(e.max()) + 1) for e in effects]
    for _ in range(sweeps if len(effects) > 1 else 1):
        for index, codes in enumerate(effects):
            residual += tables[index][codes]
            counts = np.bincount(codes, minlength=len(tables[index]))
            tables[index] = np.bincount(codes, residual, len(tables[index])) / np.maximum(counts + shrink[index], 1e-9)
            residual -= tables[index][codes]
    return beta, mean, tables


def cv_r2(y, x=None, effects=(), shrink=None, groups=None, folds=5, seed=42) -> dict:
    """Out-of-fold R2 for additive models. With groups, whole groups are held out."""
    y = np.asarray(y, dtype=float)
    effects = list(effects)
    shrink = list(shrink) if shrink is not None else [0.0] * len(effects)
    rng = np.random.default_rng(seed)
    if groups is None:
        assignment = rng.integers(0, folds, len(y))
    else:
        codes = pd.factorize(groups)[0]
        assignment = rng.integers(0, folds, int(codes.max()) + 1)[codes]
    scores = []
    for fold in range(folds):
        test = assignment == fold
        train = ~test
        train_effects = [e[train] for e in effects]
        beta, mean, tables = _fit_additive(y[train], None if x is None else x[train], train_effects, shrink)
        prediction = np.full(int(test.sum()), mean)
        if beta is not None:
            prediction = prediction + x[test] @ beta
        for table, codes in zip(tables, effects):
            known = codes[test] < len(table)
            prediction[known] += table[codes[test][known]]
        scores.append(1 - ((y[test] - prediction) ** 2).sum() / ((y[test] - y[train].mean()) ** 2).sum())
    return {"r2": float(np.mean(scores)), "sd": float(np.std(scores, ddof=1)), "folds": folds}


# ---------------------------------------------------------------- time series


def theil_sen(years, values, block=5, n_boot=1000, seed=42) -> dict:
    """Theil-Sen slope, moving-block bootstrap interval and Mann-Kendall test."""
    t, y = np.asarray(years, dtype=float), np.asarray(values, dtype=float)
    slope = float(st.theilslopes(y, t).slope)
    tau, p = st.kendalltau(t, y)
    rng = np.random.default_rng(seed)
    n = len(y)
    intercept = np.median(y - slope * t)
    fitted = intercept + slope * t
    residual = y - fitted
    draws = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, n // block + 1)
        sample = np.concatenate([residual[s:s + block] for s in starts])[:n]
        draws.append(st.theilslopes(fitted + sample, t).slope)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"slope_per_year": slope, "ci": [float(low), float(high)], "kendall_tau": float(tau), "p": float(p),
            "lag1_autocorr_residual": float(np.corrcoef(residual[:-1], residual[1:])[0, 1])}


def change_points(years, values, min_segment=5, max_breaks=4) -> list[dict]:
    """Binary segmentation on the mean, stopping when BIC no longer improves."""
    y = np.asarray(values, dtype=float)
    t = np.asarray(years)
    n = len(y)
    breaks: list[int] = []
    cost = lambda a, b: float(((y[a:b] - y[a:b].mean()) ** 2).sum())

    def total(points):
        edges = [0] + sorted(points) + [n]
        return sum(cost(a, b) for a, b in zip(edges[:-1], edges[1:]))

    bic = lambda points: n * np.log(max(total(points), 1e-12) / n) + (2 * len(points) + 1) * np.log(n)
    current = bic(breaks)
    for _ in range(max_breaks):
        options = []
        edges = [0] + sorted(breaks) + [n]
        for a, b in zip(edges[:-1], edges[1:]):
            options += [c for c in range(a + min_segment, b - min_segment + 1)]
        if not options:
            break
        candidate = min(options, key=lambda c: total(breaks + [c]))
        score = bic(breaks + [candidate])
        if score >= current:
            break
        breaks.append(candidate)
        current = score
    breaks.sort()
    edges = [0] + breaks + [n]
    return [{"year": int(t[c]), "before": float(y[edges[i]:c].mean()), "after": float(y[c:edges[i + 2]].mean())}
            for i, c in enumerate(breaks)]


def robust_anomalies(years, values, window=7, threshold=3.5) -> list[dict]:
    """Years far from the median of their neighbours, scaled by the MAD.

    The year being tested is left out of its own baseline.
    """
    series = pd.Series(np.asarray(values, dtype=float), index=np.asarray(years))
    half = window // 2
    numbers = series.to_numpy()
    baseline = pd.Series([
        np.median(np.delete(numbers[max(0, i - half):i + half + 1], min(i, half)))
        for i in range(len(numbers))], index=series.index)
    residual = series - baseline
    mad = 1.4826 * np.median(np.abs(residual - residual.median()))
    z = residual / mad if mad > 0 else residual * 0
    return [{"year": int(y), "value": float(series[y]), "expected": float(baseline[y]), "robust_z": float(v)}
            for y, v in z.items() if abs(v) >= threshold]


def kaplan_meier(durations, observed) -> dict:
    d, e = np.asarray(durations, dtype=float), np.asarray(observed, dtype=bool)
    times = np.unique(d[e])
    at_risk = np.array([(d >= t).sum() for t in times])
    events = np.array([((d == t) & e).sum() for t in times])
    survival = np.cumprod(1 - events / at_risk)
    below = np.nonzero(survival <= 0.5)[0]
    return {"t": times.tolist(), "s": survival.tolist(), "median": float(times[below[0]]) if len(below) else None}


def logrank(groups: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    d = np.concatenate([g[0] for g in groups]).astype(float)
    e = np.concatenate([g[1] for g in groups]).astype(bool)
    label = np.concatenate([np.full(len(g[0]), i) for i, g in enumerate(groups)])
    k = len(groups)
    observed, expected = np.zeros(k - 1), np.zeros(k - 1)
    variance = np.zeros((k - 1, k - 1))
    for t in np.unique(d[e]):
        risk = d >= t
        n = risk.sum()
        died = ((d == t) & e).sum()
        if n < 2:
            continue
        at_risk = np.array([(risk & (label == i)).sum() for i in range(k - 1)])
        events = np.array([((d == t) & e & (label == i)).sum() for i in range(k - 1)])
        share = at_risk / n
        observed += events
        expected += died * share
        factor = died * (n - died) / (n - 1)
        variance += factor * (np.diag(share) - np.outer(share, share))
    difference = observed - expected
    statistic = float(difference @ np.linalg.pinv(variance) @ difference)
    return {"chi2": statistic, "dof": k - 1, "p": float(st.chi2.sf(statistic, k - 1))}
