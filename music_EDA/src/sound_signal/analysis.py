"""Investigations A to E, plus descriptive statistics.

Each public function reads cleaned data, computes its statistics and writes one
JSON file to outputs/results. Figures and the publication are built from those
JSON files, so every number shown can be traced to one place.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from scipy import stats as st
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform

from . import stats as ss
from .common import transform_features
from .config import AUDIO_FEATURES, PROCESSED_DIR, RANDOM_SEED, RESULTS_DIR

NON_MUSIC_GENRES = {"comedy", "sleep"}
EMOTION_FEATURES = ["energy", "valence", "danceability", "tempo", "acousticness"]
DISPLAY_FEATURES = AUDIO_FEATURES + ["duration_min"]
FEATURE_BINS = {
    "danceability": (0, 1), "energy": (0, 1), "speechiness": (0, 1), "acousticness": (0, 1),
    "instrumentalness": (0, 1), "liveness": (0, 1), "valence": (0, 1),
    "loudness": (-40, 5), "tempo": (40, 220), "duration_min": (0, 10),
}
MOOD_LEXICON = {
    "happy": {"happy", "upbeat", "joyful", "cheerful", "feel good", "feelgood", "uplifting", "sunny", "joy", "fun"},
    "angry": {"angry", "aggressive", "rage", "furious", "anger", "hate", "pissed off"},
    "sad": {"sad", "melancholy", "melancholic", "depressing", "depressed", "heartbreak", "sorrow", "gloomy",
            "grief", "lonely", "sadness", "tearjerker", "heartbroken"},
    "calm": {"calm", "relaxing", "relax", "relaxed", "chill", "mellow", "peaceful", "soothing", "serene", "tranquil"},
}
MOOD_QUADRANT = {"happy": (1, 1), "angry": (0, 1), "sad": (0, 0), "calm": (1, 0)}
FOCAL_GENRES = ["pop", "rock", "hip-hop", "country", "edm", "latin", "r-n-b", "classical", "jazz",
                "heavy-metal", "folk", "k-pop"]


def read(name: str, columns=None) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / f"{name}.parquet", columns=columns)


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.ndarray, pd.Series)):
        return _plain(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return 0.0 if number == 0 else float(f"{number:.5g}")
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return value


def save(name: str, payload: dict) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = _plain(payload)
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(payload, separators=(",", ":")))
    print(f"saved {name}", flush=True)
    return payload


def histogram(values, feature: str, bins=30) -> list[float]:
    low, high = FEATURE_BINS[feature]
    counts, _ = np.histogram(np.clip(values, low, high), bins=bins, range=(low, high))
    return (counts / max(1, counts.sum())).round(5).tolist()


def wilson(successes, n, z=1.96):
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def genre_universe() -> pd.DataFrame:
    columns = ["track_id", "name", "artist", "genre", "popularity", "explicit", "duration_ms", "mode"] + AUDIO_FEATURES
    frame = read("genre_tracks", columns)
    frame = frame.loc[~frame["genre"].isin(NON_MUSIC_GENRES) & (frame["speechiness"] <= 0.66)].copy()
    frame["duration_min"] = frame["duration_ms"] / 60000
    return frame


def popularity_universe() -> pd.DataFrame:
    columns = ["track_id", "artist", "year", "genre", "popularity", "duration_ms"] + AUDIO_FEATURES
    frame = read("popularity_1m", columns)
    frame = frame.loc[~frame["genre"].isin(NON_MUSIC_GENRES) & (frame["speechiness"] <= 0.66)].copy()
    frame["duration_min"] = frame["duration_ms"] / 60000
    frame["pop_pct"] = frame.groupby("year")["popularity"].rank(pct=True).astype("float32")
    frame["hit"] = (frame["pop_pct"] >= 0.9) & (frame["popularity"] > 0)
    frame["low"] = frame["pop_pct"] <= 0.5
    # Genre labels here come from how the sample was collected, so every headline
    # comparison is repeated inside genre-year cells.
    frame["genre_pct"] = frame.groupby(["genre", "year"])["popularity"].rank(pct=True).astype("float32")
    frame["genre_hit"] = (frame["genre_pct"] >= 0.9) & (frame["popularity"] > 0)
    frame["genre_low"] = frame["genre_pct"] <= 0.5
    return frame.reset_index(drop=True)


# ---------------------------------------------------------------- descriptive


def overview() -> dict:
    genre = genre_universe().drop_duplicates("track_id")
    pop = popularity_universe()
    catalogue = read("catalogue", AUDIO_FEATURES + ["duration_ms"])
    catalogue["duration_min"] = catalogue["duration_ms"] / 60000
    table, outliers, distributions = {}, {}, {}
    for feature in DISPLAY_FEATURES:
        values = catalogue[feature].to_numpy(dtype=float)
        table[feature] = ss.describe(values)
        q1, q3 = table[feature]["q1"], table[feature]["q3"]
        spread = q3 - q1
        outliers[feature] = float(((values < q1 - 1.5 * spread) | (values > q3 + 1.5 * spread)).mean() * 100)
        distributions[feature] = {
            "range": FEATURE_BINS[feature],
            "catalogue": histogram(values, feature), "genre_sample": histogram(genre[feature], feature),
            "popularity_sample": histogram(pop[feature], feature),
        }
    counts, _ = np.histogram(pop["popularity"], bins=np.arange(0, 102, 2))
    genre_counts, _ = np.histogram(genre["popularity"], bins=np.arange(0, 102, 2))
    return save("overview", {
        "descriptive_catalogue": table, "outlier_pct_tukey_kept": outliers, "distributions": distributions,
        "popularity": {
            "bin_edges": np.arange(0, 102, 2),
            "sample_1m": (counts / counts.sum()), "sample_genre": (genre_counts / genre_counts.sum()),
            "describe_1m": ss.describe(pop["popularity"]), "describe_genre": ss.describe(genre["popularity"]),
            "zero_share_1m": float((pop["popularity"] == 0).mean()),
            "zero_share_genre": float((genre["popularity"] == 0).mean()),
            "median_by_year": pop.groupby("year")["popularity"].median().to_dict(),
            "tracks_by_year": pop.groupby("year").size().to_dict(),
        },
        "universes": {"catalogue": len(catalogue), "genre_unique": len(genre), "popularity": len(pop)},
    })


# ---------------------------------------------------------------- A. emotion


def _mood_of(tag_text: str):
    tags = {t.strip() for t in tag_text.lower().split(",")}
    found = [mood for mood, words in MOOD_LEXICON.items() if tags & words]
    return found[0] if len(found) == 1 else None


def emotion() -> dict:
    frame = genre_universe().drop_duplicates("track_id")
    columns = AUDIO_FEATURES + ["duration_min"]
    pairs = ss.correlation_table(frame[EMOTION_FEATURES], EMOTION_FEATURES)
    pearson = frame[columns].corr()
    spearman = frame[columns].rank().corr()
    order_link = hierarchy.linkage(squareform(1 - spearman.abs().to_numpy(), checks=False), "average")
    order = hierarchy.leaves_list(order_link)
    feature_groups = hierarchy.fcluster(order_link, 0.6, "distance")
    grid, x_edges, y_edges = np.histogram2d(frame["valence"], frame["energy"], bins=40, range=[[0, 1], [0, 1]])
    high_v, high_e = frame["valence"] >= 0.5, frame["energy"] >= 0.5
    quadrants = {
        "elated (high valence, high energy)": high_v & high_e, "tense (low valence, high energy)": ~high_v & high_e,
        "sombre (low valence, low energy)": ~high_v & ~high_e, "serene (high valence, low energy)": high_v & ~high_e,
    }
    z = transform_features(frame, AUDIO_FEATURES)
    components = ss.pca(z)
    genre_points = frame.groupby("genre")[["valence", "energy"]].median()
    return save("emotion", {
        "n": len(frame), "pairs": pairs, "matrix_features": columns,
        "pearson": pearson.to_numpy(), "spearman": spearman.to_numpy(),
        "clustered_order": [columns[i] for i in order],
        "feature_clusters": {c: int(g) for c, g in zip(columns, feature_groups)},
        "map": {"density": (grid.T / grid.sum()), "edges": x_edges},
        "quadrant_share": {k: float(v.mean()) for k, v in quadrants.items()},
        "genre_points": {g: [float(r.valence), float(r.energy)] for g, r in genre_points.iterrows()},
        "pca": {"features": AUDIO_FEATURES, "explained": components["explained"], "loadings": components["loadings"],
                "eigenvalues": components["eigenvalues"]},
        "tags": mood_tags(),
    })


def mood_tags() -> dict:
    tags = read("tags", ["track_id", "artist", "valence", "energy", "tags"])
    medians = {"valence": float(tags["valence"].median()), "energy": float(tags["energy"].median())}
    pattern = "|".join(sorted({w for words in MOOD_LEXICON.values() for w in words}))
    frame = tags.loc[tags["tags"].str.lower().str.contains(pattern, regex=True)].copy()
    del tags
    frame["mood"] = [_mood_of(text) for text in frame["tags"]]
    conflicting = int(frame["mood"].isna().sum())
    per_tag = []
    words = sorted({w for group in MOOD_LEXICON.values() for w in group})
    token_sets = [{t.strip() for t in text.lower().split(",")} for text in frame["tags"]]
    for word in words:
        mask = np.fromiter((word in tokens for tokens in token_sets), bool, len(token_sets))
        if mask.sum() < 200:
            continue
        v, e = ss.compare_two(frame.loc[mask, "valence"], frame.loc[~mask, "valence"]), \
            ss.compare_two(frame.loc[mask, "energy"], frame.loc[~mask, "energy"])
        per_tag.append({"tag": word, "n": int(mask.sum()), "valence_median": v["median_a"], "energy_median": e["median_a"],
                        "valence_delta": v["cliffs_delta"], "energy_delta": e["cliffs_delta"], "p_v": v["p"], "p_e": e["p"]})
    del token_sets
    adjusted = ss.bh_fdr([r["p_v"] for r in per_tag] + [r["p_e"] for r in per_tag])
    for i, row in enumerate(per_tag):
        row["q_v"], row["q_e"] = float(adjusted[i]), float(adjusted[i + len(per_tag)])
    frame = frame.dropna(subset=["mood"]).copy()
    moods = list(MOOD_LEXICON)
    summary, tests = {}, {}
    for mood in moods:
        part = frame.loc[frame["mood"] == mood]
        summary[mood] = {
            "n": len(part), "artists": int(part["artist"].nunique()),
            **{f"{f}_{name}": float(np.percentile(part[f], q)) for f in ("valence", "energy")
               for name, q in (("q1", 25), ("median", 50), ("q3", 75))},
        }
    for feature in ("valence", "energy"):
        groups = {m: frame.loc[frame["mood"] == m, feature].to_numpy(dtype=float) for m in moods}
        tests[feature] = {**ss.omnibus(list(groups.values())), "assumptions": ss.assumption_checks(list(groups.values())),
                          "dunn": ss.dunn_test(groups)}
    results = {}
    for label, (cut_v, cut_e) in {"catalogue_medians": (medians["valence"], medians["energy"]), "fixed_0.5": (.5, .5)}.items():
        v, e = (frame["valence"] >= cut_v).astype(int), (frame["energy"] >= cut_e).astype(int)
        lookup = {key: mood for mood, key in MOOD_QUADRANT.items()}
        measured = pd.Series([lookup[k] for k in zip(v, e)], index=frame.index)
        table = pd.crosstab(frame["mood"], measured).reindex(index=moods, columns=moods, fill_value=0)
        codes = pd.factorize(frame["artist"])[0]
        tag_i, meas_i = frame["mood"].map(moods.index).to_numpy(), measured.map(moods.index).to_numpy()
        artist_tables = np.zeros((codes.max() + 1, 16))
        np.add.at(artist_tables, (codes, tag_i * 4 + meas_i), 1)
        rng = np.random.default_rng(RANDOM_SEED)
        draws, accuracy = [], []
        for _ in range(500):
            weights = rng.multinomial(len(artist_tables), np.full(len(artist_tables), 1 / len(artist_tables)))
            m = (artist_tables * weights[:, None]).sum(axis=0).reshape(4, 4)
            draws.append(ss.cohen_kappa(m))
            accuracy.append(np.trace(m) / m.sum())
        results[label] = {
            "cuts": [cut_v, cut_e], "table": table.to_numpy(), "chi_square": ss.chi_square(table.to_numpy()),
            "kappa": ss.cohen_kappa(table.to_numpy()), "kappa_ci_artist_bootstrap": np.percentile(draws, [2.5, 97.5]),
            "accuracy": np.trace(table.to_numpy()) / table.to_numpy().sum(),
            "accuracy_ci_artist_bootstrap": np.percentile(accuracy, [2.5, 97.5]),
            "majority_baseline": table.sum(axis=1).max() / table.to_numpy().sum(),
        }
    return {"moods": moods, "lexicon": {k: sorted(v) for k, v in MOOD_LEXICON.items()}, "tracks_single_mood": len(frame),
            "tracks_conflicting_moods_excluded": conflicting, "summary": summary, "tests": tests,
            "agreement": results, "per_tag": per_tag}


# ---------------------------------------------------------------- B. popularity


def _within_rank(frame: pd.DataFrame, feature: str, by: str) -> np.ndarray:
    return frame.groupby(by)[feature].rank(pct=True).to_numpy(dtype=float)


def popularity() -> dict:
    frame = popularity_universe()
    year_codes = pd.factorize(frame["year"])[0]
    genre_codes = pd.factorize(frame["genre"])[0]
    artist_codes = pd.factorize(frame["artist"])[0]
    features = DISPLAY_FEATURES
    hits, lows = frame.loc[frame["hit"]], frame.loc[frame["low"]]
    comparisons, correlations = {}, []
    outcome = frame["pop_pct"].to_numpy(dtype=float)
    outcome_g = ss.absorb(outcome, [genre_codes], max_iter=1)
    nonzero = frame["popularity"].to_numpy() > 0
    for feature in features:
        comparisons[feature] = {
            **ss.compare_two(hits[feature], lows[feature]),
            "hit_hist": histogram(hits[feature], feature), "low_hist": histogram(lows[feature], feature),
            "range": FEATURE_BINS[feature],
            "hit_p05_p95": np.percentile(hits[feature], [5, 95]), "low_p05_p95": np.percentile(lows[feature], [5, 95]),
        }
        ranked = _within_rank(frame, feature, "year")
        rho = float(np.corrcoef(ranked, outcome)[0, 1])
        partial = float(np.corrcoef(ss.absorb(ranked, [genre_codes], max_iter=1), outcome_g)[0, 1])
        no_zero = float(np.corrcoef(ranked[nonzero], outcome[nonzero])[0, 1])
        n = len(frame)
        correlations.append({
            "feature": feature, "rho_within_year": rho, "ci": ss.correlation_ci(rho, n, True),
            "rho_within_year_genre": partial, "ci_genre": ss.correlation_ci(partial, n, True),
            "rho_excluding_zero_popularity": no_zero,
            "p": float(2 * st.norm.sf(abs(np.arctanh(rho)) * np.sqrt((n - 3) / 1.06))),
        })
    for row, q in zip(correlations, ss.bh_fdr([r["p"] for r in correlations])):
        row["q_fdr"] = float(q)
    for feature, q in zip(features, ss.bh_fdr([comparisons[f]["p"] for f in features])):
        comparisons[feature]["q_fdr"] = float(q)

    lift = {}
    base = float(frame["hit"].mean())
    decile_codes = {}
    for feature in features:
        codes = pd.qcut(frame[feature].rank(method="first"), 10, labels=False).to_numpy()
        decile_codes[feature] = codes
        n = np.bincount(codes, minlength=10)
        k = np.bincount(codes, frame["hit"].to_numpy(dtype=float), 10)
        low, high = wilson(k, n)
        lift[feature] = {"rate": k / n, "low": low, "high": high,
                         "median": frame.groupby(codes)[feature].median().to_numpy()}

    # Are hits confined to a narrow part of the feature space?
    bounds = {f: np.percentile(hits[f], [5, 95]) for f in AUDIO_FEATURES}
    inside = np.ones(len(frame), dtype=bool)
    single = {}
    for feature, (low, high) in bounds.items():
        within = frame[feature].between(low, high).to_numpy()
        single[feature] = {"hits": float(within[frame["hit"]].mean()), "others": float(within[~frame["hit"]].mean()),
                           "band": [low, high]}
        inside &= within
    z = transform_features(frame, AUDIO_FEATURES)
    z_genre = ss.absorb(z, [genre_codes], max_iter=1)
    rng = np.random.default_rng(RANDOM_SEED)

    def spread(matrix, mask):
        members = np.nonzero(mask)[0]
        out = []
        for _ in range(200):
            pick = rng.choice(members, 5000, replace=False)
            part = matrix[pick]
            out.append(np.sqrt(((part - part.mean(axis=0)) ** 2).sum(axis=1)).mean())
        return np.array(out)

    hit_mask, low_mask = frame["hit"].to_numpy(), frame["low"].to_numpy()
    raw_hit, raw_low = spread(z, hit_mask), spread(z, low_mask)
    within_hit, within_low = spread(z_genre, hit_mask), spread(z_genre, low_mask)
    dispersion = {
        "raw": {"hits": raw_hit.mean(), "low": raw_low.mean(), "ratio": raw_hit.mean() / raw_low.mean(),
                "ratio_ci": np.percentile(raw_hit / rng.permutation(raw_low), [2.5, 97.5])},
        "within_genre": {"hits": within_hit.mean(), "low": within_low.mean(), "ratio": within_hit.mean() / within_low.mean(),
                         "ratio_ci": np.percentile(within_hit / rng.permutation(within_low), [2.5, 97.5])},
    }

    # How much of popularity can each kind of information predict out of sample?
    y = outcome
    x = (frame[features].to_numpy(dtype=float) - frame[features].mean().to_numpy()) / frame[features].std().to_numpy()
    deciles = [decile_codes[f] for f in features]
    ladder = {
        "audio_linear": ss.cv_r2(y, x),
        "audio_flexible_deciles": ss.cv_r2(y, effects=deciles),
        "genre_only": ss.cv_r2(y, effects=[genre_codes]),
        "audio_flexible_plus_genre": ss.cv_r2(y, effects=deciles + [genre_codes]),
        "audio_flexible_plus_genre_unseen_artists": ss.cv_r2(y, effects=deciles + [genre_codes], groups=artist_codes),
        "artist_only": ss.cv_r2(y, effects=[artist_codes], shrink=[2.0]),
        "all_with_artist": ss.cv_r2(y, effects=deciles + [genre_codes, artist_codes],
                                    shrink=[0.0] * len(deciles) + [0.0, 2.0]),
    }
    regression = ss.ols_cluster(y * 100, x, features, artist_codes, [genre_codes, year_codes])
    for row, q in zip(regression["coefficients"], ss.bh_fdr([c["p"] for c in regression["coefficients"]])):
        row["q_fdr"] = float(q)

    within_genre = []
    inside_hits, inside_low = frame.loc[frame["genre_hit"]], frame.loc[frame["genre_low"]]
    for feature in features:
        within_genre.append({"feature": feature, **ss.compare_two(inside_hits[feature], inside_low[feature])})
    for row, q in zip(within_genre, ss.bh_fdr([r["p"] for r in within_genre])):
        row["q_fdr"] = float(q)

    replicate = genre_universe().drop_duplicates("track_id").reset_index(drop=True)
    replication = {f: float(st.spearmanr(replicate[f], replicate["popularity"]).statistic) for f in features}
    by_genre_share = (frame.groupby("genre")["hit"].mean().sort_values(ascending=False))
    y2 = replicate["popularity"].rank(pct=True).to_numpy(dtype=float)
    x2 = ((replicate[features] - replicate[features].mean()) / replicate[features].std()).to_numpy(dtype=float)
    g2, a2 = pd.factorize(replicate["genre"])[0], pd.factorize(replicate["artist"])[0]
    second_ladder = {
        "audio_linear": ss.cv_r2(y2, x2), "genre_only": ss.cv_r2(y2, effects=[g2]),
        "audio_plus_genre": ss.cv_r2(y2, x2, [g2]), "artist_only": ss.cv_r2(y2, effects=[a2], shrink=[2.0]),
        "audio_genre_artist": ss.cv_r2(y2, x2, [g2, a2], shrink=[0.0, 2.0]),
    }
    other = read("genre_tracks", ["track_id", "genre", "popularity"])
    shared = frame[["track_id", "genre", "popularity"]].merge(other, on="track_id", suffixes=("", "_other"))
    medians = frame.groupby("genre")["popularity"].median().sort_values(ascending=False)
    audit = {
        "shared_tracks": int(shared["track_id"].nunique()),
        "label_agreement_share": float((shared["genre"] == shared["genre_other"]).groupby(shared["track_id"]).max().mean()),
        "popularity_snapshot_spearman": float(shared.drop_duplicates("track_id")[["popularity", "popularity_other"]].corr(method="spearman").iloc[0, 1]),
        "highest_median_popularity": medians.head(5).to_dict(), "lowest_median_popularity": medians.tail(5).to_dict(),
        "genres_with_zero_median": int((medians == 0).sum()), "genres": int(len(medians)),
    }
    return save("popularity", {
        "definition": "Hit: top 10% of Spotify popularity within release year, popularity above 0. "
                      "Low: bottom half within release year.",
        "n": len(frame), "n_hit": int(frame["hit"].sum()), "n_low": int(frame["low"].sum()), "base_rate": base,
        "comparisons": comparisons, "correlations": correlations, "lift": lift,
        "envelope": {"bands": single, "inside_all_hits": float(inside[hit_mask].mean()),
                     "inside_all_others": float(inside[~hit_mask].mean())},
        "dispersion": dispersion, "ladder": ladder, "regression": regression,
        "within_genre_year_comparisons": within_genre, "ladder_independent_genre_sample": second_ladder,
        "genre_label_audit": audit,
        "replication_genre_sample_spearman": replication,
        "hit_rate_by_genre": by_genre_share.to_dict(),
    })


# ---------------------------------------------------------------- C. evolution


def _weighted_year(frame: pd.DataFrame, feature: str, rng, draws=300) -> dict:
    x, w = frame[feature].to_numpy(dtype=float), frame["slots"].to_numpy(dtype=float)
    index = rng.integers(0, len(x), (draws, len(x)))
    boot = (x[index] * w[index]).sum(axis=1) / w[index].sum(axis=1)
    return {"mean": float(np.average(x, weights=w)), "low": float(np.percentile(boot, 2.5)),
            "high": float(np.percentile(boot, 97.5))}


def evolution() -> dict:
    rng = np.random.default_rng(RANDOM_SEED)
    songs = read("billboard_songs", ["song_id", "song", "performer", "debut_year", "weeks", "peak", "track_id",
                                     "duration_ms", "mode"] + AUDIO_FEATURES)
    weeks = read("billboard_weeks")
    weeks["year"] = weeks["week"].dt.year
    songs = songs.dropna(subset=["track_id"]).copy()
    songs["duration_min"] = songs["duration_ms"] / 60000
    slots = weeks.groupby(["year", "song_id"]).size().rename("slots").reset_index()
    panel = slots.merge(songs, on="song_id")
    years = list(range(1959, 2021))
    panel = panel.loc[panel["year"].isin(years)]
    coverage = (panel.groupby("year")["slots"].sum() / weeks.loc[weeks["year"].isin(years)].groupby("year").size())
    catalogue = read("catalogue", ["year", "duration_ms"] + AUDIO_FEATURES)
    catalogue = catalogue.loc[catalogue["year"].between(1959, 2020)].copy()
    catalogue["duration_min"] = catalogue["duration_ms"] / 60000
    counts = catalogue.groupby("year").size()
    features = ["tempo", "loudness", "energy", "danceability", "valence", "duration_min", "acousticness", "speechiness"]
    series, trends, tests = {}, {}, []
    for feature in features:
        rows = [_weighted_year(part, feature, rng) for _, part in panel.groupby("year")]
        chart_mean = np.array([r["mean"] for r in rows])
        grouped = catalogue.groupby("year")[feature]
        cat_mean = grouped.mean().reindex(years)
        cat_se = (grouped.std() / np.sqrt(counts)).reindex(years)
        series[feature] = {
            "chart": chart_mean, "chart_low": [r["low"] for r in rows], "chart_high": [r["high"] for r in rows],
            "catalogue": cat_mean.to_numpy(), "catalogue_low": (cat_mean - 1.96 * cat_se).to_numpy(),
            "catalogue_high": (cat_mean + 1.96 * cat_se).to_numpy(),
        }
        chart_trend = ss.theil_sen(years, chart_mean)
        valid = cat_mean.notna().to_numpy() & (counts.reindex(years).fillna(0).to_numpy() >= 200)
        cat_years, cat_values = np.array(years)[valid], cat_mean.to_numpy()[valid]
        cat_trend = ss.theil_sen(cat_years, cat_values)
        gap = chart_mean[valid] - cat_values
        trends[feature] = {
            "chart": {**chart_trend, "per_decade": chart_trend["slope_per_year"] * 10,
                      "change_points": ss.change_points(years, chart_mean),
                      "anomalies": ss.robust_anomalies(years, chart_mean)},
            "catalogue": {**cat_trend, "per_decade": cat_trend["slope_per_year"] * 10,
                          "change_points": ss.change_points(cat_years, cat_values),
                          "first_year": int(cat_years[0])},
            "trajectory_correlation": float(st.spearmanr(chart_mean[valid], cat_values).statistic),
            "gap_mean": float(gap.mean()), "gap_first_decade": float(gap[:10].mean()), "gap_last_decade": float(gap[-10:].mean()),
        }
        tests += [("chart", feature, chart_trend["p"]), ("catalogue", feature, cat_trend["p"])]
    for (scope, feature, _), q in zip(tests, ss.bh_fdr([t[2] for t in tests])):
        trends[feature][scope]["q_fdr"] = float(q)

    unique = songs.loc[songs["debut_year"].between(1960, 2019)].copy()
    unique["decade"] = unique["debut_year"] // 10 * 10
    decades = {}
    for feature in features:
        groups = {f"{d}s": part[feature].to_numpy(dtype=float) for d, part in unique.groupby("decade")}
        test = ss.omnibus(list(groups.values()))
        dunn = ss.dunn_test(groups)
        adjacent = [r for r in dunn if int(r["b"][:4]) - int(r["a"][:4]) == 10]
        decades[feature] = {
            **test, "assumptions": ss.assumption_checks(list(groups.values())), "adjacent_dunn": adjacent,
            "summary": {k: {"median": float(np.median(v)), "q1": float(np.percentile(v, 25)), "q3": float(np.percentile(v, 75)),
                            "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)), "n": len(v),
                            "hist": histogram(v, feature, 24)} for k, v in groups.items()},
            "range": FEATURE_BINS[feature],
        }
    charts_recent = recent_global_chart(features)
    turnover = songs.groupby("debut_year")["weeks"].median().reindex(years)
    return save("evolution", {
        "years": years, "features": features, "series": series, "trends": trends, "decades": decades,
        "catalogue_tracks_by_year": counts.reindex(years).fillna(0).to_numpy(),
        "chart_slot_audio_coverage": coverage.reindex(years).to_numpy(),
        "median_weeks_by_debut_year": turnover.to_numpy(), "songs_with_audio": len(songs), "recent_global": charts_recent,
    })


def recent_global_chart(features) -> dict:
    """Spotify Global Top 50 as a modern, separately sourced reference series."""
    late = read("charts_2023_2025", ["date", "country", "rank", "audio_valid", "duration_ms"] + AUDIO_FEATURES)
    late = late.loc[(late["country"] == "GLOBAL") & late["audio_valid"]].copy()
    late["year"] = late["date"].dt.year
    late["duration_min"] = late["duration_ms"] / 60000
    early = read("global_daily_2017_2021", ["date", "rank", "song_key"])
    early = early.loc[early["rank"] <= 50].copy()
    songs = read("chart17_songs", ["song_key", "feature_id"]).dropna(subset=["feature_id"])
    tracks = read("tracks", ["track_id", "duration_ms"] + AUDIO_FEATURES)
    tracks = tracks.loc[tracks["track_id"].isin(set(songs["feature_id"]))]
    early = early.merge(songs, on="song_key").merge(tracks, left_on="feature_id", right_on="track_id")
    early["year"] = early["date"].dt.year
    early["duration_min"] = early["duration_ms"] / 60000
    combined = pd.concat([early[["year"] + features], late[["year"] + features]])
    return {"years": sorted(combined["year"].unique().tolist()),
            "means": {f: combined.groupby("year")[f].mean().to_numpy() for f in features},
            "note": "Daily Global Top 50 slots. 2023 starts in October and 2025 ends in June; 2022 is not covered."}


# ---------------------------------------------------------------- D. genre


def archetype_name(centre: dict) -> str:
    if centre["instrumentalness"] > 0.9 and centre["acousticness"] > 0.5:
        return "Quiet instrumental"
    if centre["instrumentalness"] > 0.9:
        return "Electronic instrumental" if centre["energy"] > 0 else "Ambient instrumental"
    if centre["speechiness"] > 0.9:
        return "Rhythmic spoken-vocal"
    if centre["liveness"] > 0.9:
        return "Live-room energy"
    if centre["acousticness"] > 0.6 and centre["valence"] > 0:
        return "Warm acoustic"
    if centre["acousticness"] > 0.6:
        return "Intimate acoustic"
    if centre["energy"] > 0.5 and centre["valence"] < -0.2:
        return "Heavy high-intensity"
    if centre["danceability"] > 0.4 and centre["valence"] > 0.3:
        return "Bright dance-pop"
    if centre["energy"] > 0.3:
        return "Driving mainstream"
    return "Mid-tempo balladry"


def genre() -> dict:
    rows = genre_universe()
    frame = rows.drop_duplicates("track_id").reset_index(drop=True)
    genres = sorted(rows["genre"].unique())
    tests = []
    summaries = {g: {} for g in genres}
    for feature in DISPLAY_FEATURES:
        grouped = {g: part[feature].to_numpy(dtype=float) for g, part in rows.groupby("genre")}
        arrays = [grouped[g] for g in genres]
        tests.append({"feature": feature, **ss.omnibus(arrays), "assumptions": ss.assumption_checks(arrays)})
        for g in genres:
            q = np.percentile(grouped[g], [5, 25, 50, 75, 95])
            summaries[g][feature] = {"box": q, "hist": histogram(grouped[g], feature, 20)}
    for row, q in zip(tests, ss.bh_fdr([t["kruskal_p"] for t in tests])):
        row["kruskal_q_fdr"] = float(q)
    tests.sort(key=lambda r: -r["epsilon_squared"])
    focal = rows.loc[rows["genre"].isin(FOCAL_GENRES)]
    post_hoc = {}
    for feature in ("energy", "valence", "acousticness", "danceability"):
        groups = {g: focal.loc[focal["genre"] == g, feature].to_numpy(dtype=float) for g in FOCAL_GENRES}
        dunn = ss.dunn_test(groups)
        howell = ss.games_howell(groups)
        post_hoc[feature] = {
            "pairs": len(dunn), "dunn_significant_holm": sum(r["p_holm"] < .05 for r in dunn),
            "games_howell_significant": sum(r["p"] < .05 for r in howell),
            "closest_pairs": sorted(dunn, key=lambda r: abs(r["z"]))[:5], "furthest_pairs": sorted(dunn, key=lambda r: -abs(r["z"]))[:5],
        }
    z_rows = transform_features(rows, AUDIO_FEATURES)
    codes = rows["genre"].map({g: i for i, g in enumerate(genres)}).to_numpy()
    centroids = np.vstack([z_rows[codes == i].mean(axis=0) for i in range(len(genres))])
    distance = squareform(pdist(centroids))
    link = hierarchy.linkage(centroids, "ward")
    order = hierarchy.leaves_list(link)
    similarity = 1 - squareform(pdist(centroids, "cosine"))
    families = hierarchy.fcluster(link, 10, "maxclust")

    membership = rows.groupby("track_id")["genre"].agg(list)
    co_counts = np.zeros((len(genres), len(genres)))
    index = {g: i for i, g in enumerate(genres)}
    multi = membership.loc[membership.str.len() > 1]
    for items in multi:
        positions = [index[g] for g in items]
        for a in positions:
            for b in positions:
                if a != b:
                    co_counts[a, b] += 1
    sizes = rows["genre"].value_counts().reindex(genres).to_numpy()
    jaccard = co_counts / (sizes[:, None] + sizes[None, :] - co_counts)
    np.fill_diagonal(jaccard, 0)
    i_upper, j_upper = np.triu_indices(len(genres), 1)
    top = np.argsort(-jaccard[i_upper, j_upper])[:30]
    overlap_pairs = [{"a": genres[i_upper[t]], "b": genres[j_upper[t]], "jaccard": jaccard[i_upper[t], j_upper[t]],
                      "name_contained": genres[i_upper[t]] in genres[j_upper[t]] or genres[j_upper[t]] in genres[i_upper[t]],
                      "shared_tracks": co_counts[i_upper[t], j_upper[t]], "centroid_distance": distance[i_upper[t], j_upper[t]]}
                     for t in top]
    linked = jaccard[i_upper, j_upper] > 0
    overlap_vs_distance = st.spearmanr(jaccard[i_upper, j_upper][linked], distance[i_upper, j_upper][linked])

    # Can audio alone recover the genre label? Whole artists are held out in each fold.
    artists = pd.factorize(rows["artist"])[0]
    rng = np.random.default_rng(RANDOM_SEED)
    fold = rng.integers(0, 5, artists.max() + 1)[artists]
    correct = np.zeros(len(rows), bool)
    top5 = np.zeros(len(rows), bool)
    entropy = np.zeros(len(rows))
    confusion = np.zeros((len(genres), len(genres)))
    for f in range(5):
        test = fold == f
        means, precision = ss.lda_fit(z_rows[~test], codes[~test], len(genres))
        posterior = ss.lda_posterior(z_rows[test], means, precision)
        predicted = posterior.argmax(axis=1)
        correct[test] = predicted == codes[test]
        top5[test] = (np.argsort(-posterior, axis=1)[:, :5] == codes[test][:, None]).any(axis=1)
        entropy[test] = -(posterior * np.log(np.clip(posterior, 1e-12, 1))).sum(axis=1) / np.log(len(genres))
        np.add.at(confusion, (codes[test], predicted), 1)
    recall = np.diag(confusion) / confusion.sum(axis=1)
    importance = {}
    held = fold == 0
    baseline = correct[held].mean()
    for j, feature in enumerate(AUDIO_FEATURES):
        keep = [i for i in range(len(AUDIO_FEATURES)) if i != j]
        means, precision = ss.lda_fit(z_rows[~held][:, keep], codes[~held], len(genres))
        accuracy = (ss.lda_posterior(z_rows[held][:, keep], means, precision).argmax(axis=1) == codes[held]).mean()
        importance[feature] = float(baseline - accuracy)
    rows_entropy = pd.Series(entropy, index=rows["track_id"].to_numpy()).groupby(level=0).mean()
    is_multi = rows_entropy.index.isin(multi.index)
    hybrid = ss.compare_two(rows_entropy[is_multi], rows_entropy[~is_multi])
    confused = confusion / confusion.sum(axis=1, keepdims=True)
    np.fill_diagonal(confused, 0)
    worst = np.dstack(np.unravel_index(np.argsort(-confused.ravel())[:20], confused.shape))[0]

    # Archetypes: discovered from audio only, without genre labels.
    z = transform_features(frame, AUDIO_FEATURES)
    subset = rng.choice(len(z), min(30000, len(z)), replace=False)
    search = []
    for k in range(4, 13):
        fit = ss.kmeans(z[subset], k, restarts=3)
        ari = []
        for b in range(4):
            sample = rng.choice(subset, len(subset) // 2, replace=False)
            other = ss.kmeans(z[sample], k, seed=100 + b, restarts=1)
            ari.append(ss.adjusted_rand(ss._assign(z[subset], other["centres"])[0], fit["labels"]))
        search.append({"k": k, "inertia": fit["inertia"], "silhouette": ss.silhouette(z[subset], fit["labels"]),
                       "stability_ari": float(np.mean(ari))})
    stable = [s for s in search if s["stability_ari"] >= 0.7 and s["k"] >= 5] or search
    chosen = max(stable, key=lambda s: s["silhouette"])["k"]
    model = ss.kmeans(z, chosen, restarts=4)
    labels = model["labels"]
    components = ss.pca(z, 3)
    raw_means = frame[DISPLAY_FEATURES].groupby(labels).median()
    archetypes, used = [], {}
    for c in range(chosen):
        centre = dict(zip(AUDIO_FEATURES, model["centres"][c]))
        name = archetype_name(centre)
        used[name] = used.get(name, 0) + 1
        label = name if used[name] == 1 else f"{name} {used[name]}"
        members = frame.loc[labels == c]
        counts = rows.loc[rows["track_id"].isin(members["track_id"])]["genre"].value_counts()
        lift = (counts / counts.sum()) / (rows["genre"].value_counts() / len(rows))
        lift = lift.dropna()
        distinctive = lift.loc[counts.reindex(lift.index) >= 50].sort_values(ascending=False).head(6)
        # Examples are typical members: known tracks closest to the cluster centre.
        distance_to_centre = ((z[labels == c] - model["centres"][c]) ** 2).sum(axis=1)
        known = members.assign(distance_to_centre=distance_to_centre)
        known = known.loc[known["popularity"] >= 50] if (known["popularity"] >= 50).sum() >= 20 else known
        examples = known.sort_values("distance_to_centre").drop_duplicates("artist").head(5)
        archetypes.append({
            "id": c, "name": label, "n": len(members), "share": len(members) / len(frame), "z_centre": centre,
            "medians": raw_means.loc[c].to_dict(), "median_popularity": float(members["popularity"].median()),
            "top_genres": [{"genre": g, "lift": float(v)} for g, v in distinctive.items()],
            "examples": [f"{r['name']} / {r['artist']}" for _, r in examples.iterrows()],
        })
    label_of = pd.Series(labels, index=frame["track_id"])
    crosstab = pd.crosstab(rows["genre"], rows["track_id"].map(label_of))
    sample = rng.choice(len(z), 3000, replace=False)
    pop_groups = [frame.loc[labels == c, "popularity"].to_numpy(dtype=float) for c in range(chosen)]
    (PROCESSED_DIR / "archetype_model.json").write_text(json.dumps({
        "features": AUDIO_FEATURES, "centres": model["centres"].tolist(),
        "names": [a["name"] for a in archetypes],
        "note": "centres are in transformed, standardised units",
    }))
    pd.DataFrame({"track_id": frame["track_id"], "archetype": labels}).to_parquet(PROCESSED_DIR / "archetypes.parquet", index=False)
    return save("genre", {
        "n_rows": len(rows), "n_tracks": len(frame), "genres": genres, "tests": tests, "summaries": summaries,
        "post_hoc_focal": post_hoc, "focal_genres": FOCAL_GENRES,
        "similarity": {"order": [genres[i] for i in order], "cosine": similarity[order][:, order].round(3),
                       "families": {genres[i]: int(families[i]) for i in range(len(genres))}},
        "overlap": {"multi_genre_tracks": len(multi), "share_of_tracks": len(multi) / len(membership), "top_pairs": overlap_pairs,
                    "jaccard_vs_audio_distance_spearman": float(overlap_vs_distance.statistic),
                    "jaccard_vs_audio_distance_p": float(overlap_vs_distance.pvalue),
                    "pairs_with_overlap": int(linked.sum())},
        "classifier": {"model": "linear discriminant analysis, 5 folds holding out whole artists",
                       "accuracy": correct.mean(), "top5": top5.mean(), "chance": 1 / len(genres),
                       "recall": {genres[i]: recall[i] for i in np.argsort(-recall)},
                       "accuracy_drop_without_feature": importance,
                       "most_confused": [{"true": genres[a], "predicted": genres[b], "share": confused[a, b]} for a, b in worst]},
        "hybrid": {"multi_vs_single_entropy": hybrid},
        "archetypes": {"search": search, "k": chosen, "items": archetypes, "features": AUDIO_FEATURES,
                       "genre_association": ss.chi_square(crosstab.to_numpy()),
                       "popularity_test": {**ss.omnibus(pop_groups), "assumptions": ss.assumption_checks(pop_groups)},
                       "pca": {"explained": components["explained"], "loadings": components["loadings"],
                               "x": components["scores"][sample, 0].round(2), "y": components["scores"][sample, 1].round(2),
                               "cluster": labels[sample]}},
    })


# ---------------------------------------------------------------- E. sadness


def sadness() -> dict:
    frame = popularity_universe()
    genre_codes = pd.factorize(frame["genre"])[0]
    year_codes = pd.factorize(frame["year"])[0]
    artist_codes = pd.factorize(frame["artist"])[0]
    band = np.select([frame["valence"] < 1 / 3, frame["valence"] < 2 / 3], ["low", "mid"], "high")
    frame["valence_band"] = band
    rates = {}
    hit_all = frame["hit"].to_numpy()
    base_rate_high = hit_all[band == "high"].mean()
    for name in ("low", "mid", "high"):
        mask = band == name
        k, n = hit_all[mask].sum(), mask.sum()
        p = k / n
        low, high = wilson(k, n)
        log_rr = np.log(p / base_rate_high)
        other_k, other_n = hit_all[band == "high"].sum(), (band == "high").sum()
        se = np.sqrt(1 / max(k, 1) - 1 / n + 1 / other_k - 1 / other_n) if name != "high" else 0
        rates[name] = {"n": int(n), "hits": int(k), "hit_rate": p, "ci": [low, high], "risk_ratio_vs_high": float(np.exp(log_rr)),
                       "rr_ci": [float(np.exp(log_rr - 1.96 * se)), float(np.exp(log_rr + 1.96 * se))],
                       "median_pop_pct": float(frame.loc[mask, "pop_pct"].median())}
    table = pd.crosstab(band, hit_all)
    valence_hit = ss.chi_square(table.to_numpy())
    # Adjusted hit-rate within genre and year: expected rate is the genre-year average.
    expected = frame.groupby(["genre", "year"])["hit"].transform("mean").to_numpy(dtype=float)
    adjusted = {name: float(hit_all[band == name].sum() / expected[band == name].sum()) for name in ("low", "mid", "high")}

    v_bin = np.minimum((frame["valence"].to_numpy() * 10).astype(int), 9)
    e_bin = np.minimum((frame["energy"].to_numpy() * 10).astype(int), 9)
    cell = v_bin * 10 + e_bin
    n_cell = np.bincount(cell, minlength=100)
    hit_cell = np.bincount(cell, hit_all.astype(float), 100)
    pct_cell = np.bincount(cell, frame["pop_pct"].to_numpy(dtype=float), 100)
    expected_cell = np.bincount(cell, expected, 100)
    with np.errstate(divide="ignore", invalid="ignore"):
        surface = {"n": n_cell.reshape(10, 10), "hit_rate": np.where(n_cell >= 200, hit_cell / n_cell, np.nan).reshape(10, 10),
                   "mean_pct": np.where(n_cell >= 200, pct_cell / n_cell, np.nan).reshape(10, 10),
                   "lift_adjusted": np.where(n_cell >= 200, hit_cell / expected_cell, np.nan).reshape(10, 10)}
    z = lambda c: ((frame[c] - frame[c].mean()) / frame[c].std()).to_numpy(dtype=float)
    v, e = z("valence"), z("energy")
    controls = ["danceability", "acousticness", "loudness", "tempo", "speechiness", "instrumentalness", "liveness", "duration_min"]
    x = np.column_stack([v, e, v * e] + [z(c) for c in controls])
    names = ["valence", "energy", "valence_x_energy"] + controls
    y = frame["pop_pct"].to_numpy(dtype=float) * 100
    model = ss.ols_cluster(y, x, names, artist_codes, [genre_codes, year_codes])
    no_control = ss.ols_cluster(y, x[:, :3], names[:3], artist_codes, [genre_codes, year_codes])
    year_only = ss.ols_cluster(y, x, names, artist_codes, [year_codes])
    terms = {c["term"]: c for c in model["coefficients"]}
    slopes = {f"energy_{label}": terms["valence"]["beta"] + level * terms["valence_x_energy"]["beta"]
              for label, level in (("minus_1sd", -1), ("mean", 0), ("plus_1sd", 1))}
    per_genre = []
    for name, part in frame.groupby("genre"):
        if part["genre_hit"].sum() < 50:
            continue
        rho = st.spearmanr(part["valence"], part["genre_pct"])
        hits, others = part.loc[part["genre_hit"], "valence"], part.loc[~part["genre_hit"], "valence"]
        per_genre.append({"genre": name, "n": len(part), "hits": int(part["genre_hit"].sum()), "rho": float(rho.statistic),
                          "p": float(rho.pvalue), "hit_median_valence": float(hits.median()),
                          "other_median_valence": float(others.median()), "delta": ss.cliffs_delta(hits, others),
                          "low_valence_share_of_hits": float((hits < 1 / 3).mean()),
                          "hit_median_energy": float(part.loc[part["genre_hit"], "energy"].median())})
    for row, q in zip(per_genre, ss.bh_fdr([r["p"] for r in per_genre])):
        row["q_fdr"] = float(q)
    per_genre.sort(key=lambda r: r["rho"])
    by_year = frame.loc[frame["hit"]].groupby("year").agg(valence=("valence", "median"), low_share=("valence", lambda s: (s < 1 / 3).mean()), n=("valence", "size"))
    all_year = frame.groupby("year")["valence"].median()
    trend_years = by_year.index[by_year.index <= 2022].to_numpy()
    songs = read("billboard_songs", ["debut_year", "weeks", "peak", "valence", "energy"]).dropna(subset=["valence"])
    songs = songs.loc[songs["debut_year"].between(1960, 2019)]
    songs["decade"] = songs["debut_year"] // 10 * 10
    billboard = []
    for decade, part in songs.groupby("decade"):
        top = part.loc[part["peak"] <= 10]
        rest = part.loc[part["peak"] > 40]
        billboard.append({"decade": int(decade), "top10_low_valence_share": float((top["valence"] < 1 / 3).mean()),
                          "all_low_valence_share": float((part["valence"] < 1 / 3).mean()),
                          "rho_valence_weeks": float(st.spearmanr(part["valence"], part["weeks"]).statistic),
                          "top10_median_valence": float(top["valence"].median()), "below40_median_valence": float(rest["valence"].median()),
                          "n_top10": len(top)})
    return save("sadness", {
        "bands": {"low": "valence below 1/3", "mid": "1/3 to 2/3", "high": "2/3 or above"},
        "rates": rates, "chi_square_band_by_hit": valence_hit, "observed_over_expected_within_genre_year": adjusted,
        "surface": surface, "interaction_model": model, "interaction_without_controls": no_control, "interaction_year_effects_only": year_only, "simple_slopes_valence": slopes,
        "per_genre": per_genre, "share_genres_negative_significant": float(np.mean([r["rho"] < 0 and r["q_fdr"] < .05 for r in per_genre])),
        "share_genres_positive_significant": float(np.mean([r["rho"] > 0 and r["q_fdr"] < .05 for r in per_genre])),
        "hit_valence_by_year": {"years": by_year.index.to_numpy(), "hit_median": by_year["valence"].to_numpy(),
                                "all_median": all_year.reindex(by_year.index).to_numpy(), "low_share": by_year["low_share"].to_numpy(),
                                "trend": ss.theil_sen(trend_years, by_year.loc[trend_years, "valence"].to_numpy(), block=3)},
        "billboard_by_decade": billboard,
    })
