"""Investigations F and G, the good / liked / popular comparison, and dashboard data."""
from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as st

from . import stats as ss
from .analysis import DISPLAY_FEATURES, genre_universe, popularity_universe, read, save, wilson
from .common import COUNTRY_NAMES, ISO_TO_GROUP, REGION_GROUPS
from .config import AUDIO_FEATURES, PROCESSED_DIR, RANDOM_SEED, RESULTS_DIR

AFFECTION_TAGS = {
    "favorite", "favorites", "favourite", "favourites", "favorite songs", "favourite songs", "love", "loved",
    "love it", "i love this song", "love this", "awesome", "amazing", "beautiful", "masterpiece", "perfect",
    "best", "brilliant", "great", "wonderful", "excellent", "my favorite", "my favourite",
}
QUARTER_MIN_DAYS = 60


def song_audio() -> pd.DataFrame:
    """One audio profile per song key, from Spotify IDs first."""
    columns = AUDIO_FEATURES + ["duration_ms", "mode"]
    late = read("charts_2023_2025", ["song_key", "audio_valid"] + columns)
    late = late.loc[late["audio_valid"]].drop(columns="audio_valid").drop_duplicates("song_key")
    songs = read("chart17_songs", ["song_key", "feature_id"]).dropna(subset=["feature_id"])
    tracks = read("tracks", ["track_id"] + columns)
    tracks = tracks.loc[tracks["track_id"].isin(set(songs["feature_id"]))]
    early = songs.merge(tracks, left_on="feature_id", right_on="track_id").drop(columns=["feature_id", "track_id"])
    combined = pd.concat([late, early], ignore_index=True).drop_duplicates("song_key").set_index("song_key")
    combined["duration_min"] = combined["duration_ms"] / 60000
    return combined.drop(columns="duration_ms")


def chart_slots() -> pd.DataFrame:
    """Song x country x quarter days inside a Top 50, for both chart sources."""
    early = read("charts_2017_2021", ["song_key", "country", "quarter", "days_top50"])
    early = early.loc[early["days_top50"] > 0].rename(columns={"days_top50": "days"})
    early["period"] = "2017-2021"
    late = read("charts_2023_2025", ["song_key", "country", "date"]).dropna(subset=["song_key"])
    late["quarter"] = late["date"].dt.year * 10 + late["date"].dt.quarter
    late = late.groupby(["song_key", "country", "quarter"], as_index=False).size().rename(columns={"size": "days"})
    late["period"] = "2023-2025"
    return pd.concat([early, late], ignore_index=True)


def profile_table(slots: pd.DataFrame, audio: pd.DataFrame, features: list[str]):
    grouped = slots.groupby(["period", "country", "song_key"], as_index=False)["days"].sum()
    joined = grouped.join(audio[features], on="song_key")
    joined["matched"] = joined[features[0]].notna()
    coverage = joined.groupby(["period", "country"])[["days"]].sum().join(
        joined.loc[joined["matched"]].groupby(["period", "country"])[["days"]].sum().rename(columns={"days": "matched_days"}))
    coverage["coverage"] = coverage["matched_days"] / coverage["days"]
    usable = joined.loc[joined["matched"]].copy()
    for feature in features:
        usable[feature + "_w"] = usable[feature] * usable["days"]
    sums = usable.groupby(["period", "country"])[[f + "_w" for f in features] + ["days"]].sum()
    profile = pd.DataFrame({f: sums[f + "_w"] / sums["days"] for f in features})
    return usable[["period", "country", "song_key", "days"] + features], profile, coverage


def permutation_test(frame: pd.DataFrame, feature: str, n_perm=300, seed=RANDOM_SEED) -> dict:
    """Between-country variation of chart-day-weighted means, against shuffled songs."""
    rng = np.random.default_rng(seed)
    codes = pd.factorize(frame["country"])[0]
    x, w = frame[feature].to_numpy(dtype=float), frame["days"].to_numpy(dtype=float)
    total = np.bincount(codes, w)

    def statistic(values, weights):
        sums = np.bincount(codes, values * weights)
        grand = (values * weights).sum() / weights.sum()
        return (np.bincount(codes, weights) * (sums / np.bincount(codes, weights) - grand) ** 2).sum()

    observed = statistic(x, w)
    pairs = np.column_stack([x, w])
    null = [statistic(*rng.permutation(pairs).T) for _ in range(n_perm)]
    grand = np.average(x, weights=w)
    ss_total = (w * (x - grand) ** 2).sum()
    return {"between_share_of_variance": float(observed / ss_total), "null_mean_share": float(np.mean(null) / ss_total),
            "p_permutation": float((1 + np.sum(np.array(null) >= observed)) / (n_perm + 1)), "countries": int(len(total))}


def quarter_sets(slots: pd.DataFrame, quarters, countries, top=None):
    """Songs per country-quarter. With top set, keep only the N most persistent songs."""
    subset = slots.loc[(slots["days"] >= 7) & slots["quarter"].isin(quarters) & slots["country"].isin(countries)]
    if top:
        subset = subset.sort_values(["days", "song_key"], ascending=[False, True]).groupby(["quarter", "country"]).head(top)
    return {key: set(part["song_key"]) for key, part in subset.groupby(["quarter", "country"])}


def geography() -> dict:
    rng = np.random.default_rng(RANDOM_SEED)
    features = AUDIO_FEATURES + ["duration_min"]
    audio = song_audio()
    slots = chart_slots()
    markets = slots.loc[slots["country"] != "GLOBAL"]
    usable, profile, coverage = profile_table(markets, audio, features)
    world_used, world_profile, _ = profile_table(slots.loc[slots["country"] == "GLOBAL"], audio, features)
    results: dict = {"features": features, "periods": {}}
    for period in ("2017-2021", "2023-2025"):
        p = profile.loc[period]
        cov = coverage.loc[period]
        keep = cov.index[(cov["coverage"] >= 0.6) & (cov["days"] >= 2000)]
        p = p.loc[keep]
        scale = usable.loc[usable["period"] == period, features].std()
        world = world_profile.loc[period]
        z = (p - world.iloc[0]) / scale
        rows = usable.loc[(usable["period"] == period) & usable["country"].isin(keep)]
        tests = {}
        for feature in features:
            groups = [g[feature].to_numpy(dtype=float) for _, g in rows.groupby("country")]
            tests[feature] = {**ss.omnibus(groups), "permutation": permutation_test(rows, feature)}
        for feature, q in zip(features, ss.bh_fdr([tests[f]["permutation"]["p_permutation"] for f in features])):
            tests[feature]["permutation"]["q_fdr"] = float(q)
        distance = np.sqrt(((z.to_numpy()[:, None] - z.to_numpy()[None]) ** 2).sum(axis=2))
        from scipy.cluster import hierarchy
        order = hierarchy.leaves_list(hierarchy.linkage(z.to_numpy(), "ward"))
        np.fill_diagonal(distance, np.inf)
        results["periods"][period] = {
            "countries": list(p.index), "names": [COUNTRY_NAMES[c] for c in p.index],
            "groups": [ISO_TO_GROUP[c] for c in p.index], "order": [p.index[i] for i in order],
            "profile": p.to_numpy(), "z_vs_global": z.to_numpy(), "global_profile": world.iloc[0].to_numpy(),
            "coverage": cov.loc[keep, "coverage"].to_numpy(), "tests": tests,
            "nearest": {p.index[i]: p.index[int(distance[i].argmin())] for i in range(len(p))},
            "distance_from_global": np.sqrt((z.to_numpy() ** 2).sum(axis=1)),
            "excluded_low_coverage": [c for c in cov.index if c not in keep],
        }
    results["reach"] = reach(markets, audio, features)
    results["convergence"] = convergence(markets, audio, rng)
    results["origin"] = origin_analysis(markets)
    yearly = markets.assign(year=markets["quarter"] // 10)
    _, by_year, cov_year = profile_table(yearly.assign(period=yearly["year"].astype(str)), audio, features)
    by_year = by_year.loc[cov_year["coverage"] >= 0.5]
    results["country_year"] = {"rows": [[int(y), c] + by_year.loc[(y, c)].round(3).tolist() for y, c in by_year.index]}
    return save("geography", results)


def reach(markets, audio, features) -> dict:
    late = markets.loc[markets["period"] == "2023-2025"].groupby(["song_key", "country"], as_index=False)["days"].sum()
    grouped = late.groupby("song_key")["days"]
    table = pd.DataFrame({"countries": grouped.size(), "days": grouped.sum(), "top": grouped.max()})
    table["hhi"] = (late["days"] ** 2).groupby(late["song_key"]).sum() / table["days"] ** 2
    table["effective_countries"] = 1 / table["hhi"]
    table["top_share"] = table["top"] / table["days"]
    table = table.loc[table["days"] >= 14].copy()
    table["class"] = np.select([table["countries"] >= 20, table["countries"] <= 3], ["international", "concentrated"], "regional")
    labels = read("charts_2023_2025", ["song_key", "name", "artist"]).drop_duplicates("song_key").set_index("song_key")
    joined = table.join(audio[features + ["mode"]], how="inner")
    counts = table["class"].value_counts().to_dict()
    groups = {c: joined.loc[joined["class"] == c] for c in ("international", "regional", "concentrated")}
    tests = []
    for feature in features:
        tests.append({"feature": feature, **ss.compare_two(groups["international"][feature], groups["concentrated"][feature]),
                      "regional_median": float(groups["regional"][feature].median()),
                      "kruskal": ss.omnibus([g[feature].to_numpy(dtype=float) for g in groups.values()])["kruskal_p"]})
    for row, q in zip(tests, ss.bh_fdr([t["p"] for t in tests])):
        row["q_fdr"] = float(q)
    mode = pd.crosstab(joined["class"], joined["mode"])
    examples = {}
    for name, part in groups.items():
        best = part.sort_values("days", ascending=False).head(10).join(labels)
        examples[name] = [{"name": r["name"], "artist": r["artist"], "countries": int(r["countries"]), "days": int(r["days"]),
                           "top_share": float(r["top_share"])} for _, r in best.iterrows()]
    hist, edges = np.histogram(table["countries"], bins=np.arange(1, 75, 3))
    return {"n_songs": len(table), "class_counts": counts, "rules": "international: 20+ countries; concentrated: 1-3; minimum 14 chart days",
            "tests": tests, "mode_by_class": ss.chi_square(mode.to_numpy()), "examples": examples,
            "countries_hist": hist, "countries_edges": edges,
            "median_effective_countries": float(table["effective_countries"].median()),
            "single_country_share": float((table["countries"] == 1).mean())}


def convergence(markets, audio, rng) -> dict:
    days_seen = pd.concat([
        read("charts_2023_2025", ["date"]).drop_duplicates().assign(q=lambda d: d["date"].dt.year * 10 + d["date"].dt.quarter),
    ])
    late_days = days_seen.groupby("q").size()
    late_quarters = [int(q) for q, n in late_days.items() if n >= QUARTER_MIN_DAYS]
    early_quarters = [y * 10 + q for y in range(2017, 2022) for q in range(1, 5)]
    quarters = early_quarters + late_quarters
    present = markets.loc[(markets["days"] >= 7) & markets["quarter"].isin(quarters)].groupby("quarter")["country"].agg(set)
    panel = sorted(set.intersection(*[present[q] for q in quarters if q in present.index]))
    sets = quarter_sets(markets, quarters, panel)
    core_sets = quarter_sets(markets, quarters, panel, top=25)
    index = {c: i for i, c in enumerate(panel)}
    frame = markets.loc[(markets["days"] >= 7) & markets["quarter"].isin(quarters) & markets["country"].isin(panel)]
    joined = frame.join(audio[AUDIO_FEATURES], on="song_key").dropna(subset=["energy"])
    transformed = joined[AUDIO_FEATURES].copy()
    scale = transformed.std()
    n = len(panel)
    jaccard = np.full((len(quarters), n, n), np.nan)
    distance = np.full((len(quarters), n, n), np.nan)
    core = np.full((len(quarters), n, n), np.nan)
    noise_floor = np.zeros(len(quarters))
    set_sizes = np.zeros(len(quarters))
    for q_i, quarter in enumerate(quarters):
        set_sizes[q_i] = np.mean([len(sets[(quarter, c)]) for c in panel])
        for a, b in combinations(panel, 2):
            sa, sb = sets[(quarter, a)], sets[(quarter, b)]
            value = len(sa & sb) / len(sa | sb)
            jaccard[q_i, index[a], index[b]] = jaccard[q_i, index[b], index[a]] = value
            ca, cb = core_sets[(quarter, a)], core_sets[(quarter, b)]
            core[q_i, index[a], index[b]] = core[q_i, index[b], index[a]] = len(ca & cb) / len(ca | cb)
        part = joined.loc[joined["quarter"] == quarter]
        weights = part["days"].to_numpy(dtype=float)
        codes = part["country"].map(index).to_numpy()
        means = np.column_stack([np.bincount(codes, part[f].to_numpy(dtype=float) * weights, n) / np.bincount(codes, weights, n)
                                 for f in AUDIO_FEATURES]) / scale.to_numpy()
        distance[q_i] = np.sqrt(((means[:, None] - means[None]) ** 2).sum(axis=2))
        # Noise floor: the distance expected if songs were dealt to countries at random.
        values = part[AUDIO_FEATURES].to_numpy(dtype=float) / scale.to_numpy()
        shuffled = []
        for _ in range(20):
            mixed = rng.permutation(codes)
            m = np.column_stack([np.bincount(mixed, values[:, j] * weights, n) / np.bincount(mixed, weights, n)
                                 for j in range(values.shape[1])])
            d = np.sqrt(((m[:, None] - m[None]) ** 2).sum(axis=2))
            shuffled.append(d[np.triu_indices(n, 1)].mean())
        noise_floor[q_i] = np.mean(shuffled)
    upper = np.triu_indices(n, 1)
    core_series = np.nanmean(core[:, upper[0], upper[1]], axis=1)
    overlap_series = np.nanmean(jaccard[:, upper[0], upper[1]], axis=1)
    distance_series = np.nanmean(distance[:, upper[0], upper[1]], axis=1)
    draws_j, draws_d, period_j, period_d = [], [], [], []
    is_late = np.array([q in late_quarters for q in quarters])
    for _ in range(500):
        pick = rng.integers(0, n, n)
        a, b = np.triu_indices(n, 1)
        valid = pick[a] != pick[b]
        ia, ib = pick[a][valid], pick[b][valid]
        j = np.nanmean(jaccard[:, ia, ib], axis=1)
        d = np.nanmean(distance[:, ia, ib], axis=1)
        draws_j.append(j), draws_d.append(d)
        period_j.append(j[is_late].mean() - j[~is_late].mean())
        period_d.append(d[is_late].mean() - d[~is_late].mean())
    position = np.arange(len(early_quarters))
    groups = np.array([ISO_TO_GROUP[c] for c in panel])
    same = groups[upper[0]] == groups[upper[1]]
    within, between = {}, {}
    for label, mask in (("2017-2021", ~is_late), ("2023-2025", is_late)):
        values = np.nanmean(jaccard[mask][:, upper[0], upper[1]], axis=0)
        within[label], between[label] = float(values[same].mean()), float(values[~same].mean())
    group_pairs = {}
    for group in REGION_GROUPS:
        chosen = (groups[upper[0]] == group) & (groups[upper[1]] == group)
        if chosen.sum():
            group_pairs[group] = [float(np.nanmean(jaccard[~is_late][:, upper[0][chosen], upper[1][chosen]])),
                                  float(np.nanmean(jaccard[is_late][:, upper[0][chosen], upper[1][chosen]])), int(chosen.sum())]
    return {
        "quarters": quarters, "is_late": is_late, "panel": panel, "panel_size": n,
        "rule": "Top 50 songs held for at least 7 days in a quarter; only countries present in every quarter",
        "jaccard": overlap_series, "jaccard_low": np.percentile(draws_j, 2.5, axis=0), "jaccard_high": np.percentile(draws_j, 97.5, axis=0),
        "audio_distance": distance_series, "audio_low": np.percentile(draws_d, 2.5, axis=0), "audio_high": np.percentile(draws_d, 97.5, axis=0),
        "core25_jaccard": core_series, "mean_songs_per_country_quarter": set_sizes,
        "audio_noise_floor": noise_floor, "audio_excess_over_noise": distance_series - noise_floor,
        "core25_period_change": float(core_series[is_late].mean() - core_series[~is_late].mean()),
        "audio_excess_period_change": float((distance_series - noise_floor)[is_late].mean() - (distance_series - noise_floor)[~is_late].mean()),
        "early_trend_core25": ss.theil_sen(position, core_series[:len(position)], block=4),
        "early_trend_jaccard": ss.theil_sen(position, overlap_series[:len(position)], block=4),
        "early_trend_audio": ss.theil_sen(position, distance_series[:len(position)], block=4),
        "period_change_jaccard": {"estimate": float(overlap_series[is_late].mean() - overlap_series[~is_late].mean()),
                                  "ci": np.percentile(period_j, [2.5, 97.5])},
        "period_change_audio": {"estimate": float(distance_series[is_late].mean() - distance_series[~is_late].mean()),
                                "ci": np.percentile(period_d, [2.5, 97.5])},
        "within_region_jaccard": within, "between_region_jaccard": between, "region_detail": group_pairs,
    }


def origin_analysis(markets) -> dict | None:
    path = PROCESSED_DIR / "artist_origin.parquet"
    if not path.exists():
        return None
    origins = pd.read_parquet(path)
    resolved = origins.loc[origins["status"] == "resolved"].set_index("key")["origin"]
    frame = markets.copy()
    frame["key"] = frame["song_key"].str.split("|", n=1).str[0]
    frame["origin"] = frame["key"].map(resolved)
    output = {"periods": {}}
    for period, part in frame.groupby("period"):
        known = part.dropna(subset=["origin"])
        share_known = known.groupby("country")["days"].sum() / part.groupby("country")["days"].sum()
        local = known.loc[known["origin"] == known["country"]].groupby("country")["days"].sum()
        totals = known.groupby("country")["days"].sum()
        local_share = (local / totals).fillna(0)
        world_share = known.groupby("origin")["days"].sum() / known["days"].sum()
        rows = []
        for country in totals.index:
            if share_known[country] < 0.3 or totals[country] < 1000:
                continue
            baseline = world_share.get(country, 0.0)
            rows.append({"country": country, "name": COUNTRY_NAMES[country], "group": ISO_TO_GROUP[country],
                         "local_share": float(local_share[country]), "known_share": float(share_known[country]),
                         "us_share": float(known.loc[(known["country"] == country) & (known["origin"] == "US"), "days"].sum() / totals[country]),
                         "world_share_of_local_artists": float(baseline),
                         "home_bias_ratio": float(local_share[country] / baseline) if baseline > 0 else None})
        known = known.assign(market_group=known["country"].map(ISO_TO_GROUP),
                             origin_group=known["origin"].map(lambda c: ISO_TO_GROUP.get(c, "Other origin")))
        flow = known.pivot_table(index="market_group", columns="origin_group", values="days", aggfunc="sum", fill_value=0)
        flow = flow.div(flow.sum(axis=1), axis=0)
        output["periods"][period] = {"countries": sorted(rows, key=lambda r: -r["local_share"]), "flow_rows": list(flow.index),
                                     "flow_columns": list(flow.columns), "flow": flow.to_numpy(),
                                     "top_origins": world_share.sort_values(ascending=False).head(12).to_dict()}
    return output


# ---------------------------------------------------------------- G. hits


def _standard(frame, columns):
    return ((frame[columns] - frame[columns].mean()) / frame[columns].std()).to_numpy(dtype=float)


def global_song_table() -> pd.DataFrame:
    daily = read("global_daily_2017_2021", ["date", "rank", "song_key", "streams"]).sort_values(["song_key", "date"])
    start, end = daily["date"].min(), daily["date"].max()
    rows = []
    for key, part in daily.groupby("song_key", sort=False):
        dates, streams, ranks = part["date"].to_numpy(), part["streams"].to_numpy(dtype=float), part["rank"].to_numpy()
        first = dates[0]
        total = np.nansum(streams)
        peak_i = int(np.nanargmax(streams))
        half_day = None
        offsets = ((dates - dates[peak_i]) / np.timedelta64(1, "D")).astype(int)
        lookup = dict(zip(offsets, streams))
        final_offset = int((np.datetime64(end) - dates[peak_i]) / np.timedelta64(1, "D"))
        for offset in range(1, final_offset + 1):
            if lookup.get(offset, 0) < streams[peak_i] / 2:
                half_day = offset
                break
        early = ((dates - first) / np.timedelta64(1, "D")) < 14
        month_day = pd.DatetimeIndex(dates)
        holiday = ((month_day.month == 12) | ((month_day.month == 11) & (month_day.day >= 20)) | ((month_day.month == 1) & (month_day.day <= 6)))
        rows.append((key, first, dates[-1], len(dates), int(ranks.min()), total, streams[peak_i], half_day,
                     float(np.nansum(streams[early]) / total), float(holiday.mean()), int((ranks <= 10).sum())))
    table = pd.DataFrame(rows, columns=["song_key", "first", "last", "days", "peak", "streams", "peak_streams", "half_life",
                                        "front_load", "holiday_share", "top10_days"])
    table["left_truncated"] = table["first"] <= start + pd.Timedelta(days=14)
    table["censored"] = table["last"] >= end - pd.Timedelta(days=14)
    table["span"] = (table["last"] - table["first"]).dt.days + 1
    table["seasonal"] = table["holiday_share"] >= 0.6
    table["debut_year"] = table["first"].dt.year
    table["artist_key"] = table["song_key"].str.split("|", n=1).str[0]
    return table


def _effects(y, x, names, clusters, effects):
    model = ss.ols_cluster(y, x, names, clusters, effects)
    for row, q in zip(model["coefficients"], ss.bh_fdr([c["p"] for c in model["coefficients"]])):
        row["q_fdr"] = float(q)
    return model


def year_adjusted(frame: pd.DataFrame, features: list[str], by: str) -> pd.DataFrame:
    """Express features relative to their debut-year mean, so eras are not compared."""
    out = frame.copy()
    out[features] = frame[features] - frame.groupby(by)[features].transform("mean")
    return out


def hits() -> dict:
    features = DISPLAY_FEATURES
    songs = read("billboard_songs").dropna(subset=["track_id"])
    songs = songs.loc[songs["debut_year"].between(1960, 2019)].copy()
    songs["duration_min"] = songs["duration_ms"] / 60000
    x = _standard(songs, features)
    years, performers = pd.factorize(songs["debut_year"])[0], pd.factorize(songs["performer"])[0]
    weeks, peak = np.log(songs["weeks"].to_numpy(dtype=float)), -np.log(songs["peak"].to_numpy(dtype=float))
    tiers = pd.cut(songs["peak"], [0, 1, 10, 40, 100], labels=["#1", "2-10", "11-40", "41-100"])
    songs["q"] = songs.groupby("debut_year")["weeks"].rank(pct=True)
    long_lived = pd.cut(songs["q"], [0, .25, .75, 1], labels=["short", "typical", "long"], include_lowest=True)
    adjusted = year_adjusted(songs, features, "debut_year")
    contenders = adjusted.loc[adjusted["peak"] <= 40]
    long_set, short_set = contenders.loc[contenders["q"] >= .75], contenders.loc[contenders["q"] <= .25]
    sustained = [{"feature": f, **ss.compare_two(long_set[f], short_set[f])} for f in features]
    for row, q in zip(sustained, ss.bh_fdr([r["p"] for r in sustained])):
        row["q_fdr"] = float(q)
    top10 = [{"feature": f, **ss.compare_two(adjusted.loc[adjusted["peak"] <= 10, f], adjusted.loc[adjusted["peak"] > 40, f])} for f in features]
    for row, q in zip(top10, ss.bh_fdr([r["p"] for r in top10])):
        row["q_fdr"] = float(q)
    billboard = {
        "n": len(songs), "peak_tier_by_longevity": pd.crosstab(tiers, long_lived).to_numpy(),
        "tiers": ["#1", "2-10", "11-40", "41-100"], "longevity": ["short", "typical", "long"],
        "chi_square_peak_longevity": ss.chi_square(pd.crosstab(tiers, long_lived).to_numpy()),
        "median_weeks_by_tier": songs.groupby(tiers, observed=True)["weeks"].median().to_numpy(),
        "weeks_model": _effects(weeks, x, features, performers, [years]), "peak_model": _effects(peak, x, features, performers, [years]),
        "sustained_vs_short_among_top40": sustained, "top10_vs_below40": top10,
        "cv": {"audio_plus_year": ss.cv_r2(weeks, x, [years], groups=performers),
               "year_only": ss.cv_r2(weeks, effects=[years], groups=performers),
               "audio_plus_year_plus_known_performer": ss.cv_r2(weeks, x, [years, performers], shrink=[0, 2.0])},
        "weeks_describe": ss.describe(songs["weeks"]), "weeks_hist": np.histogram(songs["weeks"], bins=np.arange(0, 66, 2))[0],
        "peak_vs_weeks_spearman": float(st.spearmanr(songs["peak"], songs["weeks"]).statistic),
    }

    table = global_song_table()
    audio = song_audio()
    joined = table.join(audio[features], on="song_key")
    coverage = float(joined[features[0]].notna().mean())
    streams_cover = float(joined.loc[joined[features[0]].notna(), "streams"].sum() / joined["streams"].sum())
    usable = joined.dropna(subset=[features[0]])
    usable = usable.loc[~usable["left_truncated"] & ~usable["seasonal"]].copy()
    finished = usable.loc[~usable["censored"]].copy()
    z = _standard(finished, features)
    debut, artists = pd.factorize(finished["debut_year"])[0], pd.factorize(finished["artist_key"])[0]
    outcomes = {"log_total_streams": np.log(finished["streams"]), "log_chart_days": np.log(finished["days"]),
                "minus_log_peak_rank": -np.log(finished["peak"]), "front_load_share": finished["front_load"]}
    models = {k: _effects(v.to_numpy(dtype=float), z, features, artists, [debut]) for k, v in outcomes.items()}
    cv = {
        "audio_plus_year": ss.cv_r2(outcomes["log_total_streams"].to_numpy(dtype=float), z, [debut], groups=artists),
        "year_only": ss.cv_r2(outcomes["log_total_streams"].to_numpy(dtype=float), effects=[debut], groups=artists),
        "audio_plus_year_plus_known_artist": ss.cv_r2(outcomes["log_total_streams"].to_numpy(dtype=float), z, [debut, artists], shrink=[0, 2.0]),
    }
    outcome_columns = ["streams", "days", "peak", "front_load", "half_life"]
    rank_matrix = finished[outcome_columns].rank().corr().to_numpy()
    survival, tests = {}, []
    for feature in ["energy", "valence", "danceability", "acousticness", "loudness", "tempo", "speechiness", "duration_min"]:
        cuts = usable[feature].quantile([1 / 3, 2 / 3]).to_numpy()
        lo, hi = usable.loc[usable[feature] <= cuts[0]], usable.loc[usable[feature] >= cuts[1]]
        pair = [(g["span"].to_numpy(), (~g["censored"]).to_numpy()) for g in (lo, hi)]
        test = ss.logrank(pair)
        low_curve, high_curve = ss.kaplan_meier(*pair[0]), ss.kaplan_meier(*pair[1])
        tests.append(test["p"])
        thin = lambda c: {"t": c["t"][:: max(1, len(c["t"]) // 60)], "s": c["s"][:: max(1, len(c["t"]) // 60)], "median": c["median"]}
        survival[feature] = {"low_tertile": thin(low_curve), "high_tertile": thin(high_curve), **test, "cuts": cuts}
    for feature, q in zip(survival, ss.bh_fdr(tests)):
        survival[feature]["q_fdr"] = float(q)
    contenders = year_adjusted(finished, features, "debut_year").loc[finished["peak"] <= 50]
    q1, q3 = contenders["front_load"].quantile([.25, .75])
    flash, slow = contenders.loc[contenders["front_load"] >= q3], contenders.loc[contenders["front_load"] <= q1]
    flash_tests = [{"feature": f, **ss.compare_two(flash[f], slow[f])} for f in features]
    for row, q in zip(flash_tests, ss.bh_fdr([r["p"] for r in flash_tests])):
        row["q_fdr"] = float(q)
    labels = read("chart17_songs", ["song_key", "name", "artist"]).set_index("song_key")
    longest = finished.sort_values("days", ascending=False).head(10).join(labels, on="song_key", rsuffix="_label")
    biggest_flash = contenders.sort_values(["front_load"], ascending=False).head(10).join(labels, on="song_key", rsuffix="_label")
    list_rows = lambda f: [{"name": r["name"], "artist": r["artist"], "days": int(r["days"]), "peak": int(r["peak"]),
                            "streams": float(r["streams"]), "front_load": float(r["front_load"])} for _, r in f.iterrows()]
    spotify = {
        "songs_on_chart": len(table), "audio_match_share": coverage, "stream_share_with_audio": streams_cover,
        "excluded": {"debut_in_first_14_days": int(table["left_truncated"].sum()), "seasonal": int(table["seasonal"].sum()),
                     "still_charting_at_end": int(table["censored"].sum())},
        "usable_for_survival": len(usable), "finished_runs": len(finished), "models": models, "cv": cv,
        "outcome_columns": outcome_columns, "outcome_spearman": rank_matrix, "survival": survival,
        "flash_vs_slow_burn": flash_tests, "flash_threshold": float(q3), "slow_threshold": float(q1),
        "median_days": float(finished["days"].median()), "median_half_life": float(finished["half_life"].median()),
        "median_front_load": float(finished["front_load"].median()), "longest": list_rows(longest), "most_front_loaded": list_rows(biggest_flash),
        "scatter": finished.sample(min(1500, len(finished)), random_state=RANDOM_SEED)[["peak", "days", "streams", "front_load"]].round(3).to_numpy(),
        "by_peak_band": [{"band": name, "n": len(g), "median_days": float(g["days"].median()), "median_streams": float(g["streams"].median())}
                         for name, g in finished.groupby(pd.cut(finished["peak"], [0, 1, 10, 50, 100, 200], labels=["#1", "2-10", "11-50", "51-100", "101-200"]), observed=True)],
    }
    table.to_parquet(PROCESSED_DIR / "global_songs_2017_2021.parquet", index=False)
    return save("hits", {"features": features, "billboard": billboard, "spotify": spotify})


# ---------------------------------------------------------------- good, liked, popular


def good() -> dict:
    tags = read("tags", ["track_id", "tags"])
    tag_lists = [{t.strip() for t in text.lower().split(",") if t.strip()} for text in tags["tags"]]
    tags["affection"] = np.fromiter((bool(s & AFFECTION_TAGS) for s in tag_lists), bool, len(tag_lists))
    tags["n_tags"] = np.fromiter((len(s) for s in tag_lists), int, len(tag_lists))
    del tag_lists
    pop = popularity_universe()
    frame = pop.merge(tags.drop(columns="tags"), on="track_id")
    del tags
    rich = frame.loc[frame["n_tags"] >= 5].copy()
    rich["tag_band"] = pd.cut(rich["n_tags"], [4, 7, 10, 15, 25, 1000], labels=False)
    z = _standard(rich, DISPLAY_FEATURES)
    effects = [pd.factorize(rich["tag_band"])[0], pd.factorize(rich["year"])[0], pd.factorize(rich["genre"])[0]]
    clusters = pd.factorize(rich["artist"])[0]
    y = rich["affection"].to_numpy(dtype=float) * 100
    model = _effects(y, z, DISPLAY_FEATURES, clusters, effects)
    with_pop = _effects(y, np.column_stack([z, (rich["pop_pct"] - rich["pop_pct"].mean()) / rich["pop_pct"].std()]),
                        DISPLAY_FEATURES + ["popularity_percentile"], clusters, effects)
    deciles = pd.qcut(rich["pop_pct"].rank(method="first"), 10, labels=False)
    table = rich.groupby([rich["tag_band"], deciles])["affection"].mean().unstack()
    bins = rich.groupby(deciles)["affection"].agg(["sum", "size"])
    low, high = wilson(bins["sum"].to_numpy(), bins["size"].to_numpy())
    popularity = json.loads((RESULTS_DIR / "popularity.json").read_text())
    hit_results = json.loads((RESULTS_DIR / "hits.json").read_text())
    pop_betas = {c["term"]: c for c in popularity["regression"]["coefficients"]}
    days = {c["term"]: c for c in hit_results["spotify"]["models"]["log_chart_days"]["coefficients"]}
    streams = {c["term"]: c for c in hit_results["spotify"]["models"]["log_total_streams"]["coefficients"]}
    liked = {c["term"]: c for c in model["coefficients"]}
    matrix = [{"feature": f, "popularity_pct_points": pop_betas[f]["beta"], "popularity_q": pop_betas[f]["q_fdr"],
               "chart_days_pct": float(np.expm1(days[f]["beta"]) * 100), "chart_days_q": days[f]["q_fdr"],
               "streams_pct": float(np.expm1(streams[f]["beta"]) * 100), "streams_q": streams[f]["q_fdr"],
               "affection_points": liked[f]["beta"], "affection_q": liked[f]["q_fdr"]} for f in DISPLAY_FEATURES]
    rho = st.spearmanr(rich["pop_pct"], rich["affection"])
    return save("good", {
        "note": "No dataset measures whether music is good. Three observable proxies are compared.",
        "affection_tags": sorted(AFFECTION_TAGS), "n_matched": len(frame), "n_with_5_plus_tags": len(rich),
        "affection_rate": float(rich["affection"].mean()), "by_popularity_decile": {"rate": bins["sum"] / bins["size"], "low": low, "high": high},
        "by_tag_band_and_decile": table.to_numpy(), "tag_bands": ["5-7", "8-10", "11-15", "16-25", "26+"],
        "popularity_affection_spearman": float(rho.statistic), "popularity_affection_p": float(rho.pvalue),
        "affection_model": model, "affection_model_with_popularity": with_pop, "matrix": matrix,
    })


# ---------------------------------------------------------------- dashboard


def dashboard() -> dict:
    audio_columns = AUDIO_FEATURES + ["duration_ms"]
    early = read("charts_2017_2021", ["song_key", "country", "year", "days_top50", "best_rank", "streams"])
    late = read("charts_2023_2025", ["song_key", "country", "date", "rank", "name", "artist", "track_id", "popularity",
                                     "album_release_date", "audio_valid"] + audio_columns)
    late = late.loc[late["audio_valid"]].copy()
    late["year"] = late["date"].dt.year
    e_days = early.groupby("song_key")["days_top50"].sum().sort_values(ascending=False)
    l_days = late.groupby("song_key").size().sort_values(ascending=False)
    names = read("chart17_songs", ["song_key", "name", "artist", "feature_id"]).dropna(subset=["feature_id"])
    names = names.set_index("song_key")
    late_first = late.drop_duplicates("song_key").set_index("song_key")
    # The 1.2M catalogue has no stated licence, so none of its rows are published here.
    tracks = read("tracks", ["track_id", "song_key", "audio_source"] + audio_columns)
    tracks = tracks.loc[tracks["audio_source"] != "spotify_catalogue"].drop(columns="audio_source")
    genre_rows = genre_universe()
    genre_of = genre_rows.drop_duplicates("track_id").set_index("track_id")["genre"]
    pop = read("popularity_1m", ["track_id", "song_key", "name", "artist", "genre", "year", "popularity"])
    bill = read("billboard_songs", ["song_id", "song", "performer", "song_key", "track_id", "debut_year", "weeks", "peak"]).dropna(subset=["track_id"])
    records: dict[str, dict] = {}

    def entry(key, track_id, name, artist):
        item = records.setdefault(key, {"key": key, "track_id": track_id, "name": name, "artist": artist})
        return item

    for key in e_days.head(3500).index:
        if key in names.index:
            row = names.loc[key]
            entry(key, row["feature_id"], row["name"], str(row["artist"]).split(",")[0])
    for key in l_days.head(3500).index:
        row = late_first.loc[key]
        entry(key, row["track_id"], row["name"], row["artist"])
    for row in bill.sort_values("weeks", ascending=False).head(2800).itertuples():
        if isinstance(row.song_key, str):
            item = entry(row.song_key, row.track_id, row.song, row.performer)
            item.update(bb_weeks=int(row.weeks), bb_peak=int(row.peak), first_year=int(row.debut_year))
    top_genre = genre_rows.sort_values("popularity", ascending=False).groupby("genre").head(15).drop_duplicates("track_id")
    genre_keys = read("genre_tracks", ["track_id", "song_key"]).drop_duplicates("track_id").set_index("track_id")["song_key"]
    for row in top_genre.itertuples():
        key = genre_keys.get(row.track_id)
        if isinstance(key, str):
            item = entry(key, row.track_id, row.name, row.artist)
            item.setdefault("genre", row.genre)
            item.setdefault("pop", int(row.popularity))
    table = pd.DataFrame(records.values())
    features = tracks.drop(columns="song_key").set_index("track_id")
    table = table.join(features, on="track_id").dropna(subset=["energy"])
    table["genre"] = table["genre"].where(table["genre"].notna(), table["track_id"].map(genre_of))
    pop_by_id = pop.drop_duplicates("track_id").set_index("track_id")
    table["genre"] = table["genre"].where(table["genre"].notna(), table["track_id"].map(pop_by_id["genre"]))
    pop_by_key = pop.dropna(subset=["song_key"]).sort_values("popularity", ascending=False).drop_duplicates("song_key").set_index("song_key")
    table["genre"] = table["genre"].where(table["genre"].notna(), table["key"].map(pop_by_key["genre"]))
    release = pd.to_datetime(late_first["album_release_date"], errors="coerce").dt.year
    candidates = pd.DataFrame({
        "a": table.get("first_year"), "c": table["track_id"].map(pop_by_id["year"]),
        "d": table["key"].map(pop_by_key["year"]), "e": table["key"].map(release),
    }).astype(float)
    table["year"] = candidates.min(axis=1)
    chart_early = early.loc[early["country"] != "GLOBAL"]
    chart_early = chart_early.loc[(chart_early["days_top50"] > 0) & chart_early["song_key"].isin(set(table["key"]))]
    chart_late = late.loc[(late["country"] != "GLOBAL") & late["song_key"].isin(set(table["key"]))]
    pairs = pd.concat([chart_early[["song_key", "country", "days_top50"]].rename(columns={"days_top50": "days"}),
                       chart_late.groupby(["song_key", "country"], as_index=False).size().rename(columns={"size": "days"})])
    pairs = pairs.groupby(["song_key", "country"], as_index=False)["days"].sum()
    pairs = pairs.loc[pairs["days"] > 0]
    countries = sorted(pairs["country"].unique())
    position = {c: i for i, c in enumerate(countries)}
    country_lists = pairs.groupby("song_key")["country"].agg(lambda s: sorted(position[c] for c in s))
    total_days = pairs.groupby("song_key")["days"].sum()
    streams = early.loc[early["country"] == "GLOBAL"].groupby("song_key")["streams"].sum()
    best = pd.concat([early.groupby("song_key")["best_rank"].min(), late.groupby("song_key")["rank"].min().rename("best_rank")]).groupby(level=0).min()

    model = json.loads((PROCESSED_DIR / "archetype_model.json").read_text())
    reference = genre_rows.drop_duplicates("track_id")[AUDIO_FEATURES].to_numpy(dtype=float).copy()
    log_columns = [i for i, f in enumerate(AUDIO_FEATURES) if f in {"speechiness", "instrumentalness", "liveness", "acousticness"}]
    reference[:, log_columns] = np.log10(reference[:, log_columns] + 0.01)
    mean, sd = reference.mean(axis=0), reference.std(axis=0)
    matrix = table[AUDIO_FEATURES].to_numpy(dtype=float).copy()
    matrix[:, log_columns] = np.log10(matrix[:, log_columns] + 0.01)
    archetype = ss._assign((matrix - mean) / sd, np.array(model["centres"]))[0]
    rows = []
    genres = sorted(g for g in table["genre"].dropna().unique())
    genre_index = {g: i for i, g in enumerate(genres)}
    for i, r in enumerate(table.itertuples()):
        rows.append([
            r.name, r.artist, None if np.isnan(r.year) else int(r.year), genre_index.get(r.genre, -1), int(archetype[i]),
            round(r.danceability, 2), round(r.energy, 2), round(r.valence, 2), round(r.acousticness, 2),
            round(r.speechiness, 2), round(r.instrumentalness, 2), round(r.liveness, 2), round(r.loudness, 1),
            int(round(r.tempo)), round(r.duration_ms / 60000, 2),
            None if pd.isna(getattr(r, "bb_weeks", np.nan)) else int(r.bb_weeks),
            None if pd.isna(getattr(r, "bb_peak", np.nan)) else int(r.bb_peak),
            int(total_days.get(r.key, 0)), None if r.key not in best.index else int(best[r.key]),
            None if r.key not in streams.index else int(streams[r.key]),
            country_lists.get(r.key, []), r.track_id,
        ])
    payload = {
        "columns": ["name", "artist", "year", "genre", "archetype", "danceability", "energy", "valence", "acousticness",
                    "speechiness", "instrumentalness", "liveness", "loudness", "tempo", "duration", "hot100_weeks", "hot100_peak",
                    "top50_days", "best_rank", "global_streams", "countries", "id"],
        "rows": rows, "genres": genres, "countries": countries, "country_names": [COUNTRY_NAMES[c] for c in countries],
        "archetypes": model["names"],
    }
    return save("dashboard", payload)
