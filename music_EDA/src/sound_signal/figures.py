"""Static figures for the notebook and README, drawn from the saved result files."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from .config import FIGURES_DIR  # noqa: E402
from .findings import NAMES, load_results  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#184f95", "#5598e7", "#f0efec", "#ec835a", "#b03030"])
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 140, "savefig.bbox": "tight", "font.size": 9, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": MUTED, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def _heat(ax, data, xlabels, ylabels, cmap, norm=None, fmt=None, size=7):
    ax.grid(False)
    image = ax.imshow(np.array(data, dtype=float), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=45, ha="right", fontsize=size)
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=size)
    if fmt:
        for (i, j), value in np.ndenumerate(np.array(data, dtype=float)):
            if np.isfinite(value):
                ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=size - .5,
                        color="white" if abs(value) > .55 * np.nanmax(np.abs(data)) else INK)
    return image


def emotion_map(R):
    e, tags = R["emotion"], R["emotion"]["tags"]
    fig, ax = plt.subplots(figsize=(7, 5.6))
    ax.grid(False)
    density = np.array(e["map"]["density"], dtype=float)
    image = ax.imshow(density * 100, origin="lower", extent=[0, 1, 0, 1], cmap=SEQUENTIAL, aspect="auto")
    fig.colorbar(image, label="share of tracks per cell (%)", shrink=.8)
    ax.axhline(.5, color=MUTED, lw=.6), ax.axvline(.5, color=MUTED, lw=.6)
    for colour, mood in zip((ORANGE, RED, VIOLET, AQUA), tags["moods"]):
        row = tags["summary"][mood]
        ax.errorbar(row["valence_median"], row["energy_median"], fmt="o", ms=8, color=colour, mec="white",
                    xerr=[[row["valence_median"] - row["valence_q1"]], [row["valence_q3"] - row["valence_median"]]],
                    yerr=[[row["energy_median"] - row["energy_q1"]], [row["energy_q3"] - row["energy_median"]]], lw=1, label=f"tagged {mood}")
    ax.set(xlabel="valence (measured brightness)", ylabel="energy", title="Musical emotion map, with Last.fm mood tags (median and IQR)")
    ax.legend(loc="lower right")
    return fig


def correlation(R):
    e = R["emotion"]
    names = e["matrix_features"]
    order = [names.index(f) for f in e["clustered_order"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    for ax, key, title in zip(axes, ("pearson", "spearman"), ("Pearson", "Spearman")):
        data = np.array(e[key])[order][:, order]
        labels = [NAMES[names[i]] for i in order]
        image = _heat(ax, data, labels, labels, DIVERGING, TwoSlopeNorm(0, -1, 1), ".2f")
        ax.set_title(f"{title} correlation")
    fig.colorbar(image, ax=axes, shrink=.8)
    return fig


def mood_agreement(R):
    t = R["emotion"]["tags"]
    a = t["agreement"]["catalogue_medians"]
    table = np.array(a["table"], dtype=float)
    share = table / table.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 4))
    _heat(ax, share, [f"{m} quadrant" for m in t["moods"]], [f"tagged {m}" for m in t["moods"]], SEQUENTIAL, fmt=".0%", size=9)
    ax.set_title(f"Tagged mood against measured quadrant (κ = {a['kappa']:.2f})")
    return fig


def popularity_distribution(R):
    o = R["overview"]["popularity"]
    edges = np.array(o["bin_edges"])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.stairs(np.array(o["sample_1m"]) * 100, edges, color=BLUE, lw=1.6, label="1M-track sample, 2000-2023")
    ax.stairs(np.array(o["sample_genre"]) * 100, edges, color=ORANGE, lw=1.6, label="genre sample")
    ax.set(xlabel="Spotify popularity (0-100)", ylabel="share of tracks (%)", title="Popularity distribution")
    ax.legend()
    return fig


def _forest(ax, rows, label="Cliff's δ"):
    rows = sorted(rows, key=lambda r: r["cliffs_delta"])
    y = np.arange(len(rows))
    ax.errorbar([r["cliffs_delta"] for r in rows], y, fmt="o", color=BLUE, ms=5, lw=1.2,
                xerr=[[r["cliffs_delta"] - r["delta_ci"][0] for r in rows], [r["delta_ci"][1] - r["cliffs_delta"] for r in rows]])
    ax.axvline(0, color=MUTED, lw=.8)
    ax.axvspan(-.147, .147, color="#e1e0d9", alpha=.45, lw=0)
    ax.set_yticks(y, [NAMES[r["feature"]] for r in rows])
    ax.set_xlabel(label + " (grey band: negligible)")


def hit_profile(R):
    p = R["popularity"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharex=True)
    _forest(axes[0], [{"feature": f, **c} for f, c in p["comparisons"].items()])
    axes[0].set_title("Hits vs low popularity, same year")
    _forest(axes[1], p["within_genre_year_comparisons"])
    axes[1].set_title("Hits vs low popularity, same genre and year")
    return fig


def ladder(R):
    p = R["popularity"]
    a, b = p["ladder"], p["ladder_independent_genre_sample"]
    labels = ["Audio, linear", "Audio, flexible", "Genre label*", "Audio + genre*", "Artist's other tracks", "All combined"]
    first = [a["audio_linear"], a["audio_flexible_deciles"], a["genre_only"], a["audio_flexible_plus_genre"], a["artist_only"], a["all_with_artist"]]
    second = [b["audio_linear"], None, b["genre_only"], b["audio_plus_genre"], b["artist_only"], b["audio_genre_artist"]]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    y = np.arange(len(labels))[::-1]
    ax.barh(y + .19, [r["r2"] * 100 for r in first], .34, color=BLUE, label="1M-track sample")
    ax.barh(y - .19, [0 if r is None else r["r2"] * 100 for r in second], .34, color=ORANGE, label="independent genre sample")
    ax.set_yticks(y, labels)
    ax.set(xlabel="out-of-sample R² (%)", title="What predicts popularity? (*labels reflect how samples were collected)")
    ax.legend(loc="lower right")
    return fig


def evolution(R):
    v = R["evolution"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharex=True)
    units = {"tempo": "BPM", "loudness": "dB", "duration_min": "minutes"}
    for ax, f in zip(axes.flat, ["loudness", "energy", "danceability", "valence", "duration_min", "tempo"]):
        s = v["series"][f]
        ax.fill_between(v["years"], s["chart_low"], s["chart_high"], color=BLUE, alpha=.18, lw=0)
        ax.plot(v["years"], s["chart"], color=BLUE, lw=1.8, label="Hot 100 (by chart year)")
        ax.plot(v["years"], s["catalogue"], color=ORANGE, lw=1.4, label="Catalogue (by release year)")
        for c in v["trends"][f]["chart"]["change_points"]:
            ax.axvline(c["year"], color=GRID, lw=1, zorder=0)
        ax.set_title(f"{NAMES[f]}" + (f" ({units[f]})" if f in units else ""))
    axes[0, 0].legend(fontsize=7.5)
    return fig


def decades(R):
    v = R["evolution"]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.6))
    for ax, f in zip(axes, ["loudness", "valence", "acousticness", "duration_min"]):
        names = list(v["decades"][f]["summary"])
        stats = [{"label": n[:-1][-3:] + "s", "med": s["median"], "q1": s["q1"], "q3": s["q3"], "whislo": s["p05"], "whishi": s["p95"], "fliers": []}
                 for n, s in v["decades"][f]["summary"].items()]
        ax.bxp(stats, showfliers=False, patch_artist=True, widths=.6, boxprops={"facecolor": "#cde2fb", "edgecolor": BLUE},
               medianprops={"color": "#0d366b", "lw": 1.6}, whiskerprops={"color": MUTED}, capprops={"color": MUTED})
        ax.set_title(f"{NAMES[f]} (ε²={v['decades'][f]['epsilon_squared']:.2f})")
        ax.tick_params(axis="x", labelsize=7.5)
    return fig


def genre_violins(R):
    g = R["genre"]
    focal = g["focal_genres"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), sharey=True)
    for ax, f in zip(axes, ["energy", "valence", "acousticness"]):
        order = sorted(focal, key=lambda x: g["summaries"][x]["energy"]["box"][2])
        for i, name in enumerate(order):
            s = g["summaries"][name][f]
            h = np.array(s["hist"], dtype=float)
            x = np.linspace(0, 1, len(h))
            ax.fill_between(x, i - h / h.max() * .42, i + h / h.max() * .42, color="#86b6ef", lw=0)
            ax.plot([s["box"][1], s["box"][3]], [i, i], color="#0d366b", lw=2)
            ax.plot(s["box"][2], i, "o", color="white", mec="#0d366b", ms=4)
        ax.set_yticks(range(len(order)), order)
        ax.set_title(NAMES[f])
    return fig


def genre_similarity(R):
    s = R["genre"]["similarity"]
    fig, ax = plt.subplots(figsize=(10, 9))
    image = _heat(ax, s["cosine"], s["order"], s["order"], DIVERGING, TwoSlopeNorm(0, -1, 1), size=4.4)
    ax.set_title("Genre similarity (cosine of audio centroids, Ward order)")
    fig.colorbar(image, shrink=.6)
    return fig


def archetypes(R):
    a = R["genre"]["archetypes"]
    colours = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.15], "wspace": .42})
    x, y, c = (np.array(a["pca"][k]) for k in ("x", "y", "cluster"))
    for i, item in enumerate(a["items"]):
        axes[0].scatter(x[c == i], y[c == i], s=5, alpha=.5, color=colours[i], lw=0, label=item["name"])
    axes[0].set(xlabel=f"PC1 ({a['pca']['explained'][0]:.0%})", ylabel=f"PC2 ({a['pca']['explained'][1]:.0%})", title="Archetypes on the first two principal components")
    axes[0].legend(markerscale=2.5, fontsize=7, loc="upper right")
    z = [[i["z_centre"][f] for f in a["features"]] for i in a["items"]]
    image = _heat(axes[1], z, [NAMES[f] for f in a["features"]], [i["name"] for i in a["items"]], DIVERGING, TwoSlopeNorm(0, -2.4, 2.4), ".1f", 7.5)
    axes[1].set_title("Archetype profiles (SD from average)")
    fig.colorbar(image, ax=axes[1], shrink=.8)
    return fig


def sad_surface(R):
    s = np.array(R["sadness"]["surface"]["lift_adjusted"], dtype=float).T
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.grid(False)
    image = ax.imshow(s, origin="lower", extent=[0, 1, 0, 1], cmap=DIVERGING, norm=TwoSlopeNorm(1, .4, 1.6), aspect="auto")
    fig.colorbar(image, label="hits relative to genre-year expectation")
    ax.set(xlabel="valence", ylabel="energy", title="Hit likelihood across the emotion map")
    return fig


def country_heat(R):
    geo = R["geography"]
    p = geo["periods"]["2023-2025"]
    index = {c: i for i, c in enumerate(p["countries"])}
    rows = [index[c] for c in p["order"]]
    use = [geo["features"].index(f) for f in ("danceability", "energy", "loudness", "speechiness", "acousticness", "valence", "tempo", "duration_min")]
    data = np.array(p["z_vs_global"])[rows][:, use]
    fig, ax = plt.subplots(figsize=(7.5, 11))
    image = _heat(ax, data, [NAMES[geo["features"][i]] for i in use], [p["names"][i] for i in rows], DIVERGING, TwoSlopeNorm(0, -1.2, 1.2), size=6.5)
    ax.set_title("Country charts vs the Global chart, 2023-2025 (SD)")
    fig.colorbar(image, shrink=.4)
    return fig


def convergence(R):
    c = R["geography"]["convergence"]
    late = np.array(c["is_late"], bool)
    x = np.arange(len(c["quarters"])) + late * 3
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for mask in (~late, late):
        axes[0].fill_between(x[mask], np.array(c["jaccard_low"])[mask] * 100, np.array(c["jaccard_high"])[mask] * 100, color=BLUE, alpha=.18, lw=0)
        axes[0].plot(x[mask], np.array(c["jaccard"])[mask] * 100, color=BLUE, lw=1.8, label="all qualifying songs" if mask[0] else None)
        axes[0].plot(x[mask], np.array(c["core25_jaccard"])[mask] * 100, color=ORANGE, lw=1.5, label="core 25 songs" if mask[0] else None)
        axes[1].plot(x[mask], np.array(c["audio_distance"])[mask], color=BLUE, lw=1.8, label="observed" if mask[0] else None)
        axes[1].plot(x[mask], np.array(c["audio_noise_floor"])[mask], color=MUTED, lw=1.2, label="random-assignment baseline" if mask[0] else None)
    ticks = [i for i, q in enumerate(c["quarters"]) if q % 10 == 1]
    for ax in axes:
        ax.set_xticks(x[ticks], [str(c["quarters"][i] // 10) for i in ticks])
        ax.legend()
    axes[0].set(title="Songs shared by two countries' charts", ylabel="Jaccard overlap (%)")
    axes[1].set(title="Distance between countries' sound profiles", ylabel="mean distance (SD units)")
    return fig


def longevity(R):
    s = R["hits"]["spotify"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for ax, f in zip(axes[:2], ("speechiness", "valence")):
        for key, colour, label in (("low_tertile", BLUE, "lowest third"), ("high_tertile", ORANGE, "highest third")):
            ax.step(s["survival"][f][key]["t"], s["survival"][f][key]["s"], where="post", color=colour, label=label, lw=1.6)
        ax.set_xscale("log")
        ax.set(title=f"Chart survival by {NAMES[f]}", xlabel="days after debut (log scale)", ylabel="share still charting")
        ax.legend()
    bands = s["by_peak_band"]
    axes[2].barh([b["band"] for b in bands][::-1], [b["median_days"] for b in bands][::-1], color=BLUE)
    axes[2].set(title="Median chart days by peak position", xlabel="days")
    return fig


def good_matrix(R):
    m = R["good"]["matrix"]
    keys = ["popularity_pct_points", "chart_days_pct", "affection_points"]
    scale = [max(abs(r[k]) for r in m) for k in keys]
    data = [[r[k] / s for k, s in zip(keys, scale)] for r in m]
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    image = _heat(ax, data, ["popularity", "chart days", "affectionate tags"], [NAMES[r["feature"]] for r in m], DIVERGING, TwoSlopeNorm(0, -1, 1), ".2f", 8.5)
    ax.set_title("Association per SD (scaled within each outcome)")
    fig.colorbar(image, shrink=.8)
    return fig


ALL = {
    "01_emotion_map": emotion_map, "02_correlation_heatmap": correlation, "03_mood_agreement": mood_agreement,
    "04_popularity_distribution": popularity_distribution, "05_hit_profile": hit_profile, "06_prediction_ladder": ladder,
    "07_evolution_timeline": evolution, "08_decade_distributions": decades, "09_genre_violins": genre_violins,
    "10_genre_similarity": genre_similarity, "11_archetypes_pca": archetypes, "12_sadness_surface": sad_surface,
    "13_country_comparison": country_heat, "14_convergence": convergence, "15_chart_longevity": longevity, "16_good_liked_popular": good_matrix,
}


def save_all() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    R = load_results()
    for name, function in ALL.items():
        fig = function(R)
        fig.savefig(FIGURES_DIR / f"{name}.png")
        plt.close(fig)
    print(f"saved {len(ALL)} figures to {FIGURES_DIR}")
