"""Build the SOUND / SIGNAL publication as one self-contained HTML file.

Usage: python scripts/07_build_site.py [--fragment path]
The optional fragment omits the document wrapper, for hosts that add their own.
"""
from __future__ import annotations

import html
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sound_signal.common import COUNTRY_NAMES  # noqa: E402
from sound_signal.config import RESULTS_DIR, SITE_DIR  # noqa: E402
from sound_signal.findings import build, check, coef, load_results, n0, pc, pval  # noqa: E402

R = load_results()
F = build(R)
check(F)
E, P, V, G, S, GEO, H, GOOD, Q, O = (R[k] for k in ("emotion", "popularity", "evolution", "genre", "sadness", "geography", "hits", "good", "data_quality", "overview"))
DASH = json.loads((RESULTS_DIR / "dashboard.json").read_text(encoding="utf8"))
MANIFEST = json.loads((ROOT / "docs" / "raw_manifest.json").read_text(encoding="utf8"))
MB_PATH = RESULTS_DIR / "musicbrainz_quality.json"
MB = json.loads(MB_PATH.read_text(encoding="utf8")) if MB_PATH.exists() else None
GENRE_FEATURES = ["energy", "valence", "danceability", "acousticness", "loudness", "tempo", "instrumentalness"]


def round_deep(value, digits=4):
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, digits)
    if isinstance(value, list):
        return [round_deep(v, digits) for v in value]
    if isinstance(value, dict):
        return {k: round_deep(v, digits) for k, v in value.items()}
    return value


def percent_row(c):
    return [(math.exp(x) - 1) * 100 for x in (c["beta"], c["ci"][0], c["ci"][1])]


def payload():
    tags = E["tags"]
    names = E["matrix_features"]
    order = [names.index(n) for n in E["clustered_order"]]
    matrix = lambda key: [[E[key][i][j] for j in order] for i in order]
    within = {r["feature"]: r for r in P["within_genre_year_comparisons"]}
    a, b = P["ladder"], P["ladder_independent_genre_sample"]
    ladder = [("Audio features, straight-line", a["audio_linear"]["r2"], b["audio_linear"]["r2"]), ("Audio features, flexible", a["audio_flexible_deciles"]["r2"], None),
              ("Genre label (see note)", a["genre_only"]["r2"], b["genre_only"]["r2"]), ("Audio + genre", a["audio_flexible_plus_genre"]["r2"], b["audio_plus_genre"]["r2"]),
              ("The artist's other tracks", a["artist_only"]["r2"], b["artist_only"]["r2"]), ("Everything combined", a["all_with_artist"]["r2"], b["audio_genre_artist"]["r2"])]
    trends = {f: {"pd": t["chart"]["per_decade"], "lo": t["chart"]["ci"][0] * 10, "hi": t["chart"]["ci"][1] * 10,
                  "cps": [c["year"] for c in t["chart"]["change_points"]], "anom": [x["year"] for x in t["chart"]["anomalies"]]} for f, t in V["trends"].items()}
    decades = {f: {"s": d["summary"], "range": d["range"], "eps": d["epsilon_squared"]} for f, d in V["decades"].items()}
    summaries = {g: {f: {"hist": G["summaries"][g][f]["hist"], "box": G["summaries"][g][f]["box"]} for f in GENRE_FEATURES} for g in G["genres"]}
    ranges = {"energy": [0, 1], "valence": [0, 1], "danceability": [0, 1], "acousticness": [0, 1], "instrumentalness": [0, 1], "loudness": [-40, 5], "tempo": [40, 220]}
    arch = G["archetypes"]
    items = [{"name": i["name"], "share": i["share"], "pop": i["median_popularity"], "top_genres": i["top_genres"], "examples": i["examples"], "z": i["z_centre"]} for i in arch["items"]]
    rates = [{"band": S["bands"][k], **{x: S["rates"][k][x] for x in ("n", "hits", "hit_rate")}, "rr": S["rates"][k]["risk_ratio_vs_high"]} for k in ("low", "mid", "high")]
    year_data = S["hit_valence_by_year"]
    conv = GEO["convergence"]
    x, out = [], {k: [] for k in ("j", "jl", "jh", "c", "d", "n", "q")}
    sources = {"j": "jaccard", "jl": "jaccard_low", "jh": "jaccard_high", "c": "core25_jaccard", "d": "audio_distance", "n": "audio_noise_floor"}
    for i, quarter in enumerate(conv["quarters"]):
        if i and conv["is_late"][i] and not conv["is_late"][i - 1]:
            x.append(2022.5)
            for k in sources:
                out[k].append(None)
            out["q"].append("")
        year, q = quarter // 10, quarter % 10
        x.append(year + (q - 1) / 4)
        for k, source in sources.items():
            out[k].append(conv[source][i])
        out["q"].append(f"{year} Q{q}")
    periods = {p: {k: v[k] for k in ("countries", "names", "order")} | {"z": v["z_vs_global"], "profile": v["profile"]} for p, v in GEO["periods"].items()}
    reach = GEO["reach"]
    origin = None
    if GEO.get("origin"):
        origin = {p: [{"name": r["name"], "local": r["local_share"], "us": r["us_share"], "known": r["known_share"]} for r in v["countries"]] for p, v in GEO["origin"]["periods"].items()}
    years = list(range(2017, 2026))
    country_rows, seen = {}, set()
    for row in GEO["country_year"]["rows"]:
        country_rows.setdefault(row[1], {})[row[0]] = row[2:]
        seen.add(row[1])
    sp, bb = H["spotify"], H["billboard"]
    models = [{"f": f, "spot": percent_row(coef(sp["models"]["log_chart_days"], f)), "sq": pval(coef(sp["models"]["log_chart_days"], f)["q_fdr"]),
               "bb": percent_row(coef(bb["weeks_model"], f)), "bq": pval(coef(bb["weeks_model"], f)["q_fdr"])} for f in H["features"]]
    surv = {f: {"low": v["low_tertile"], "high": v["high_tertile"], "chi2": v["chi2"], "q": pval(v["q_fdr"])} for f, v in sp["survival"].items()}
    return {
        "emotion": {"density": E["map"]["density"], "moods": tags["moods"], "summary": tags["summary"], "order": E["clustered_order"], "pearson": matrix("pearson"), "spearman": matrix("spearman"), "agreement": {"table": tags["agreement"]["catalogue_medians"]["table"]}},
        "pop": {"hist": {"edges": O["popularity"]["bin_edges"], "a": O["popularity"]["sample_1m"], "b": O["popularity"]["sample_genre"]},
                "cmp": [{"f": f, "d": c["cliffs_delta"], "lo": c["delta_ci"][0], "hi": c["delta_ci"][1], "wd": within[f]["cliffs_delta"], "wlo": within[f]["delta_ci"][0], "whi": within[f]["delta_ci"][1], "ha": c["median_a"], "hb": c["median_b"], "q": pval(c["q_fdr"])} for f, c in P["comparisons"].items()],
                "env": {"bands": P["envelope"]["bands"], "all_hits": P["envelope"]["inside_all_hits"], "all_others": P["envelope"]["inside_all_others"]},
                "ladder": [{"label": l, "v": [x1, x2]} for l, x1, x2 in ladder]},
        "evo": {"years": V["years"], "features": V["features"], "series": V["series"], "trends": trends, "decades": decades},
        "genre": {"focal": G["focal_genres"], "all": G["genres"], "sum": summaries, "features": GENRE_FEATURES, "ranges": ranges, "sim": {"order": G["similarity"]["order"], "cosine": G["similarity"]["cosine"]},
                  "confused": G["classifier"]["most_confused"], "overlap": G["overlap"]["top_pairs"][:14],
                  "arch": {"items": items, "features": arch["features"], "x": arch["pca"]["x"], "y": arch["pca"]["y"], "c": arch["pca"]["cluster"], "expl": arch["pca"]["explained"]}},
        "sad": {"surface": {"lift": S["surface"]["lift_adjusted"], "n": S["surface"]["n"]}, "rates": rates, "years": {"years": year_data["years"], "hit": year_data["hit_median"], "all": year_data["all_median"]},
                "genres": [{"genre": r["genre"], "rho": r["rho"], "q": pval(r["q_fdr"])} for r in S["per_genre"]]},
        "geo": {"features": GEO["features"], "periods": periods, "conv": {"x": x, **out, "size": conv["panel_size"]},
                "reach": {"edges": reach["countries_edges"], "hist": reach["countries_hist"], "ex": reach["examples"], "tests": [{"feature": t["feature"], "a": t["median_a"], "b": t["median_b"], "d": t["cliffs_delta"], "q": pval(t["q_fdr"])} for t in reach["tests"]]},
                "origin": origin, "yearFeatures": GEO["features"], "countryYear": {"years": years, "names": {c: COUNTRY_NAMES[c] for c in sorted(seen) if c in COUNTRY_NAMES}, "rows": country_rows}},
        "hits": {"models": models, "surv": surv, "peaks": sp["by_peak_band"], "flash": [{"f": r["feature"], "d": r["cliffs_delta"], "lo": r["delta_ci"][0], "hi": r["delta_ci"][1], "q": pval(r["q_fdr"])} for r in sp["flash_vs_slow_burn"]]},
        "good": {"matrix": [{"f": m["feature"], "pop": m["popularity_pct_points"], "popQ": pval(m["popularity_q"]), "days": m["chart_days_pct"], "daysQ": pval(m["chart_days_q"]), "love": m["affection_points"], "loveQ": pval(m["affection_q"])} for m in GOOD["matrix"]]},
        "dash": DASH,
    }


e = html.escape


def fig(key, extra=""):
    item = F["figures"][key]
    return (f'<figure class="fig" data-fig="{key}"><h3>{e(item["question"])}</h3><div class="ctl"></div><div class="plot"></div>'
            f'<p class="reading">{e(item["reading"])}</p>{extra}<details class="data"><summary>View the data as a table</summary><div class="tw"></div></details></figure>')


def source_rows():
    q = Q
    facts = {
        "spotify_tracks_genre": (q["spotify_tracks_genre"]["rows_raw"], q["spotify_tracks_genre"]["rows_clean"], "Track ID"),
        "spotify_catalogue": (q["spotify_catalogue"]["rows_raw"], q["spotify_catalogue"]["rows_clean"], "Track ID"),
        "spotify_1m": (q["spotify_1m"]["rows_raw"], q["spotify_1m"]["rows_clean"], "Track ID"),
        "charts_2017_2021": (q["charts_2017_2021"]["rows_raw"], q["charts_2017_2021"]["rows_top200"], "Track ID from URL; artist + title"),
        "charts_2023_2025": (q["charts_2023_2025"]["rows_raw"], q["charts_2023_2025"]["rows_clean"], "Track ID"),
        "billboard_weekly": (q["billboard"]["weekly"]["rows_raw"], q["billboard"]["weekly"]["rows_raw"] - q["billboard"]["weekly"]["repeated_week_position_rows"], "Source song ID"),
        "billboard_audio": (q["billboard"]["audio"]["rows_raw"], q["billboard"]["audio"]["rows_clean"], "Source song ID, track ID"),
    }
    rows = []
    for key, (raw, clean, join) in facts.items():
        m = MANIFEST[key]
        rows.append(f'<tr><td><a href="{e(m["page"])}" target="_blank" rel="noopener">{e(m["title"])}</a></td><td>{e(m["role"])}</td><td>{n0(raw)}</td><td>{n0(clean)}</td><td>{e(join)}</td><td>{e(m["licence"])}</td></tr>')
    t = q["lastfm_tags"]
    rows.append(f'<tr><td><a href="{e(MANIFEST["lastfm_tags_1"]["page"])}" target="_blank" rel="noopener">Music Tags Dataset, Last.fm files (tranbren)</a></td><td>Human-assigned tags</td><td>{n0(t["rows_raw"])}</td><td>{n0(t["join"]["matched_tracks"])}</td><td>Lead artist + title + year</td><td>{e(MANIFEST["lastfm_tags_1"]["licence"])}</td></tr>')
    if MB:
        rows.append(f'<tr><td><a href="https://musicbrainz.org/" target="_blank" rel="noopener">MusicBrainz</a></td><td>Home country of leading chart artists</td><td>{n0(MB["queried_now"])}</td><td>{n0(MB["status_counts"].get("resolved", 0))}</td><td>Exact normalised artist name</td><td>Core data CC0</td></tr>')
    return "".join(rows)


def hypothesis_rows():
    return "".join(f'<tr><td>{e(h["investigation"])}</td><td>{e(h["claim"])}</td><td>{e(h["method"])}</td><td>{e(h["result"])}</td><td>{e(h["effect"])}</td><td>{e(h["q"])}</td><td><span class="tag{" yes" if h["significant"] else ""}">{e(h["verdict"])}</span></td></tr>' for h in F["hypotheses"])


def body(data_json):
    join = Q["merge"]["charts_2017_2021_to_audio"]["stream_share_pct_by_tier"]
    bb = Q["billboard"]["join"]
    tag = Q["lastfm_tags"]["join"]
    total_ids = Q["merge"]["unique_spotify_ids"]
    agree = Q["merge"]["feature_agreement_pearson"]["spotify_1m / spotify_catalogue"]
    chapters = [("intro", "Question and method"), ("data", "Data and quality"), ("explore", "Exploratory analysis"), ("tests", "Hypothesis tests"), ("genre", "Genres and archetypes"), ("evolution", "Evolution"), ("geography", "Geography"), ("limits", "Findings and limits"), ("conclusion", "Conclusions"), ("dashboard", "Explorer")]
    rail = "".join(f'<li><a href="#{k}"><span>{i}</span>{e(t)}</a></li>' for i, (k, t) in enumerate(chapters, 1))
    answers = "".join(f'<div class="answer"><h3>{e(a["topic"])}<b>{e(a["verdict"])}</b></h3><p>{e(a["text"])}</p></div>' for a in F["answers"])
    limits = "".join(f"<li>{e(x)}</li>" for x in F["limitations"])
    good = F["figures"]["good-matrix"]["reading"].replace("Only partly. ", "Popular, lasting and loved music overlap only partly. ", 1)
    mb_line = ""
    if MB:
        mb_line = f' MusicBrainz identified a home country for {n0(MB["status_counts"].get("resolved", 0))} of the {n0(MB["queried_now"])} most-charted artists looked up, covering {MB["resolved_share_of_all_chart_days_pct"]:.0f}% of all country chart days. The service limits request rates, so the lookup stops at the leading artists.'
    return f'''<div id="tip" role="status"></div>
<div class="shell">
<nav class="rail" aria-label="Chapters"><a class="brand" href="#top">SOUND / SIGNAL</a><ol>{rail}</ol><a class="go" href="#dashboard">Open the explorer</a></nav>
<main id="top">
<header class="hero">
<p class="kicker">A statistical study of music, taste and attention</p>
<h1>SOUND <i>/</i> SIGNAL</h1>
<p class="deck">What can the measurable properties of a recording tell us about whether people will love it, stream it, or remember it?</p>
<figure class="wave"><div id="wave"></div><figcaption>Each bar is one year of the Billboard Hot 100, 1959 to 2020. Bar height is the chart's mean valence, Spotify's measure of musical brightness. The tallest year, in orange, is {V["years"][max(range(len(V["years"])), key=lambda i: V["series"]["valence"]["chart"][i])]}.</figcaption></figure>
<div class="answers">{answers}</div>
</header>

<section class="chapter" id="intro"><p class="eyebrow">Chapter 1</p><h2>Can music's success be measured in its sound?</h2><div class="prose">
<p>This study asks whether measurable patterns explain popularity, emotional perception, genre identity and the evolution of mainstream music. It treats a song as both a piece of culture and a set of numbers, and it keeps those two views apart: the numbers can describe what is common among successful music, but they cannot say why any song moved someone.</p>
<p>The question of what makes music <em>good</em> has no column in any dataset. We therefore compare three things that can be observed: how much music is played (popularity), how long it stays on the charts (sustained attention), and whether listeners tag it with words of affection such as "favorite" or "beautiful" (expressed regard).</p>
<h3>Method</h3>
<p>Every investigation follows the same order: inspect distributions, check test assumptions, test the claim, report an effect size with an interval, and correct for multiple comparisons. Sample sizes are so large that tiny differences reach significance, so effect sizes carry the interpretation.</p>
<ul><li><strong>Group comparisons:</strong> Kruskal-Wallis, ANOVA and Welch ANOVA, with Dunn-Holm and Games-Howell post-hoc tests, reporting ε² and η².</li><li><strong>Two-group comparisons:</strong> Mann-Whitney tests with Cliff's δ and bootstrap intervals.</li><li><strong>Associations:</strong> Pearson and Spearman correlation, and chi-square with Cramér's V.</li><li><strong>Structure:</strong> PCA, k-means and Ward clustering, with cluster count chosen by silhouette and bootstrap stability.</li><li><strong>Models:</strong> multiple regression with year and genre effects and standard errors clustered by artist, checked out of sample.</li><li><strong>Time:</strong> Theil-Sen slopes, Mann-Kendall tests, block bootstrap intervals, change-point detection and survival curves.</li></ul>
<p>False-discovery-rate correction (Benjamini-Hochberg) is applied within each family of tests. No finding in this report establishes that a musical trait causes success.</p></div></section>

<section class="chapter" id="data"><p class="eyebrow">Chapter 2</p><h2>Nine open sources, joined with care</h2><div class="prose"><p>The study covers {n0(total_ids)} distinct Spotify track IDs, {n0(Q["charts_2017_2021"]["rows_top200"])} daily Top 200 chart rows, {n0(Q["charts_2023_2025"]["rows_clean"])} daily Top 50 rows across {Q["charts_2023_2025"]["countries"]} countries, and {n0(Q["billboard"]["weekly"]["rows_raw"])} weekly Billboard positions. Raw files are kept unchanged and checksummed. All cleaning is scripted.</p></div>
<div class="wide-table"><table><thead><tr><th>Source</th><th>Used for</th><th>Raw rows</th><th>Rows kept</th><th>Join key</th><th>Licence as listed</th></tr></thead><tbody>{source_rows()}</tbody></table></div>
<div class="prose"><p>Spotify Charts, the Million Song Dataset and the Music Information Retrieval collections listed in the brief were reviewed. Spotify's own chart site requires an account for downloads, so the two chart datasets above are open archives of it. The Million Song Dataset uses older Echo Nest features without Spotify IDs, so it was not merged.</p></div>
<ol class="pipeline"><li><b>Download</b>Files are fetched once and recorded with SHA-256 checksums.</li><li><b>Validate</b>Rows with impossible audio values, no tempo or lengths outside 30 seconds to 15 minutes are removed and counted.</li><li><b>Join on IDs</b>Spotify track IDs link sources wherever possible.</li><li><b>Validated fallback</b>Lead artist + title is used only if every candidate recording has near-identical valence and energy.</li><li><b>Analyse</b>Each investigation writes a results file that feeds this page.</li></ol>
<h3>Quality checks that changed the analysis</h3><div class="prose"><ul>
<li><strong>Sources agree.</strong> For {n0(agree["shared_ids"])} tracks present in two large Spotify samples, energy correlates at {agree["energy"]:.3f} and valence at {agree["valence"]:.3f}.</li>
<li><strong>Join coverage is reported.</strong> Audio features cover {join["spotify_id"] + join["artist_title_validated"]:.0f}% of 2017-2021 chart streams ({join["spotify_id"]:.0f}% by ID and {join["artist_title_validated"]:.0f}% through the validated fallback), {bb["chart_slot_match_pct"]:.0f}% of Hot 100 chart slots, and {tag["match_pct"]:.0f}% of Last.fm tag rows.{mb_line}</li>
<li><strong>A genre artefact was found.</strong> In the 1M-track sample, tracks labelled pop have a median popularity of {P["genre_label_audit"]["highest_median_popularity"]["pop"]:.0f}, while {P["genre_label_audit"]["genres_with_zero_median"]} genres have a median of zero. Labels match the other genre dataset for only {pc(P["genre_label_audit"]["label_agreement_share"], 0)} of shared tracks. Tracks were gathered through genre-and-year queries, so key comparisons are repeated inside genre and year.</li>
<li><strong>Blank countries are the Global chart.</strong> In the 2023-2025 file a blank country marks the Global chart, so those rows are relabelled and never counted as a market.</li>
<li><strong>Outliers are kept.</strong> Extreme but valid values remain in the data, flagged with Tukey's rule, and rank-based methods limit their influence.</li></ul></div></section>

<section class="chapter" id="explore"><p class="eyebrow">Chapter 3</p><h2>The shape of sound and attention</h2>
<div class="prose"><p>The first task is to see the raw structure: how musical characteristics relate to one another, whether they match what people hear, and how attention is shared out.</p></div>
{fig("emotion-map")}{fig("correlation")}{fig("mood-agreement")}{fig("popularity-dist")}{fig("hit-profile")}{fig("narrow")}{fig("ladder")}</section>

<section class="chapter" id="tests"><p class="eyebrow">Chapter 4</p><h2>Testing the common beliefs</h2><div class="prose"><p>The table collects the main hypothesis tests. The q-values are adjusted across the table. Normality and equal-variance checks were run first. They fail at this scale, so rank-based tests are primary and parametric versions serve as cross-checks.</p></div>
<div class="wide-table"><table><thead><tr><th>Inv.</th><th>Claim tested</th><th>Method</th><th>Result</th><th>Effect</th><th>q</th><th>Reading</th></tr></thead><tbody>{hypothesis_rows()}</tbody></table></div>
<h3>The sadness paradox</h3>{fig("sad-surface")}{fig("sad-time")}<h3>What makes a hit?</h3>{fig("hit-model")}{fig("longevity")}{fig("flash")}<h3>Good, liked and popular</h3>{fig("good-matrix")}</section>

<section class="chapter" id="genre"><p class="eyebrow">Chapter 5</p><h2>Genres are real, and their edges are blurred</h2>{fig("genre-violins")}{fig("genre-similarity")}{fig("genre-overlap")}{fig("archetypes")}</section>
<section class="chapter" id="evolution"><p class="eyebrow">Chapter 6</p><h2>Sixty years of the mainstream</h2><div class="prose"><p>The Hot 100 series weights each song by its weeks on the chart in that year. The catalogue series averages tracks by release year from a 1.2-million-track sample, so the two lines describe different populations.</p></div>{fig("evolution")}{fig("decades")}</section>
<section class="chapter" id="geography"><p class="eyebrow">Chapter 7</p><h2>The world is not listening to one chart</h2>{fig("country-heat")}{fig("convergence")}{fig("reach")}{fig("origin")}</section>

<section class="chapter" id="limits"><p class="eyebrow">Chapter 8</p><h2>What we found, and what we cannot claim</h2><div class="prose"><p>The strongest pattern across the study is a mismatch between what is easy to measure and what drives attention. Audio features describe genre and era well. They describe popularity poorly. {e(good)}</p><h3>Limitations</h3><ul>{limits}</ul></div></section>
<section class="chapter" id="conclusion"><p class="eyebrow">Chapter 9</p><h2>So, can we measure what makes music succeed?</h2><div class="prose"><p>We can measure the conditions around success much better than success itself. The sound of a track says a lot about what kind of music it is and when it was made. It says little about whether a given track will be popular, once the artist and context are known.</p><p>Popular music does share some tendencies: it is usually vocal, relatively loud, and a little shorter and more danceable than the average release. Those tendencies largely describe the styles that dominate attention, because they shrink to negligible size when tracks are compared within the same genre and year. Sad songs succeed, charts have grown less bright, and the charts of different countries have grown further apart.</p><p>Whether music is good stays with its listeners. The data can show who is listening, for how long, and what they say about it.</p></div></section>

<section class="chapter" id="dashboard"><p class="eyebrow">Interactive explorer</p><h2>Explore {n0(len(DASH["rows"]))} tracks</h2><div class="prose"><p>Filter charting and genre-leading tracks by sound, genre, archetype, year and country, then inspect any song. Tracks come from the chart and genre sources with published licences. The year shown is the earliest known release or chart year.</p></div><div class="explorer" id="explorer"></div></section>
<footer>All figures are computed from public datasets listed in Chapter 2. Associations are descriptive. Audio features originate from Spotify; chart data originates from Spotify and Billboard; tags from Last.fm users; artist data from MusicBrainz. Reproduce the study with the project's notebook and scripts.</footer>
</main></div>
<script type="application/json" id="sound-data">{data_json}</script>'''


def main():
    fragment = None
    if "--fragment" in sys.argv:
        fragment = Path(sys.argv[sys.argv.index("--fragment") + 1])
    data = json.dumps(round_deep(payload()), ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    css = (ROOT / "site_src" / "style.css").read_text(encoding="utf8")
    js = (ROOT / "site_src" / "app.js").read_text(encoding="utf8")
    fonts = '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,300;12..96,700;12..96,800&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">'
    head = f'<title>SOUND / SIGNAL</title><meta name="claude:viewport-fit-sides" content="cover">{fonts}<style>{css}</style>'
    content = body(data) + f"<script>{js}</script>"
    SITE_DIR.mkdir(exist_ok=True)
    page = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"><meta name="description" content="A statistical study of music, taste, popularity and cultural change.">{head}</head><body>{content}</body></html>'
    (SITE_DIR / "index.html").write_text(page, encoding="utf8")
    print(f"site/index.html: {len(page.encode()) / 1e6:.2f} MB")
    if fragment:
        fragment.write_text(head + content, encoding="utf8")
        print("fragment written")


if __name__ == "__main__":
    main()
