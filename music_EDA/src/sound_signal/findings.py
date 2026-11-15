"""Turns result files into the written findings used by the notebook and publication.

Numbers in the prose are read from outputs/results. The qualitative direction of
each claim is listed in `claims`, and `check()` stops the build if new data
contradicts the wording.
"""
from __future__ import annotations

import json

import numpy as np

from . import stats as ss
from .config import RESULTS_DIR

NAMES = {
    "danceability": "danceability", "energy": "energy", "loudness": "loudness", "speechiness": "speechiness",
    "acousticness": "acousticness", "instrumentalness": "instrumentalness", "liveness": "liveness",
    "valence": "valence", "tempo": "tempo", "duration_min": "duration",
}


def load_results() -> dict:
    names = ["overview", "emotion", "popularity", "evolution", "genre", "sadness", "geography", "hits", "good", "data_quality"]
    return {n: json.loads((RESULTS_DIR / f"{n}.json").read_text(encoding="utf8")) for n in names}


def n0(x):
    return f"{x:,.0f}"


def pc(x, digits=1):
    return f"{x * 100:.{digits}f}%"


def d2(x):
    return f"{x:.2f}"


def d3(x):
    return f"{x:.3f}"


def signed(x, digits=2):
    return f"{x:+.{digits}f}"


def pval(p):
    if p is None:
        return "n/a"
    return "<1e-300" if p == 0 else (f"{p:.1e}" if p < 0.001 else f"{p:.3f}")


def coef(model, term):
    return next(c for c in model["coefficients"] if c["term"] == term)


def pair(emotion, a, b):
    return next(p for p in emotion["pairs"] if {p["a"], p["b"]} == {a, b})


def build(R: dict | None = None) -> dict:
    R = R or load_results()
    e, p, v, g, s, geo, h, good, q = (R[k] for k in ("emotion", "popularity", "evolution", "genre", "sadness", "geography", "hits", "good", "data_quality"))
    tags = e["tags"]
    agreement = tags["agreement"]["catalogue_medians"]
    ladder = {k: x["r2"] for k, x in p["ladder"].items()}
    ladder2 = {k: x["r2"] for k, x in p["ladder_independent_genre_sample"].items()}
    cmp = p["comparisons"]
    within = {r["feature"]: r for r in p["within_genre_year_comparisons"]}
    trend = v["trends"]
    clf = g["classifier"]
    tests = {t["feature"]: t for t in g["tests"]}
    arch = g["archetypes"]
    bb, sp = h["billboard"], h["spotify"]
    conv = geo["convergence"]
    late = np.array(conv["is_late"], dtype=bool)
    jac = np.array(conv["jaccard"])
    core = np.array(conv["core25_jaccard"])
    reach = geo["reach"]
    recent = geo["periods"]["2023-2025"]
    early = geo["periods"]["2017-2021"]
    sad_rates = s["rates"]
    inter = s["interaction_model"]
    inter_year = s["interaction_year_effects_only"]
    energy_ac = pair(e, "energy", "acousticness")
    val_dance = pair(e, "valence", "danceability")
    energy_tempo = pair(e, "energy", "tempo")
    days_model = sp["models"]["log_chart_days"]
    ev = lambda f: trend[f]["chart"]
    dec = lambda f: v["decades"][f]["summary"]
    surv = sp["survival"]
    sustained = {r["feature"]: r for r in bb["sustained_vs_short_among_top40"]}
    flash = {r["feature"]: r for r in sp["flash_vs_slow_burn"]}
    reach_tests = {t["feature"]: t for t in reach["tests"]}
    matrix = {m["feature"]: m for m in good["matrix"]}
    geo_tests = recent["tests"]
    strongest_geo = max(geo_tests, key=lambda f: geo_tests[f]["permutation"]["between_share_of_variance"])
    furthest = sorted(zip(recent["distance_from_global"], recent["names"]), reverse=True)[:3]
    closest = sorted(zip(recent["distance_from_global"], recent["names"]))[:3]
    origin = geo.get("origin")
    origin_text = "Artist home countries were not available for this build."
    origin_facts = {}
    if origin:
        rows = origin["periods"]["2023-2025"]["countries"]
        ranked = sorted(rows, key=lambda r: -r["local_share"])
        low = [r for r in ranked if r["group"] != "North America"][-3:]
        origin_facts = {"top": ranked[:3], "low": low, "n": len(rows)}
        origin_text = (
            f"Among chart days whose lead artist could be placed, home-country artists hold the largest share in "
            f"{ranked[0]['name']} ({pc(ranked[0]['local_share'], 0)}), {ranked[1]['name']} ({pc(ranked[1]['local_share'], 0)}) and "
            f"{ranked[2]['name']} ({pc(ranked[2]['local_share'], 0)}). The lookup covers only the most-charted artists, "
            f"so these are shares of identified chart days, and small local acts are under-counted."
        )

    figures = {
        "emotion-map": {
            "question": "Where does recorded music sit on a map of mood?",
            "reading": f"Most tracks are energetic: {pc(e['quadrant_share']['elated (high valence, high energy)'] + e['quadrant_share']['tense (low valence, high energy)'], 0)} score at least 0.5 on energy, and only {pc(e['quadrant_share']['serene (high valence, low energy)'], 0)} combine high valence with low energy. The serene corner is thin in this sample, which suggests calm music is rarely measured as bright.",
        },
        "correlation": {
            "question": "Which audio measurements carry the same information?",
            "reading": f"Energy and acousticness are close to opposites (Spearman {d2(energy_ac['spearman'])}), and energy clusters tightly with loudness. Valence moves with danceability ({d2(val_dance['spearman'])}), so bright-sounding tracks also tend to be easier to dance to. Tempo has almost no link with valence ({d2(pair(e, 'valence', 'tempo')['spearman'])}) and only a modest one with energy ({d2(energy_tempo['spearman'])}).",
        },
        "mood-agreement": {
            "question": "Do listeners' mood tags agree with the measured audio?",
            "reading": f"Partly. Tags separate energy well (ε² {d2(tags['tests']['energy']['epsilon_squared'])}) and valence less well (ε² {d2(tags['tests']['valence']['epsilon_squared'])}). Agreement on the four-way mood quadrant is weak: Cohen's κ is {d2(agreement['kappa'])} (95% CI {d2(agreement['kappa_ci_artist_bootstrap'][0])} to {d2(agreement['kappa_ci_artist_bootstrap'][1])}). Calm-tagged music has a median valence of only {d2(tags['summary']['calm']['valence_median'])}, close to sad-tagged music at {d2(tags['summary']['sad']['valence_median'])}, so valence does not capture the difference people hear between calm and sad.",
        },
        "popularity-dist": {
            "question": "How is popularity distributed across tracks?",
            "reading": f"Popularity is strongly right-skewed. The median track scores {n0(R['overview']['popularity']['describe_1m']['median'])} out of 100, and {pc(R['overview']['popularity']['zero_share_1m'], 0)} score exactly zero. Scores are stable measurements: for {n0(p['genre_label_audit']['shared_tracks'])} tracks seen in two separately collected datasets, the rank correlation is {d2(p['genre_label_audit']['popularity_snapshot_spearman'])}.",
        },
        "hit-profile": {
            "question": "How do popular tracks differ from unpopular ones?",
            "reading": f"The largest gap is in instrumentalness (Cliff's δ {d2(cmp['instrumentalness']['cliffs_delta'])}): the typical hit scores close to zero on it. Hits are also louder (δ {d2(cmp['loudness']['cliffs_delta'])}), more danceable ({d2(cmp['danceability']['cliffs_delta'])}) and shorter ({d2(cmp['duration_min']['cliffs_delta'])}). Inside the same genre and year every one of these gaps shrinks to a negligible size, the largest being {d2(within['instrumentalness']['cliffs_delta'])} for instrumentalness and {d2(within['loudness']['cliffs_delta'])} for loudness.",
        },
        "narrow": {
            "question": "Is popularity confined to a narrow musical range?",
            "reading": f"Only loosely. {pc(p['envelope']['inside_all_hits'], 0)} of hits fall inside the central band of hit values on all nine features, and so do {pc(p['envelope']['inside_all_others'], 0)} of other tracks. Hits look {pc(1 - p['dispersion']['raw']['ratio'], 0)} less spread out overall, but within genre the ratio is {d2(p['dispersion']['within_genre']['ratio'])}, which means no narrower. The apparent formula mostly reflects which styles hits come from.",
        },
        "ladder": {
            "question": "How much of popularity can each kind of information predict?",
            "reading": f"Audio alone predicts {pc(ladder['audio_flexible_deciles'])} of the variation in unseen tracks at best. Knowing the artist's other tracks predicts {pc(ladder['artist_only'])}. An independently collected sample gives the same order: {pc(ladder2['audio_linear'])} for audio and {pc(ladder2['artist_only'])} for artist. Genre scores high here ({pc(ladder['genre_only'])}), but those labels reflect how the samples were gathered, so that bar should not be read as the effect of genre.",
        },
        "evolution": {
            "question": "How has mainstream music changed since 1959, and did the wider catalogue change with it?",
            "reading": f"Hot 100 music became louder by {d2(ev('loudness')['per_decade'])} dB per decade, less acoustic by {d3(-ev('acousticness')['per_decade'])} and less bright by {d3(-ev('valence')['per_decade'])} valence points per decade. The median song debuting in the 2010s has valence {d2(dec('valence')['2010s']['median'])}, against {d2(dec('valence')['1960s']['median'])} in the 1960s. The wider catalogue is thinly sampled before the 1970s and its valence rose until about 1980, so its overall slope is about half as steep ({d3(-trend['valence']['catalogue']['per_decade'])} per decade). Tempo shows no reliable long-run trend on the chart.",
        },
        "decades": {
            "question": "How do feature distributions differ between decades?",
            "reading": f"Decade explains the most variation in duration (ε² {d2(v['decades']['duration_min']['epsilon_squared'])}) and loudness ({d2(v['decades']['loudness']['epsilon_squared'])}), and almost none in tempo ({d3(v['decades']['tempo']['epsilon_squared'])}). The median charting song grew from {d2(dec('duration_min')['1960s']['median'])} minutes in the 1960s to {d2(dec('duration_min')['1990s']['median'])} in the 1990s, then shortened to {d2(dec('duration_min')['2010s']['median'])} in the 2010s.",
        },
        "genre-violins": {
            "question": "How differently are genres distributed on key features?",
            "reading": f"Genres differ most on acousticness (ε² {d2(tests['acousticness']['epsilon_squared'])}) and energy ({d2(tests['energy']['epsilon_squared'])}). Tempo ({d2(tests['tempo']['epsilon_squared'])}) and liveness ({d2(tests['liveness']['epsilon_squared'])}) say little about genre. Spreads still overlap widely, so a single feature rarely identifies a genre.",
        },
        "genre-similarity": {
            "question": "Which genres sound alike?",
            "reading": f"Blocks along the diagonal are families of genres with similar audio profiles. From audio alone, a simple classifier names the right genre {pc(clf['accuracy'])} of the time, about {clf['accuracy'] / clf['chance']:.0f} times chance among {len(g['genres'])} genres, and puts it in the top five {pc(clf['top5'], 0)} of the time. It tested on artists it had not seen. Sonically specific styles such as grindcore and minimal techno are among the easiest to recover. Labels based on language or place, such as German and Swedish, are among the hardest.",
        },
        "genre-overlap": {
            "question": "Which genres share the same tracks?",
            "reading": f"{pc(g['overlap']['share_of_tracks'], 0)} of tracks carry more than one genre label. Genres that share tracks also sound closer (rank correlation {d2(g['overlap']['jaccard_vs_audio_distance_spearman'])} between overlap and audio distance). Treat the very top pairs with care: 'singer-songwriter' and 'songwriter' are the same list, and pairs whose names contain each other, such as dub and dubstep, may partly reflect keyword searches.",
        },
        "archetypes": {
            "question": "What musical types appear if genre labels are ignored?",
            "reading": f"Clustering the audio alone gives {arch['k']} recurring archetypes, chosen for separation and stability. They line up strongly with genre labels the algorithm never saw (Cramér's V {d2(arch['genre_association']['cramers_v'])}). The boundaries are soft: the silhouette score is only {d2(next(x['silhouette'] for x in arch['search'] if x['k'] == arch['k']))}, so these are regions of a continuum, and archetype says little about popularity (ε² {d3(arch['popularity_test']['epsilon_squared'])}).",
        },
        "sad-surface": {
            "question": "Do sad songs become hits, and does energy change that?",
            "reading": f"Yes, they do. {n0(sad_rates['low']['hits'])} low-valence tracks are hits. Their hit rate is {pc(sad_rates['low']['hit_rate'])}, against {pc(sad_rates['high']['hit_rate'])} for high-valence tracks, and the gap narrows to {pc(1 - s['observed_over_expected_within_genre_year']['low'], 0)} below expectation within genre and year. The interaction is positive: the valence penalty is smaller for energetic songs ({signed(s['simple_slopes_valence']['energy_plus_1sd'])} percentile points per SD at high energy, {signed(s['simple_slopes_valence']['energy_minus_1sd'])} at low energy).",
        },
        "sad-time": {
            "question": "Have popular songs become sadder?",
            "reading": f"Yes. The median valence of hits fell by {d3(-s['hit_valence_by_year']['trend']['slope_per_year'] * 10)} per decade from 2000, but the whole sample fell too, and hits stayed slightly brighter than other tracks. Within individual genres the direction is mixed: valence relates negatively to popularity in {pc(s['share_genres_negative_significant'], 0)} of genres and positively in {pc(s['share_genres_positive_significant'], 0)}.",
        },
        "country-heat": {
            "question": "How do countries' charts differ in sound?",
            "reading": f"{furthest[0][1]}, {furthest[1][1]} and {furthest[2][1]} sit furthest from the Global chart in 2023-2025, while {closest[0][1]}, {closest[1][1]} and {closest[2][1]} sit closest. Country accounts for at most {pc(geo_tests[strongest_geo]['permutation']['between_share_of_variance'], 0)} of chart-weighted variation in any feature (the largest is {NAMES[strongest_geo]}). Those differences are far above chance but small beside the variety within each chart.",
        },
        "convergence": {
            "question": "Are countries' charts converging or diverging?",
            "reading": f"Diverging. Two countries' Top 50 lists shared {pc(jac[~late].mean(), 0)} of their songs on average in 2017-2021 and {pc(jac[late].mean(), 0)} in 2023-2025. The drop remains for a fixed core of 25 songs per country ({pc(core[~late].mean(), 0)} to {pc(core[late].mean(), 0)}), and sound profiles moved apart even after allowing for sampling noise. The two periods come from different collections and 2022 is missing, so the size of the step is less certain than its direction.",
        },
        "reach": {
            "question": "Which songs travel, and which stay home?",
            "reading": f"Most charting songs stay local: {pc(reach['single_country_share'], 0)} reached a Top 50 in just one of the 72 countries, and only {n0(reach['class_counts']['international'])} of {n0(reach['n_songs'])} reached 20 or more. Songs that travel have lower median valence ({d2(reach_tests['valence']['median_a'])} against {d2(reach_tests['valence']['median_b'])}), are less danceable and run longer. The differences are small, and language and marketing are not measured here.",
        },
        "origin": {"question": "How much of each chart comes from home-grown artists?", "reading": origin_text},
        "hit-model": {
            "question": "Which audio features go with longer chart runs?",
            "reading": f"On the Spotify Global chart, one standard deviation more loudness goes with {signed(matrix['loudness']['chart_days_pct'], 0)}% chart days and more danceability with {signed(matrix['danceability']['chart_days_pct'], 0)}%, while energy ({signed(matrix['energy']['chart_days_pct'], 0)}%) and speechiness ({signed(matrix['speechiness']['chart_days_pct'], 0)}%) go the other way. All audio features together still explain just {pc(days_model['r2_within'])} of within-year variation in chart days. These are associations among songs that had already reached the chart.",
        },
        "longevity": {
            "question": "Do some kinds of songs last longer on the chart?",
            "reading": f"Songs in the most speech-heavy third last a median of {surv['speechiness']['high_tertile']['median']:.0f} days on the Global chart, against {surv['speechiness']['low_tertile']['median']:.0f} for the least speech-heavy third. The brightest third lasts {surv['valence']['high_tertile']['median']:.0f} days against {surv['valence']['low_tertile']['median']:.0f} for the darkest. Peak position separates songs far more: the median #1 song stayed {n0(sp['by_peak_band'][0]['median_days'])} days, while the median song peaking at 101-200 stayed {n0(sp['by_peak_band'][4]['median_days'])}.",
        },
        "flash": {
            "question": "What separates short-lived spikes from slow burners?",
            "reading": f"Among songs that reached the Global Top 50, those earning most of their streams in the first two weeks are less bright (δ {d2(flash['valence']['cliffs_delta'])}), less danceable ({d2(flash['danceability']['cliffs_delta'])}) and quieter ({d2(flash['loudness']['cliffs_delta'])}) than slow burners from the same year. Many spikes are album tracks that chart briefly on release, so release strategy is a likely part of this pattern.",
        },
        "good-matrix": {
            "question": "Do popular, lasting and loved music share the same traits?",
            "reading": f"Only partly. Loudness goes with all three outcomes, and speechiness and instrumentalness go against all three. Danceability splits them: it goes with popularity and chart life but slightly against affectionate tags. Longer songs are less popular yet receive more affection. Popularity and affection are only weakly related (Spearman {d2(good['popularity_affection_spearman'])}).",
        },
    }

    rows = [
        ("A", "Energy and acousticness move in opposite directions", "Spearman correlation", f"ρ = {d2(energy_ac['spearman'])}", "large", energy_ac["p"], "Supported"),
        ("A", "Mood tags differ in measured energy", "Kruskal-Wallis", f"H = {n0(tags['tests']['energy']['kruskal_h'])}", f"ε² = {d2(tags['tests']['energy']['epsilon_squared'])}", tags["tests"]["energy"]["kruskal_p"], "Supported"),
        ("A", "Mood tags differ in measured valence", "Kruskal-Wallis", f"H = {n0(tags['tests']['valence']['kruskal_h'])}", f"ε² = {d2(tags['tests']['valence']['epsilon_squared'])}", tags["tests"]["valence"]["kruskal_p"], "Supported"),
        ("A", "Tagged mood matches the measured mood quadrant", "Chi-square, Cohen's κ", f"χ² = {n0(agreement['chi_square']['chi2'])}", f"κ = {d2(agreement['kappa'])}, V = {d2(agreement['chi_square']['cramers_v'])}", agreement["chi_square"]["p"], "Weak agreement"),
        ("B", "Hits are louder than low-popularity tracks", "Mann-Whitney", f"medians {cmp['loudness']['median_a']:.1f} vs {cmp['loudness']['median_b']:.1f} dB", f"δ = {d2(cmp['loudness']['cliffs_delta'])}", cmp["loudness"]["p"], "Supported, small"),
        ("B", "The loudness gap survives inside genre and year", "Mann-Whitney", f"medians differ by {within['loudness']['median_a'] - within['loudness']['median_b']:.1f} dB", f"δ = {d2(within['loudness']['cliffs_delta'])}", within["loudness"]["p"], "Significant, negligible size"),
        ("B", "Hits are less instrumental", "Mann-Whitney", "hit median near 0", f"δ = {d2(cmp['instrumentalness']['cliffs_delta'])}", cmp["instrumentalness"]["p"], "Supported, medium"),
        ("C", "Hot 100 loudness rises over time", "Theil-Sen, Mann-Kendall", f"{signed(ev('loudness')['per_decade'])} dB per decade", f"τ = {d2(ev('loudness')['kendall_tau'])}", ev("loudness")["p"], "Supported"),
        ("C", "Hot 100 valence falls over time", "Theil-Sen, Mann-Kendall", f"{signed(ev('valence')['per_decade'], 3)} per decade", f"τ = {d2(ev('valence')['kendall_tau'])}", ev("valence")["p"], "Supported"),
        ("C", "Hot 100 tempo trends over time", "Theil-Sen, Mann-Kendall", f"{signed(ev('tempo')['per_decade'])} BPM per decade", f"τ = {d2(ev('tempo')['kendall_tau'])}", ev("tempo")["p"], "Interval includes zero"),
        ("C", "Song duration differs across decades", "Kruskal-Wallis, Dunn", f"H = {n0(v['decades']['duration_min']['kruskal_h'])}", f"ε² = {d2(v['decades']['duration_min']['epsilon_squared'])}", v["decades"]["duration_min"]["kruskal_p"], "Supported"),
        ("D", "Genres differ in acousticness", "Kruskal-Wallis, Welch ANOVA", f"H = {n0(tests['acousticness']['kruskal_h'])}", f"ε² = {d2(tests['acousticness']['epsilon_squared'])}", tests["acousticness"]["kruskal_p"], "Supported, large"),
        ("D", "Audio archetypes align with genre labels", "Chi-square", f"χ² = {n0(arch['genre_association']['chi2'])}", f"V = {d2(arch['genre_association']['cramers_v'])}", arch["genre_association"]["p"], "Supported"),
        ("E", "Hit rate depends on valence band", "Chi-square", f"χ² = {n0(s['chi_square_band_by_hit']['chi2'])}", f"V = {d2(s['chi_square_band_by_hit']['cramers_v'])}", s["chi_square_band_by_hit"]["p"], "Significant, very small"),
        ("E", "Energy changes the valence-popularity link", "OLS, clustered by artist", f"interaction {signed(coef(inter, 'valence_x_energy')['beta'])}", "points per SD²", coef(inter, "valence_x_energy")["p"], "Supported, small"),
        ("F", "Countries differ in chart valence", "Permutation test", f"{pc(geo_tests['valence']['permutation']['between_share_of_variance'])} between countries", f"chance {pc(geo_tests['valence']['permutation']['null_mean_share'])}", geo_tests["valence"]["permutation"]["p_permutation"], "Supported"),
        ("F", "Chart overlap declined during 2017-2021", "Theil-Sen, Mann-Kendall", f"{signed(conv['early_trend_jaccard']['slope_per_year'] * 4, 3)} per year", f"τ = {d2(conv['early_trend_jaccard']['kendall_tau'])}", conv["early_trend_jaccard"]["p"], "Supported"),
        ("G", "Peak position and chart longevity are related", "Chi-square", f"χ² = {n0(bb['chi_square_peak_longevity']['chi2'])}", f"V = {d2(bb['chi_square_peak_longevity']['cramers_v'])}", bb["chi_square_peak_longevity"]["p"], "Supported, large"),
        ("G", "Speech-heavy songs have different chart survival", "Log-rank", f"χ² = {surv['speechiness']['chi2']:.1f}", f"medians {surv['speechiness']['high_tertile']['median']:.0f} vs {surv['speechiness']['low_tertile']['median']:.0f} days", surv["speechiness"]["p"], "Supported"),
        ("G", "Danceability goes with more chart days", "OLS, clustered by artist", f"{signed(matrix['danceability']['chart_days_pct'], 0)}% per SD", f"model R² {pc(days_model['r2_within'])}", coef(days_model, "danceability")["p"], "Supported, small"),
    ]
    adjusted = ss.bh_fdr([max(r[5], 0) for r in rows])
    hypotheses = [{"investigation": r[0], "claim": r[1], "method": r[2], "result": r[3], "effect": r[4], "p": pval(r[5]),
                   "q": pval(float(a)), "verdict": r[6], "significant": bool(a < .05)} for r, a in zip(rows, adjusted)]

    answers = [
        {"topic": "Popularity", "verdict": "Weakly", "text": f"Audio features predict at most {pc(ladder['audio_flexible_deciles'], 0)} of popularity among unseen tracks. The artist's other tracks predict {pc(ladder['artist_only'], 0)}. Sound sets the conditions, and who releases the music explains far more."},
        {"topic": "Emotional perception", "verdict": "Partly", "text": f"Measured energy agrees with what listeners tag. Measured valence agrees less, and full mood agreement is weak (κ {d2(agreement['kappa'])})."},
        {"topic": "Genre identity", "verdict": "Substantially", "text": f"Genre explains up to {pc(tests['acousticness']['epsilon_squared'], 0)} of the rank variation in a single feature, and audio alone identifies genre about {clf['accuracy'] / clf['chance']:.0f} times better than chance."},
        {"topic": "Evolution of the mainstream", "verdict": "Clearly", "text": f"Over six decades the Hot 100 became louder, less acoustic and less bright. Song length rose until the 1990s and has fallen since."},
        {"topic": "Geography", "verdict": "Clearly, and it is diverging", "text": f"Average chart overlap between countries fell from {pc(jac[~late].mean(), 0)} to {pc(jac[late].mean(), 0)}. Most charting songs never leave one country."},
    ]
    limitations = [
        "Audio features are Spotify's model estimates. Valence in particular is an algorithmic score, and Spotify stopped offering these features to new API users in November 2024.",
        "Older recordings may be measured from remasters, which can inflate loudness and alter other features for early decades.",
        "The large Spotify samples were collected through queries and are not random samples of all music. Their genre labels partly reflect that collection.",
        "Spotify popularity is a current score that favours recent listening, so analyses compare tracks within release year.",
        "Every chart analysis studies songs that already charted. It cannot show what separates them from songs that never did.",
        f"Audio features could be matched to {pc(q['merge']['charts_2017_2021_to_audio']['stream_share_pct_by_tier']['spotify_id'] / 100 + q['merge']['charts_2017_2021_to_audio']['stream_share_pct_by_tier']['artist_title_validated'] / 100, 0)} of 2017-2021 chart streams. {len(early['excluded_low_coverage'])} countries with weak coverage, often those with strong local repertoire, are left out of that period's sound profiles.",
        "The two Spotify chart collections differ in source, and no data covers 2022. Changes between them are less certain than changes within either one.",
        "Marketing budgets, playlist placement, lyrics, language, label support and social media are not measured, so all of these can confound the associations reported here.",
        "Last.fm tags come from a self-selected community and are concentrated on well-known and Western music.",
        "No dataset measures whether music is good. Popularity, chart life and affectionate tags are proxies for attention and regard.",
    ]
    claims = [
        ("artist predicts more than audio", ladder["artist_only"] > ladder["audio_flexible_deciles"]),
        ("artist beats audio in the independent sample", ladder2["artist_only"] > ladder2["audio_linear"]),
        ("hit gaps negligible within genre", all(abs(r["cliffs_delta"]) < .147 for r in within.values())),
        ("hits no narrower within genre", p["dispersion"]["within_genre"]["ratio"] > .97),
        ("hits narrower before genre control", p["dispersion"]["raw"]["ratio"] < 1),
        ("instrumentalness is the largest hit gap", max(cmp, key=lambda f: abs(cmp[f]["cliffs_delta"])) == "instrumentalness"),
        ("hot 100 louder", ev("loudness")["ci"][0] > 0), ("hot 100 less bright", ev("valence")["ci"][1] < 0),
        ("hot 100 less acoustic", ev("acousticness")["ci"][1] < 0), ("tempo interval includes zero", ev("tempo")["ci"][0] < 0 < ev("tempo")["ci"][1]),
        ("catalogue valence falls more slowly", -.75 * ev("valence")["per_decade"] > -trend["valence"]["catalogue"]["per_decade"] > 0),
        ("duration peaked in 1990s", dec("duration_min")["1990s"]["median"] > max(dec("duration_min")["1960s"]["median"], dec("duration_min")["2010s"]["median"])),
        ("calm valence near sad", abs(tags["summary"]["calm"]["valence_median"] - tags["summary"]["sad"]["valence_median"]) < .1),
        ("energy tags clearer than valence", tags["tests"]["energy"]["epsilon_squared"] > tags["tests"]["valence"]["epsilon_squared"]),
        ("low-valence hit rate below high", sad_rates["low"]["hit_rate"] < sad_rates["high"]["hit_rate"]),
        ("interaction positive", coef(inter, "valence_x_energy")["beta"] > 0 and coef(inter_year, "valence_x_energy")["beta"] > 0),
        ("hit valence fell", s["hit_valence_by_year"]["trend"]["ci"][1] < 0),
        ("hits brighter than the whole sample", all(a > b for a, b in zip(s["hit_valence_by_year"]["hit_median"], s["hit_valence_by_year"]["all_median"]))),
        ("tempo unrelated to valence", abs(pair(e, "valence", "tempo")["spearman"]) < .1),
        ("charts diverged", jac[late].mean() < jac[~late].mean() and core[late].mean() < core[~late].mean() and conv["audio_excess_period_change"] > 0),
        ("travelling songs darker, less danceable, longer", reach_tests["valence"]["cliffs_delta"] < 0 and reach_tests["danceability"]["cliffs_delta"] < 0 and reach_tests["duration_min"]["cliffs_delta"] > 0),
        ("loudness positive on all outcomes", all(matrix["loudness"][k] > 0 for k in ("popularity_pct_points", "chart_days_pct", "affection_points"))),
        ("speech and instrumental negative on all outcomes", all(matrix[f][k] < 0 for f in ("speechiness", "instrumentalness") for k in ("popularity_pct_points", "chart_days_pct", "affection_points"))),
        ("danceability splits", matrix["danceability"]["popularity_pct_points"] > 0 and matrix["danceability"]["chart_days_pct"] > 0 and matrix["danceability"]["affection_points"] < 0),
        ("duration splits", matrix["duration_min"]["popularity_pct_points"] < 0 and matrix["duration_min"]["affection_points"] > 0),
        ("speech-heavy songs last less", surv["speechiness"]["high_tertile"]["median"] < surv["speechiness"]["low_tertile"]["median"]),
        ("bright songs last longer", surv["valence"]["high_tertile"]["median"] > surv["valence"]["low_tertile"]["median"]),
        ("flash songs darker, less danceable, quieter", all(flash[f]["cliffs_delta"] < 0 for f in ("valence", "danceability", "loudness"))),
        ("sound profiles: acousticness tops genre tests", g["tests"][0]["feature"] == "acousticness"),
    ]
    return {"figures": figures, "hypotheses": hypotheses, "answers": answers, "limitations": limitations, "claims": claims, "origin": origin_facts}


def check(findings: dict | None = None) -> None:
    findings = findings or build()
    failed = [name for name, ok in findings["claims"] if not ok]
    if failed:
        raise SystemExit("Written findings no longer match the results: " + "; ".join(failed))
    print(f"{len(findings['claims'])} written claims match the results")
