"""Write notebooks/sound_signal.ipynb and execute every code cell.

Cells run in this process and their text, tables and figures are stored as normal
notebook outputs, so the notebook also works where a Jupyter kernel cannot start.
It can be re-run cell by cell in Jupyter, VS Code or Colab.
"""
from __future__ import annotations

import base64
import contextlib
import io
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sound_signal import findings  # noqa: E402

F = findings.build()


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip().splitlines(True)}


def fig_cells(key, figure):
    item = F["figures"][key]
    return [md(f"#### {item['question']}"), code(f'show_figure("{figure}")'), md(f"**Interpretation.** {item['reading']}")]


SETUP = r'''
import json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "sound_signal").exists())
sys.path.insert(0, str(ROOT / "src"))
from sound_signal import figures, findings, stats
from sound_signal.config import AUDIO_FEATURES, PROCESSED_DIR, RAW_DIR

# Set to True to rebuild everything from the raw files first (about 30 minutes).
RUN_PIPELINE = False
if RUN_PIPELINE:
    for script in ("01_download.py", "02_clean_merge.py", "03_musicbrainz.py", "04_analyze.py"):
        subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True)

R = findings.load_results()
written = findings.build(R)
findings.check(written)
pd.set_option("display.max_columns", 30, "display.width", 160, "display.float_format", "{:,.3f}".format)

def show_figure(name):
    figure = figures.ALL[name](R)
    try:
        display(figure)
    finally:
        figures.plt.close(figure)
'''

cells = [
    md("""
# SOUND / SIGNAL
### Reproducible analysis notebook

**Central question.** Can we identify measurable patterns in music that explain popularity, emotional perception, genre identity and the evolution of mainstream music?

This notebook reads the cleaned data and saved results produced by the scripts in `scripts/`, shows each investigation, and re-derives a set of headline numbers directly from the cleaned tables. All findings are descriptive associations. None of them establishes what causes a song to succeed.
"""),
    code(SETUP),
    md("## 1. Data and quality\nRow counts, exclusions and join rates are logged by the cleaning step."),
    code(r'''
q = R["data_quality"]
rows = []
for name, key in [("Spotify tracks with genres", "spotify_tracks_genre"), ("Spotify 1.2M catalogue", "spotify_catalogue"), ("Spotify 1M tracks", "spotify_1m"), ("Top 50 by country, 2023-2025", "charts_2023_2025"), ("Last.fm tags", "lastfm_tags")]:
    s = q[key]
    rows.append({"source": name, "raw rows": s["rows_raw"], "invalid audio rows": s["invalid_audio_rows"], "clean rows": s.get("rows_clean", s.get("join", {}).get("matched_tracks"))})
display(pd.DataFrame(rows))
print("Billboard join:", q["billboard"]["join"])
print("2017-2021 chart audio match:", q["merge"]["charts_2017_2021_to_audio"])
pd.DataFrame(q["merge"]["feature_agreement_pearson"]).T
'''),
    md("### Reproducibility check\nThese figures are recalculated here from the cleaned tables and compared with the saved results."),
    code(r'''
if (PROCESSED_DIR / "genre_tracks.parquet").exists():
    g = pd.read_parquet(PROCESSED_DIR / "genre_tracks.parquet", columns=["track_id", "genre", "speechiness", "energy", "acousticness"])
    g = g[~g.genre.isin(["comedy", "sleep"]) & (g.speechiness <= .66)].drop_duplicates("track_id")
    live = g[["energy", "acousticness"]].rank().corr().iloc[0, 1]
    saved = next(p for p in R["emotion"]["pairs"] if {p["a"], p["b"]} == {"energy", "acousticness"})["spearman"]
    print(f"Energy-acousticness Spearman: recomputed {live:.4f}, saved {saved:.4f}")
    assert abs(live - saved) < 1e-3
    w = pd.read_parquet(PROCESSED_DIR / "billboard_weeks.parquet")
    print(f"Billboard weekly rows: {len(w):,}, weeks {w.week.min().date()} to {w.week.max().date()}")
else:
    print("Cleaned data not found. Run the pipeline to enable this check; saved results are still shown.")
'''),
    md("## 2. Descriptive statistics"),
    code(r'''pd.DataFrame(R["overview"]["descriptive_catalogue"]).T.assign(outlier_pct_kept=pd.Series(R["overview"]["outlier_pct_tukey_kept"]))'''),
    md("Outliers are flagged with Tukey's rule and kept, because extreme values such as a spoken intro or a very long track are real recordings. Rank-based tests limit their influence."),
    md("## 3. Investigation A: the emotional structure of music"),
    *fig_cells("emotion-map", "01_emotion_map"), *fig_cells("correlation", "02_correlation_heatmap"), *fig_cells("mood-agreement", "03_mood_agreement"),
    code(r'''pd.DataFrame(R["emotion"]["pairs"]).drop(columns=["pearson_ci", "spearman_ci"])'''),
    md("## 4. Investigation B: the popularity paradox"),
    *fig_cells("popularity-dist", "04_popularity_distribution"), *fig_cells("hit-profile", "05_hit_profile"), *fig_cells("ladder", "06_prediction_ladder"),
    md(f"**Is popularity confined to a narrow range?** {F['figures']['narrow']['reading']}"),
    code(r'''pd.DataFrame(R["popularity"]["regression"]["coefficients"]).assign(unit="percentile points per SD")'''),
    md("## 5. Investigation C: evolution of mainstream music"),
    *fig_cells("evolution", "07_evolution_timeline"), *fig_cells("decades", "08_decade_distributions"),
    code(r'''
pd.DataFrame({f: {"Hot 100 per decade": t["chart"]["per_decade"], "Hot 100 q": t["chart"]["q_fdr"], "catalogue per decade": t["catalogue"]["per_decade"],
                  "catalogue q": t["catalogue"]["q_fdr"], "change points": ", ".join(str(c["year"]) for c in t["chart"]["change_points"]),
                  "anomalous years": ", ".join(str(a["year"]) for a in t["chart"]["anomalies"])} for f, t in R["evolution"]["trends"].items()}).T
'''),
    md("## 6. Investigation D: genre identity and overlap"),
    *fig_cells("genre-violins", "09_genre_violins"), *fig_cells("genre-similarity", "10_genre_similarity"),
    md(f"**Overlap.** {F['figures']['genre-overlap']['reading']}"), *fig_cells("archetypes", "11_archetypes_pca"),
    code(r'''pd.DataFrame(R["genre"]["tests"]).drop(columns="assumptions")'''),
    md("Formal normality and equal-variance checks fail at this scale, and variance ratios are large, so Kruskal-Wallis with Dunn-Holm is the primary test. Welch ANOVA and Games-Howell are reported as a parametric cross-check."),
    md("## 7. Investigation E: the sadness paradox"),
    *fig_cells("sad-surface", "12_sadness_surface"), md(f"**Over time.** {F['figures']['sad-time']['reading']}"),
    code(r'''pd.DataFrame(R["sadness"]["interaction_model"]["coefficients"])'''),
    md("## 8. Investigation F: geography"),
    *fig_cells("country-heat", "13_country_comparison"), *fig_cells("convergence", "14_convergence"),
    md(f"**Reach.** {F['figures']['reach']['reading']}\n\n**Home artists.** {F['figures']['origin']['reading']}"),
    md("## 9. Investigation G: what makes a hit?"),
    *fig_cells("longevity", "15_chart_longevity"), md(f"**Features and chart life.** {F['figures']['hit-model']['reading']}\n\n**Spikes and slow burners.** {F['figures']['flash']['reading']}"),
    *fig_cells("good-matrix", "16_good_liked_popular"),
    md("## 10. Hypothesis tests\nP-values are adjusted across this table with Benjamini-Hochberg. With samples this large, the effect size matters more than significance."),
    code(r'''pd.DataFrame(written["hypotheses"]).drop(columns="significant")'''),
    md("## 11. Conclusions\n" + "\n".join(f"- **{a['topic']}: {a['verdict']}.** {a['text']}" for a in F["answers"])),
    md("## 12. Limitations\n" + "\n".join(f"- {x}" for x in F["limitations"])),
]


def run(cells_to_run):
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.figure import Figure

    def rich(value):
        if isinstance(value, Figure):
            buffer = io.BytesIO()
            value.savefig(buffer, format="png", bbox_inches="tight", dpi=110)
            return {"image/png": base64.b64encode(buffer.getvalue()).decode(), "text/plain": ["<Figure>"]}
        if isinstance(value, pd.DataFrame):
            return {"text/html": value.to_html(max_rows=40).splitlines(True), "text/plain": value.to_string(max_rows=40).splitlines(True)}
        return {"text/plain": repr(value).splitlines(True)}

    outputs: list = []
    space = {"__name__": "__main__", "display": lambda value: outputs.append({"output_type": "display_data", "metadata": {}, "data": rich(value)})}
    count = 0
    import ast
    for cell in cells_to_run:
        if cell["cell_type"] != "code":
            continue
        count += 1
        outputs = []
        cell["execution_count"] = count
        source = "".join(cell["source"])
        stream = io.StringIO()
        try:
            tree = ast.parse(source)
            last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            with contextlib.redirect_stdout(stream):
                exec(compile(tree, "cell", "exec"), space)
                result = eval(compile(ast.Expression(last.value), "cell", "eval"), space) if last else None
        except Exception:
            print(source[:300])
            traceback.print_exc()
            raise SystemExit("A notebook cell failed")
        if stream.getvalue():
            outputs.insert(0, {"output_type": "stream", "name": "stdout", "text": stream.getvalue().splitlines(True)})
        if result is not None:
            outputs.append({"output_type": "execute_result", "execution_count": count, "metadata": {}, "data": rich(result)})
        cell["outputs"] = outputs
        plt.close("all")
    return count


def main():
    executed = run(cells)
    notebook = {
        "cells": cells, "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": sys.version.split()[0]}},
    }
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index}"
    target = ROOT / "notebooks" / "sound_signal.ipynb"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf8")
    print(f"executed {executed} code cells; wrote {target.name} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
