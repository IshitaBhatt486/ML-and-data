"""Look up the home country of the most-charted artists on MusicBrainz.

The API allows one request per second, so this script waits between calls and
caches every raw response. A country is accepted only for an exact normalised
name match that is not ambiguous. The service can throttle heavily, so the script
can be stopped and resumed. --cached-only rebuilds the table from cached answers.
Usage: python scripts/03_musicbrainz.py [n_artists] [--cached-only]
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sound_signal.common import COUNTRY_NAMES, normalise_artist  # noqa: E402
from sound_signal.config import PROCESSED_DIR, RAW_DIR, RESULTS_DIR, USER_AGENT  # noqa: E402

ENDPOINT = "https://musicbrainz.org/ws/2/artist"
CACHE = RAW_DIR / "musicbrainz_artists.jsonl"


def chart_artists(limit: int) -> pd.DataFrame:
    early = pd.read_parquet(PROCESSED_DIR / "charts_2017_2021.parquet", columns=["song_key", "days_top50", "country"])
    early = early.loc[early["country"] != "GLOBAL"]
    early["key"] = early["song_key"].str.split("|", n=1).str[0]
    first = early.groupby("key")["days_top50"].sum()
    late = pd.read_parquet(PROCESSED_DIR / "charts_2023_2025.parquet", columns=["song_key", "artist", "country"])
    late = late.loc[late["country"] != "GLOBAL"]
    late["key"] = late["song_key"].str.split("|", n=1).str[0]
    second = late.groupby("key").size()
    names = late.drop_duplicates("key").set_index("key")["artist"]
    songs = pd.read_parquet(PROCESSED_DIR / "chart17_songs.parquet", columns=["song_key", "artist"])
    songs["key"] = songs["song_key"].str.split("|", n=1).str[0]
    songs["lead"] = songs["artist"].str.split(r"\s*,\s*", n=1, regex=True).str[0]
    names = pd.concat([names, songs.drop_duplicates("key").set_index("key")["lead"]])
    names = names[~names.index.duplicated()]
    total = first.add(second, fill_value=0).sort_values(ascending=False)
    table = total.rename("chart_days").reset_index().rename(columns={"index": "key"})
    table["name"] = table["key"].map(names)
    table["share"] = table["chart_days"] / table["chart_days"].sum()
    return table.dropna(subset=["name"]).head(limit)


def query(name: str) -> dict:
    escaped = re.sub(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)', r"\\\1", name)
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": f'artist:"{escaped}"', "limit": 8, "fmt": "json"})
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError) as error:
            print(f"  retry {name}: {error}", flush=True)
            time.sleep(15 * (attempt + 1))
    return {"artists": [], "error": True}


def resolve(key: str, response: dict) -> dict:
    exact = []
    for artist in response.get("artists", []):
        names = [artist.get("name", "")] + [a.get("name", "") for a in artist.get("aliases", []) or []]
        if int(artist.get("score", 0)) >= 90 and any(normalise_artist(n) == key for n in names):
            exact.append(artist)
    located = [a for a in exact if a.get("country")]
    if not located:
        return {"status": "no_country" if exact else "no_exact_match"}
    best = max(int(a["score"]) for a in located)
    leading = [a for a in located if int(a["score"]) >= best - 5]
    countries = {a["country"] for a in leading}
    if len(countries) > 1:
        return {"status": "ambiguous", "candidates": sorted(countries)}
    winner = leading[0]
    return {"status": "resolved", "origin": winner["country"], "mbid": winner["id"], "mb_name": winner["name"],
            "type": winner.get("type")}


def main() -> None:
    arguments = [a for a in sys.argv[1:] if not a.startswith("--")]
    offline = "--cached-only" in sys.argv
    limit = int(arguments[0]) if arguments else 300
    artists = chart_artists(limit)
    cached = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf8").splitlines():
            entry = json.loads(line)
            cached[entry["key"]] = entry["response"]
    with CACHE.open("a", encoding="utf8") as sink:
        for position, row in enumerate(artists.itertuples(), 1):
            if row.key in cached or offline:
                continue
            response = query(row.name)
            if response.get("error"):
                continue
            cached[row.key] = response
            sink.write(json.dumps({"key": row.key, "name": row.name, "response": response}) + "\n")
            sink.flush()
            if position % 50 == 0:
                print(f"{position}/{len(artists)}", flush=True)
            time.sleep(2.0)
    rows = [{"key": r.key, "name": r.name, "chart_days": r.chart_days, "share": r.share,
             **(resolve(r.key, cached[r.key]) if r.key in cached else {"status": "not_queried"})}
            for r in artists.itertuples()]
    table = pd.DataFrame(rows)
    for column in ("origin", "mbid", "mb_name", "type"):
        if column not in table:
            table[column] = None
    table["candidates"] = table.get("candidates", pd.Series(dtype=object)).astype(str)
    table.to_parquet(PROCESSED_DIR / "artist_origin.parquet", index=False)
    resolved = table["status"] == "resolved"
    summary = {
        "source": "MusicBrainz web service, artist search (core data CC0)",
        "artists_requested": len(table), "queried_now": int((table["status"] != "not_queried").sum()), "status_counts": table["status"].value_counts().to_dict(),
        "resolved_share_of_all_chart_days_pct": round(float(table.loc[resolved, "share"].sum() * 100), 2),
        "queried_share_of_all_chart_days_pct": round(float(table.loc[table["status"] != "not_queried", "share"].sum() * 100), 2),
        "origin_outside_study_markets": sorted(set(table.loc[resolved, "origin"]) - set(COUNTRY_NAMES)),
    }
    (RESULTS_DIR / "musicbrainz_quality.json").write_text(json.dumps(summary, indent=2))
    print(summary, "\nDONE", flush=True)


if __name__ == "__main__":
    main()
