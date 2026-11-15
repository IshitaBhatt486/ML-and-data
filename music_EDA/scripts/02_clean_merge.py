"""Clean every raw source and merge them. Raw files are only ever read.

Each source runs in its own process and is streamed in chunks, so the pipeline
works on a machine with little free memory. Every exclusion is counted and
written to outputs/results/data_quality.json.

Usage: python scripts/02_clean_merge.py [stage ...]
"""
from __future__ import annotations

import gc
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sound_signal.common import NAME_TO_ISO, cached_map, first_artist, song_key  # noqa: E402
from sound_signal.config import (  # noqa: E402
    AUDIO_FEATURES, PROCESSED_DIR, RAW_DIR, RESULTS_DIR, SOURCES, UNIT_FEATURES,
)

FEATURE_COLUMNS = AUDIO_FEATURES + ["duration_ms", "key", "mode", "time_signature"]
FLOAT32 = {c: "float32" for c in FEATURE_COLUMNS}
QUALITY_DIR = PROCESSED_DIR / "_quality"
TRACK_URL = r"track/([A-Za-z0-9]{22})"


def raw_csv(key: str, **kwargs):
    path = RAW_DIR / SOURCES[key]["file"]
    if path.suffix == ".zip":
        archive = zipfile.ZipFile(path)
        return pd.read_csv(archive.open(archive.namelist()[0]), **kwargs)
    return pd.read_csv(path, **kwargs)


def invalid_audio_mask(df: pd.DataFrame) -> pd.Series:
    """Rows whose audio features cannot describe a piece of music."""
    missing = df[AUDIO_FEATURES].isna().any(axis=1)
    out_of_range = ~df[UNIT_FEATURES].apply(lambda s: s.between(0, 1)).all(axis=1)
    no_tempo = df["tempo"].fillna(0) <= 0
    all_zero = (df[["danceability", "energy", "valence"]].fillna(0) == 0).all(axis=1)
    bad_length = ~df["duration_ms"].between(30_000, 900_000)
    return missing | out_of_range | no_tempo | all_zero | bad_length


class Tally:
    """Accumulates row counts and missing values across chunks."""

    def __init__(self) -> None:
        self.rows = 0
        self.missing: pd.Series | None = None
        self.columns: list[str] = []
        self.invalid = 0

    def add(self, df: pd.DataFrame) -> None:
        self.rows += len(df)
        self.columns = list(map(str, df.columns))
        counts = df.isna().sum()
        self.missing = counts if self.missing is None else self.missing.add(counts, fill_value=0)

    def summary(self) -> dict:
        return {
            "rows_raw": int(self.rows),
            "columns_used": self.columns,
            "missing_pct": {
                str(k): round(float(v) / self.rows * 100, 3)
                for k, v in self.missing.items() if v > 0
            },
            "invalid_audio_rows": int(self.invalid),
        }


class Sink:
    def __init__(self, path: Path) -> None:
        self.path, self.writer, self.rows = path, None, 0

    def write(self, df: pd.DataFrame) -> None:
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema)
        self.writer.write_table(table.cast(self.writer.schema))
        self.rows += len(df)

    def close(self) -> None:
        self.writer.close()


def save_quality(stage: str, payload: dict) -> None:
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    (QUALITY_DIR / f"{stage}.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"[{stage}] {json.dumps(payload, default=str)[:600]}", flush=True)


def stage_genre() -> None:
    df = raw_csv("spotify_tracks_genre", dtype=FLOAT32).drop(columns=["Unnamed: 0"])
    tally = Tally()
    tally.add(df)
    bad = invalid_audio_mask(df)
    tally.invalid = int(bad.sum())
    duplicates = int(df["track_id"].notna().sum() - df["track_id"].nunique())
    # One track may sit under several genres, so the unit here is track x genre.
    df = df.loc[~bad & df["artists"].notna()].drop_duplicates(["track_id", "track_genre"]).copy()
    df["artist"] = cached_map(df["artists"], first_artist)
    df["song_key"] = song_key(df["artists"], df["track_name"])
    df = df.rename(columns={"track_genre": "genre", "track_name": "name"})
    keep = ["track_id", "song_key", "name", "artist", "genre", "popularity", "explicit"]
    df[keep + FEATURE_COLUMNS].to_parquet(PROCESSED_DIR / "genre_tracks.parquet", index=False)
    save_quality("spotify_tracks_genre", {
        **tally.summary(), "repeated_track_id_rows": duplicates, "rows_clean": len(df),
        "unique_tracks_clean": int(df["track_id"].nunique()), "genres": int(df["genre"].nunique()),
        "tracks_with_multiple_genres": int((df.groupby("track_id")["genre"].size() > 1).sum()),
    })


def stage_catalogue() -> None:
    columns = ["id", "name", "artists", "explicit", "year"] + FEATURE_COLUMNS
    sink, tally, seen = Sink(PROCESSED_DIR / "catalogue.parquet"), Tally(), set()
    bad_year = duplicates = 0
    for chunk in raw_csv("spotify_catalogue", usecols=columns, dtype=FLOAT32, chunksize=200_000):
        chunk = chunk.rename(columns={"id": "track_id"})
        tally.add(chunk)
        implausible = ~chunk["year"].between(1920, 2020)
        bad_year += int(implausible.sum())
        bad = invalid_audio_mask(chunk)
        tally.invalid += int(bad.sum())
        chunk = chunk.loc[~bad & ~implausible].drop_duplicates("track_id")
        fresh = ~chunk["track_id"].isin(seen)
        duplicates += int((~fresh).sum())
        chunk = chunk.loc[fresh].copy()
        seen.update(chunk["track_id"].tolist())
        chunk["artist"] = cached_map(chunk["artists"], first_artist)
        chunk["song_key"] = song_key(chunk["artist"], chunk["name"])
        chunk["year"] = chunk["year"].astype("int16")
        sink.write(chunk.drop(columns="artists"))
    sink.close()
    save_quality("spotify_catalogue", {
        **tally.summary(), "implausible_year_rows": bad_year,
        "repeated_track_id_rows": duplicates, "rows_clean": sink.rows,
    })


def stage_pop() -> None:
    sink, tally, seen = Sink(PROCESSED_DIR / "popularity_1m.parquet"), Tally(), set()
    duplicates = zero = 0
    columns = ["artist_name", "track_name", "track_id", "popularity", "year", "genre"] + FEATURE_COLUMNS
    for chunk in raw_csv("spotify_1m", usecols=columns, dtype=FLOAT32, chunksize=200_000):
        tally.add(chunk)
        bad = invalid_audio_mask(chunk) | chunk["artist_name"].isna() | chunk["track_name"].isna()
        tally.invalid += int(bad.sum())
        chunk = chunk.loc[~bad].drop_duplicates("track_id")
        fresh = ~chunk["track_id"].isin(seen)
        duplicates += int((~fresh).sum())
        chunk = chunk.loc[fresh].rename(columns={"artist_name": "artist", "track_name": "name"}).copy()
        seen.update(chunk["track_id"].tolist())
        chunk["song_key"] = song_key(chunk["artist"], chunk["name"])
        zero += int((chunk["popularity"] == 0).sum())
        chunk["popularity"] = chunk["popularity"].astype("int16")
        chunk["year"] = chunk["year"].astype("int16")
        sink.write(chunk)
    sink.close()
    save_quality("spotify_1m", {
        **tally.summary(), "repeated_track_id_rows": duplicates, "rows_clean": sink.rows,
        "zero_popularity_pct": round(zero / sink.rows * 100, 2),
    })


def stage_billboard() -> None:
    # Both files hold non-UTF-8 bytes in the same places, so song_id stays consistent.
    columns = ["week_id", "week_position", "song", "performer", "song_id", "instance"]
    weeks = raw_csv("billboard_weekly", encoding="latin-1", usecols=columns)
    audio = raw_csv("billboard_audio", encoding="latin-1", dtype={
        c: "float32" for c in AUDIO_FEATURES + ["key", "mode", "time_signature"]
    }).rename(columns={
        "spotify_track_id": "track_id", "spotify_track_duration_ms": "duration_ms",
        "spotify_track_popularity": "popularity", "spotify_track_explicit": "explicit",
    })
    weekly, audio_tally = Tally(), Tally()
    weekly.add(weeks)
    audio_tally.add(audio)
    weeks["week"] = pd.to_datetime(weeks["week_id"], format="%m/%d/%Y")
    before = len(weeks)
    weeks = weeks.drop_duplicates(["week", "week_position"])
    repeated_slots = before - len(weeks)
    repeated_songs = int(audio["song_id"].duplicated().sum())
    no_id = int(audio["track_id"].isna().sum())
    audio = audio.dropna(subset=["track_id"])
    bad = invalid_audio_mask(audio)
    audio_tally.invalid = int(bad.sum())
    audio = audio.loc[~bad].drop_duplicates("song_id").copy()
    audio["song_key"] = song_key(audio["performer"], audio["song"])
    songs = weeks.groupby("song_id").agg(
        song=("song", "first"), performer=("performer", "first"), first_week=("week", "min"),
        last_week=("week", "max"), weeks=("week", "nunique"), peak=("week_position", "min"),
        mean_position=("week_position", "mean"), runs=("instance", "max"),
    ).reset_index()
    songs["debut_year"] = songs["first_week"].dt.year
    keep = ["song_id", "track_id", "song_key", "spotify_genre", "popularity", "explicit"] + FEATURE_COLUMNS
    songs = songs.merge(audio[keep], on="song_id", how="left")
    matched = songs["track_id"].notna()
    weeks = weeks[["week", "week_position", "song_id"]]
    slot_share = float(weeks["song_id"].isin(songs.loc[matched, "song_id"]).mean())
    weeks.to_parquet(PROCESSED_DIR / "billboard_weeks.parquet", index=False)
    songs.to_parquet(PROCESSED_DIR / "billboard_songs.parquet", index=False)
    save_quality("billboard", {
        "weekly": {
            **weekly.summary(), "repeated_week_position_rows": int(repeated_slots),
            "first_week": str(weeks["week"].min().date()), "last_week": str(weeks["week"].max().date()),
            "unique_songs": int(len(songs)),
        },
        "audio": {
            **audio_tally.summary(), "repeated_song_id_rows": repeated_songs,
            "rows_without_spotify_id": no_id, "rows_clean": int(len(audio)),
        },
        "join": {
            "key": "song_id supplied by the source", "songs": int(len(songs)),
            "songs_matched": int(matched.sum()), "song_match_pct": round(float(matched.mean() * 100), 2),
            "chart_slot_match_pct": round(slot_share * 100, 2),
        },
    })


def stage_chart23() -> None:
    columns = ["spotify_id", "name", "artists", "daily_rank", "country", "snapshot_date", "popularity",
               "is_explicit", "album_release_date"] + FEATURE_COLUMNS
    sink, tally, seen = Sink(PROCESSED_DIR / "charts_2023_2025.parquet"), Tally(), set()
    repeated = 0
    invalid_by_month: dict[str, list[int]] = {}
    countries: set[str] = set()
    # Country codes stay literal text. A blank country is the Global chart.
    reader = raw_csv("charts_2023_2025", usecols=columns, keep_default_na=False,
                     na_values={c: [""] for c in FEATURE_COLUMNS}, dtype=FLOAT32, chunksize=250_000)
    for chunk in reader:
        chunk["country"] = chunk["country"].replace("", "GLOBAL")
        chunk["date"] = pd.to_datetime(chunk["snapshot_date"])
        chunk = chunk.drop(columns="snapshot_date")
        tally.add(chunk)
        day = (chunk["date"] - pd.Timestamp("2020-01-01")).dt.days.to_numpy()
        slot = [f"{d}{c}{r}" for d, c, r in zip(day, chunk["country"], chunk["daily_rank"])]
        fresh = np.fromiter((s not in seen for s in slot), bool, len(slot))
        seen.update(slot)
        repeated += int((~fresh).sum())
        chunk = chunk.loc[fresh].rename(columns={
            "spotify_id": "track_id", "daily_rank": "rank", "is_explicit": "explicit"})
        chunk = chunk.drop_duplicates(["date", "country", "rank"]).copy()
        invalid = invalid_audio_mask(chunk)
        tally.invalid += int(invalid.sum())
        months = chunk["date"].dt.strftime("%Y-%m")
        for month, values in invalid.groupby(months):
            entry = invalid_by_month.setdefault(month, [0, 0])
            entry[0] += int(values.sum())
            entry[1] += len(values)
        chunk["audio_valid"] = ~invalid
        chunk["song_key"] = song_key(chunk["artists"], chunk["name"])
        chunk["artist"] = cached_map(chunk["artists"], first_artist)
        chunk["rank"] = chunk["rank"].astype("int16")
        chunk["popularity"] = chunk["popularity"].astype("int16")
        countries.update(chunk["country"].unique())
        sink.write(chunk)
    sink.close()
    dates = pq.read_table(PROCESSED_DIR / "charts_2023_2025.parquet", columns=["date"])["date"].to_pandas()
    save_quality("charts_2023_2025", {
        **tally.summary(), "repeated_slot_rows": repeated, "rows_clean": sink.rows,
        "countries": len(countries - {"GLOBAL"}),
        "first_date": str(dates.min().date()), "last_date": str(dates.max().date()),
        "snapshot_days": int(dates.nunique()),
        "invalid_audio_pct_by_month": {
            m: round(a / b * 100, 2) for m, (a, b) in sorted(invalid_by_month.items()) if a
        },
    })


def stage_chart17() -> None:
    """Stream the 3.5 GB CSV from inside its zip, keeping only the Top 200 chart.

    Partial results are held as integer codes and combined with numpy, which keeps
    memory low and needs no optional pyarrow compute components.
    """
    parts = PROCESSED_DIR / "_chart17_parts"
    shutil.rmtree(parts, ignore_errors=True)
    parts.mkdir(parents=True)
    columns = ["title", "rank", "date", "artist", "url", "region", "chart", "streams"]
    totals = {"rows_raw": 0, "rows_top200": 0, "repeated_slots": 0, "rows_without_song_key": 0}
    keys, labels, song_ids, region_ids, track_ids = {}, {}, {}, {}, {}
    epoch = pd.Timestamp("2017-01-01")
    daily = Sink(PROCESSED_DIR / "global_daily_2017_2021.parquet")
    id_sink, group_sink = Sink(parts / "ids.parquet"), Sink(parts / "groups.parquet")
    code = lambda table, values: [table.setdefault(v, len(table)) for v in values]
    reader = raw_csv("charts_2017_2021", usecols=columns, chunksize=500_000,
                     dtype={"rank": "int32", "streams": "float64"})
    for chunk in reader:
        totals["rows_raw"] += len(chunk)
        chunk = chunk.loc[chunk["chart"] == "top200"].drop(columns="chart")
        totals["rows_top200"] += len(chunk)
        if chunk.empty:
            continue
        before = len(chunk)
        chunk = chunk.drop_duplicates(["date", "region", "rank"]).copy()
        totals["repeated_slots"] += before - len(chunk)
        pairs = chunk[["artist", "title"]].drop_duplicates()
        pairs = pairs.loc[[(a, t) not in keys for a, t in zip(pairs["artist"], pairs["title"])]]
        if len(pairs):
            for artist, title, key in zip(pairs["artist"], pairs["title"], song_key(pairs["artist"], pairs["title"])):
                keys[(artist, title)] = key
                if isinstance(key, str):
                    labels.setdefault(key, (title, artist))
        chunk["song_key"] = [keys[p] for p in zip(chunk["artist"], chunk["title"])]
        missing = chunk["song_key"].isna()
        totals["rows_without_song_key"] += int(missing.sum())
        chunk = chunk.loc[~missing].copy()
        chunk["date"] = pd.to_datetime(chunk["date"])
        chunk["track_id"] = chunk["url"].str.extract(TRACK_URL, expand=False)
        chunk["quarter"] = (chunk["date"].dt.year * 10 + chunk["date"].dt.quarter).astype("int32")
        chunk["days_top50"] = (chunk["rank"] <= 50).astype("int32")
        chunk["day"] = (chunk["date"] - epoch).dt.days.astype("int32")
        chunk["sid"] = np.array(code(song_ids, chunk["song_key"]), dtype="int32")
        chunk["rid"] = np.array(code(region_ids, chunk["region"]), dtype="int16")
        group_sink.write(chunk.groupby(["sid", "rid", "quarter"], sort=False).agg(
            days=("day", "size"), days_top50=("days_top50", "sum"), best_rank=("rank", "min"),
            streams=("streams", "sum"), first=("day", "min"), last=("day", "max"),
        ).reset_index())
        known = chunk.dropna(subset=["track_id"]).copy()
        known["tid"] = np.array(code(track_ids, known["track_id"]), dtype="int32")
        id_sink.write(known.groupby(["sid", "tid"], sort=False).agg(streams=("streams", "sum")).reset_index())
        world = chunk.loc[chunk["region"] == "Global", ["date", "rank", "track_id", "song_key", "streams"]]
        if len(world):
            daily.write(world)
        print(f"  charts 2017-21: {totals['rows_raw']:,} rows", flush=True)
        del chunk, known
        gc.collect()
    daily.close(), id_sink.close(), group_sink.close()
    del keys
    gc.collect()
    song_names = np.array(list(song_ids), dtype=object)
    region_names = np.array(list(region_ids), dtype=object)
    track_names = np.array(list(track_ids), dtype=object)

    g = pd.read_parquet(parts / "groups.parquet")
    quarter_codes, quarter_values = pd.factorize(g["quarter"], sort=True)
    combined = g["sid"].to_numpy(np.int64) * 100_000 + g["rid"].to_numpy(np.int64) * 100 + quarter_codes
    unique, inverse = np.unique(combined, return_inverse=True)
    size = len(unique)
    best = np.full(size, np.iinfo(np.int32).max, dtype=np.int32)
    first = best.copy()
    last = np.full(size, np.iinfo(np.int32).min, dtype=np.int32)
    np.minimum.at(best, inverse, g["best_rank"].to_numpy())
    np.minimum.at(first, inverse, g["first"].to_numpy())
    np.maximum.at(last, inverse, g["last"].to_numpy())
    frame = pd.DataFrame({
        "song_key": song_names[unique // 100_000],
        "region": region_names[(unique % 100_000) // 100],
        "quarter": np.asarray(quarter_values)[unique % 100].astype("int32"),
        "days": np.bincount(inverse, g["days"].to_numpy(dtype=float), size).astype("int32"),
        "days_top50": np.bincount(inverse, g["days_top50"].to_numpy(dtype=float), size).astype("int32"),
        "best_rank": best,
        "streams": np.bincount(inverse, np.nan_to_num(g["streams"].to_numpy(dtype=float)), size),
        "first": epoch + pd.to_timedelta(first, unit="D"),
        "last": epoch + pd.to_timedelta(last, unit="D"),
    })
    del g, combined, inverse
    gc.collect()
    frame["song_key"] = frame["song_key"].astype("str")
    frame["region"] = frame["region"].astype("str")
    unknown = sorted(set(frame["region"]) - set(NAME_TO_ISO))
    if unknown:
        raise SystemExit(f"Add these regions to COUNTRY_NAMES: {unknown}")
    frame["country"] = frame["region"].map(NAME_TO_ISO)
    frame["year"] = (frame["quarter"] // 10).astype("int16")
    frame.drop(columns="region").to_parquet(PROCESSED_DIR / "charts_2017_2021.parquet", index=False)

    i = pd.read_parquet(parts / "ids.parquet")
    combined = i["sid"].to_numpy(np.int64) * 10_000_000 + i["tid"].to_numpy(np.int64)
    unique, inverse = np.unique(combined, return_inverse=True)
    streams = np.bincount(inverse, np.nan_to_num(i["streams"].to_numpy(dtype=float)), len(unique))
    sid, tid = unique // 10_000_000, unique % 10_000_000
    order = np.argsort(-streams, kind="stable")
    _, chosen = np.unique(sid[order], return_index=True)
    picked = order[chosen]
    ids = pd.DataFrame({"song_key": song_names[sid[picked]], "track_id": track_names[tid[picked]], "streams": streams[picked]})
    ids["song_key"] = ids["song_key"].astype("str")
    ids["track_id"] = ids["track_id"].astype("str")
    ids["name"] = ids["song_key"].map(lambda k: labels[k][0])
    ids["artist"] = ids["song_key"].map(lambda k: labels[k][1])
    ids.to_parquet(PROCESSED_DIR / "chart17_songs.parquet", index=False)
    save_quality("charts_2017_2021", {
        **{k: int(v) for k, v in totals.items()}, "columns_used": columns,
        "regions": int(frame["country"].nunique() - 1),
        "first_date": str(frame["first"].min().date()), "last_date": str(frame["last"].max().date()),
        "song_region_quarter_rows": int(len(frame)), "unique_songs": int(frame["song_key"].nunique()),
    })
    shutil.rmtree(parts, ignore_errors=True)


def stage_tags() -> None:
    tally = Tally()
    frames = []
    for key in ("lastfm_tags_1", "lastfm_tags_2", "lastfm_tags_3"):
        frame = raw_csv(key).rename(columns={
            "Artist": "artist", "SongTitle": "title", "Year": "tag_year", "Last.fm Tags": "tags"})
        tally.add(frame)
        frames.append(frame)
    tags = pd.concat(frames, ignore_index=True).dropna(subset=["artist", "title", "tags"])
    del frames
    tags["tag_year"] = pd.to_numeric(tags["tag_year"], errors="coerce")
    bad_year = int(tags["tag_year"].isna().sum())
    tags = tags.dropna(subset=["tag_year"])
    tags["tag_year"] = tags["tag_year"].astype("int16")
    before = len(tags)
    tags = tags.drop_duplicates(["artist", "title", "tag_year"]).copy()
    repeated = before - len(tags)
    tags["song_key"] = song_key(tags["artist"], tags["title"])
    tags = tags.dropna(subset=["song_key"])
    # The tag files were built from both Spotify catalogues, so both are candidates.
    # Popularity-sample IDs sort first, so a matched song can be linked to its popularity.
    wanted = set(tags["song_key"])
    sides = []
    for rank, name in enumerate(("popularity_1m", "catalogue")):
        part = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet",
                               columns=["track_id", "song_key", "artist", "year", "valence", "energy"])
        sides.append(part.loc[part["song_key"].isin(wanted)].assign(source_rank=rank))
    side = pd.concat(sides, ignore_index=True).sort_values("source_rank").drop_duplicates("track_id")
    del sides, wanted
    joined = tags[["song_key", "tag_year", "tags"]].merge(side, on="song_key")
    joined = joined.loc[(joined["year"].astype(int) - joined["tag_year"]).abs() <= 1]
    # Several catalogue rows can match. Accept them only if they sound like one recording.
    spread = joined.groupby(["song_key", "tag_year"]).agg(
        n=("track_id", "size"), v_max=("valence", "max"), v_min=("valence", "min"),
        e_max=("energy", "max"), e_min=("energy", "min"))
    spread["v"], spread["e"] = spread["v_max"] - spread["v_min"], spread["e_max"] - spread["e_min"]
    accepted = spread.loc[(spread["n"] == 1) | ((spread["v"] <= 0.05) & (spread["e"] <= 0.05))]
    joined = joined.join(accepted[["n"]], on=["song_key", "tag_year"], how="inner")
    joined = joined.sort_values(["source_rank", "track_id"]).drop_duplicates(["song_key", "tag_year"]).drop_duplicates("track_id")
    columns = ["track_id", "song_key", "artist", "tag_year", "valence", "energy", "tags"]
    joined[columns].to_parquet(PROCESSED_DIR / "tags.parquet", index=False)
    save_quality("lastfm_tags", {
        **tally.summary(), "non_numeric_year_rows": bad_year,
        "repeated_rows": int(repeated), "rows_with_key": int(len(tags)),
        "join": {
            "key": "normalised lead artist + normalised title, release year within 1 year; "
                   "multiple candidates accepted only if valence and energy agree within 0.05",
            "matched_tracks": int(len(joined)),
            "match_pct": round(len(joined) / len(tags) * 100, 2),
            "rejected_ambiguous_keys": int(len(spread) - len(accepted)),
        },
    })


def stage_merge() -> None:
    columns = ["track_id", "song_key"] + FEATURE_COLUMNS
    def load(name: str, **kwargs) -> pd.DataFrame:
        return pd.read_parquet(PROCESSED_DIR / f"{name}.parquet", columns=columns, **kwargs)

    chart = pd.read_parquet(PROCESSED_DIR / "charts_2023_2025.parquet", columns=columns + ["audio_valid"])
    chart = chart.loc[chart["audio_valid"]].drop(columns="audio_valid").drop_duplicates("track_id")
    billboard = load("billboard_songs").dropna(subset=["track_id"]).drop_duplicates("track_id")
    tables = {
        "charts_2023_2025": chart,
        "spotify_tracks_genre": load("genre_tracks").drop_duplicates("track_id"),
        "spotify_1m": load("popularity_1m"),
        "billboard_audio": billboard,
        "spotify_catalogue": load("catalogue"),
    }
    agreement, overlap = {}, {}
    names = list(tables)
    probes = ["energy", "valence", "tempo", "loudness"]
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            pair = tables[left][["track_id"] + probes].merge(
                tables[right][["track_id"] + probes], on="track_id", suffixes=("_l", "_r"))
            overlap[f"{left} / {right}"] = int(len(pair))
            if len(pair) >= 100:
                agreement[f"{left} / {right}"] = {
                    "shared_ids": int(len(pair)),
                    **{p: round(float(pair[p + "_l"].corr(pair[p + "_r"])), 4) for p in probes},
                }
    # Each ID takes features from the first source listing it. Popularity is NOT pooled,
    # because each source measured it on a different date.
    tracks = pd.concat([t.assign(audio_source=n) for n, t in tables.items()], ignore_index=True)
    del tables, chart, billboard
    gc.collect()
    tracks = tracks.drop_duplicates("track_id").reset_index(drop=True)
    tracks.to_parquet(PROCESSED_DIR / "tracks.parquet", index=False)

    # Charts join on Spotify ID first. Spotify can issue different IDs for one song in
    # different markets, so the fallback is lead artist + title, and only when every
    # candidate recording has near-identical features.
    keyed = tracks.dropna(subset=["song_key"])[["song_key", "track_id", "valence", "energy"]]
    grouped = keyed.groupby("song_key", sort=False)
    by_key = pd.DataFrame({
        "n": grouped["track_id"].size(), "v_max": grouped["valence"].max(), "v_min": grouped["valence"].min(),
        "e_max": grouped["energy"].max(), "e_min": grouped["energy"].min(), "key_id": grouped["track_id"].first(),
    })
    safe = by_key.loc[(by_key["n"] == 1) | (
        ((by_key["v_max"] - by_key["v_min"]) <= 0.05) & ((by_key["e_max"] - by_key["e_min"]) <= 0.05))]
    songs = pd.read_parquet(PROCESSED_DIR / "chart17_songs.parquet")
    songs["direct"] = songs["track_id"].isin(set(tracks["track_id"]))
    songs = songs.join(safe["key_id"], on="song_key")
    songs["feature_id"] = songs["track_id"].where(songs["direct"], songs["key_id"])
    songs["match_tier"] = np.select(
        [songs["direct"], songs["key_id"].notna()], ["spotify_id", "artist_title_validated"], "unmatched")
    songs.drop(columns=["direct", "key_id"]).to_parquet(PROCESSED_DIR / "chart17_songs.parquet", index=False)
    share = songs.groupby("match_tier")["streams"].sum()
    save_quality("merge", {
        "unique_spotify_ids": int(len(tracks)),
        "ids_by_feature_source": {k: int(v) for k, v in tracks["audio_source"].value_counts().items()},
        "shared_ids_between_sources": overlap,
        "feature_agreement_pearson": agreement,
        "charts_2017_2021_to_audio": {
            "songs_by_tier": {k: int(v) for k, v in songs["match_tier"].value_counts().items()},
            "stream_share_pct_by_tier": {k: round(float(v / share.sum() * 100), 2) for k, v in share.items()},
        },
    })


STAGES = {
    "genre": stage_genre, "catalogue": stage_catalogue, "pop": stage_pop, "billboard": stage_billboard,
    "chart23": stage_chart23, "chart17": stage_chart17, "tags": stage_tags, "merge": stage_merge,
}


def combine_quality() -> None:
    combined = {p.stem: json.loads(p.read_text()) for p in sorted(QUALITY_DIR.glob("*.json"))}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "data_quality.json").write_text(json.dumps(combined, indent=2))


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    requested = sys.argv[1:]
    if len(requested) == 1 and requested[0] in STAGES:
        STAGES[requested[0]]()
        return
    for stage in requested or STAGES:
        print(f"=== {stage}", flush=True)
        subprocess.run([sys.executable, __file__, stage], check=True)
    combine_quality()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
