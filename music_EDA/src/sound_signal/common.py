"""Text normalisation, geography lookups and small shared helpers."""
from __future__ import annotations

import ast
import re
import unicodedata

import numpy as np
import pandas as pd

COUNTRY_NAMES = {
    "AD": "Andorra", "AE": "United Arab Emirates", "AR": "Argentina", "AT": "Austria",
    "AU": "Australia", "BE": "Belgium", "BG": "Bulgaria", "BO": "Bolivia", "BR": "Brazil",
    "BY": "Belarus", "CA": "Canada", "CH": "Switzerland", "CL": "Chile", "CO": "Colombia",
    "CR": "Costa Rica", "CY": "Cyprus", "CZ": "Czech Republic", "DE": "Germany",
    "DK": "Denmark", "DO": "Dominican Republic", "EC": "Ecuador", "EE": "Estonia",
    "EG": "Egypt", "ES": "Spain", "FI": "Finland", "FR": "France", "GB": "United Kingdom",
    "GR": "Greece", "GT": "Guatemala", "HK": "Hong Kong", "HN": "Honduras", "HU": "Hungary",
    "ID": "Indonesia", "IE": "Ireland", "IL": "Israel", "IN": "India", "IS": "Iceland",
    "IT": "Italy", "JP": "Japan", "KR": "South Korea", "KZ": "Kazakhstan", "LI": "Liechtenstein",
    "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia", "MA": "Morocco", "MC": "Monaco",
    "MT": "Malta", "MX": "Mexico", "MY": "Malaysia", "NA": "Namibia", "NG": "Nigeria",
    "NI": "Nicaragua", "NL": "Netherlands", "NO": "Norway", "NZ": "New Zealand",
    "PA": "Panama", "PE": "Peru", "PH": "Philippines", "PK": "Pakistan", "PL": "Poland",
    "PT": "Portugal", "PY": "Paraguay", "RO": "Romania", "RU": "Russia", "SA": "Saudi Arabia",
    "SE": "Sweden", "SG": "Singapore", "SK": "Slovakia", "SV": "El Salvador", "TH": "Thailand",
    "TR": "Turkey", "TW": "Taiwan", "UA": "Ukraine", "US": "United States", "UY": "Uruguay",
    "VE": "Venezuela", "VN": "Vietnam", "ZA": "South Africa",
}
NAME_TO_ISO = {name: iso for iso, name in COUNTRY_NAMES.items()}
NAME_TO_ISO.update({"Czechia": "CZ", "Global": "GLOBAL", "Korea, Republic of": "KR"})

REGION_GROUPS = {
    "North America": "US CA",
    "Latin America": "AR BO BR CL CO CR DO EC GT HN MX NI PA PE PY SV UY VE",
    "Western Europe": "AD AT BE CH DE DK ES FI FR GB GR IE IS IT LI LU MC MT NL NO PT SE CY",
    "Eastern Europe": "BG BY CZ EE HU KZ LT LV PL RO RU SK TR UA",
    "Asia": "HK ID IN JP KR MY PH PK SG TH TW VN",
    "Oceania": "AU NZ",
    "Middle East & Africa": "AE EG IL MA NA NG SA ZA",
}
ISO_TO_GROUP = {iso: group for group, codes in REGION_GROUPS.items() for iso in codes.split()}

_VERSION_SUFFIX = re.compile(
    r"\s*[-(\[]\s*(?:(?:\d{4}\s+)?(?:digital(?:ly)?\s+)?remaster(?:ed)?(?:\s+(?:version|\d{4}))?"
    r"|(?:mono|stereo)(?:\s+version)?|radio edit|single version|album version)\s*[)\]]?\s*$"
)
_FEATURING = re.compile(r"\s*[(\[-]?\s*\b(?:feat|ft|featuring|with)\b\.?\s.*$")
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)


def _plain(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()


def normalise_title(title: str) -> str:
    """Remove accents, featured credits and remaster labels.

    Words such as live, acoustic and remix are kept on purpose, because those
    are usually different recordings.
    """
    text = _plain(title)
    for _ in range(2):
        text = _VERSION_SUFFIX.sub("", text)
    text = re.sub(r"\s*[(\[]\s*(?:feat|ft|featuring|with)\b.*?[)\]]", "", text)
    text = re.sub(r"\s+-\s+(?:feat|ft|featuring)\b.*$", "", text)
    return _NON_WORD.sub(" ", text).strip()


def normalise_artist(artist: str) -> str:
    text = _plain(artist)
    text = _FEATURING.sub("", text)
    text = re.sub(r"^the\s+", "", text)
    return _NON_WORD.sub(" ", text).strip()


def first_artist(value: str) -> str:
    """Return the lead artist from a list literal, or a delimited string."""
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
            return str(parsed[0]) if parsed else ""
        except (ValueError, SyntaxError):
            return text.strip("[]'\" ")
    return re.split(r"\s*(?:;|,|&|\bfeaturing\b|\bfeat\.?\b|\bft\.?\b)\s*", text, maxsplit=1)[0]


def cached_map(series: pd.Series, function) -> pd.Series:
    """Apply a string function once per unique value."""
    values = series.fillna("").astype(str)
    uniques = values.unique()
    return values.map(dict(zip(uniques, map(function, uniques))))


def song_key(artists: pd.Series, titles: pd.Series) -> pd.Series:
    """Lead artist plus title. Title alone is never used as a key."""
    lead = cached_map(cached_map(artists, first_artist), normalise_artist)
    title = cached_map(titles, normalise_title)
    key = lead + "|" + title
    return key.where((lead != "") & (title != ""))


def transform_features(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Standardise features after taming the heavily skewed ones."""
    matrix = frame[features].to_numpy(dtype=float).copy()
    for index, name in enumerate(features):
        if name in {"speechiness", "instrumentalness", "liveness", "acousticness"}:
            matrix[:, index] = np.log10(np.clip(matrix[:, index], 0, None) + 0.01)
    return (matrix - np.nanmean(matrix, axis=0)) / np.nanstd(matrix, axis=0)
