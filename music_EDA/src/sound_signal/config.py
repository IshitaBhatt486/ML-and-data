"""Paths and the source registry shared by every pipeline step.

Raw and processed data live OUTSIDE the repository by default because the raw
files total about 1.5 GB and this repository sits in a synced folder. Set the
SOUND_SIGNAL_DATA environment variable to put them somewhere else.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(
    os.environ.get(
        "SOUND_SIGNAL_DATA",
        Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "sound_signal",
    )
)
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"

DOCS_DIR = PROJECT_ROOT / "docs"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
SITE_DIR = PROJECT_ROOT / "site"

USER_AGENT = "sound-signal-research/1.0 (open-data EDA project)"
RANDOM_SEED = 42

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]
# Features bounded on [0, 1]; loudness (dB) and tempo (BPM) are handled apart.
UNIT_FEATURES = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
]

KAGGLE = "https://www.kaggle.com/api/v1/datasets/download/"
TIDY = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/"
TAGS = "https://github.com/tranbren/MusicTagsDataset/raw/main/"

SOURCES = {
    "spotify_tracks_genre": {
        "file": "spotify_tracks_genre.zip",
        "url": KAGGLE + "maharshipandya/-spotify-tracks-dataset",
        "page": "https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset",
        "title": "Spotify Tracks Dataset (M. Pandya)",
        "licence": "Kaggle listing: Open Database License; contents (c) original authors",
        "role": "Genre labels, popularity snapshot and audio features",
    },
    "spotify_catalogue": {
        "file": "spotify_catalogue.zip",
        "url": KAGGLE + "rodolfofigueroa/spotify-12m-songs",
        "page": "https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs",
        "title": "Spotify 1.2M+ Songs (R. Figueroa)",
        "licence": "Kaggle listing: Unknown. Used for analysis only; no rows are republished",
        "role": "Broad-catalogue baseline by release year",
    },
    "spotify_1m": {
        "file": "spotify_1m.zip",
        "url": KAGGLE + "amitanshjoshi/spotify-1million-tracks",
        "page": "https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks",
        "title": "Spotify 1 Million Tracks (A. Joshi)",
        "licence": "Kaggle listing: Open Database License; contents (c) original authors",
        "role": "Popularity across a broad 2000-2023 catalogue sample",
    },
    "charts_2017_2021": {
        "file": "charts_2017_2021.zip",
        "url": KAGGLE + "dhruvildave/spotify-charts",
        "page": "https://www.kaggle.com/datasets/dhruvildave/spotify-charts",
        "title": "Spotify Charts 2017-2021 (D. Dave)",
        "licence": "Kaggle listing: Open Database License; contents (c) original authors",
        "role": "Daily Top 200 rank and stream counts by region",
    },
    "charts_2023_2025": {
        "file": "charts_2023_2025.zip",
        "url": KAGGLE + "asaniczka/top-spotify-songs-in-73-countries-daily-updated",
        "page": "https://www.kaggle.com/datasets/asaniczka/top-spotify-songs-in-73-countries-daily-updated",
        "title": "Top Spotify Songs in 73 Countries (asaniczka)",
        "licence": "ODC Attribution License (ODC-By)",
        "role": "Daily Top 50 by country with audio features",
    },
    "billboard_weekly": {
        "file": "billboard.csv",
        "url": TIDY + "2021/2021-09-14/billboard.csv",
        "page": "https://github.com/rfordatascience/tidytuesday/tree/master/data/2021/2021-09-14",
        "title": "Billboard Hot 100 weekly history (TidyTuesday / Data.World)",
        "licence": "TidyTuesday repository CC0-1.0; underlying chart facts from Billboard",
        "role": "Mainstream chart history from 1958",
    },
    "billboard_audio": {
        "file": "billboard_audio_features.csv",
        "url": TIDY + "2021/2021-09-14/audio_features.csv",
        "page": "https://github.com/rfordatascience/tidytuesday/tree/master/data/2021/2021-09-14",
        "title": "Audio features for Billboard songs (TidyTuesday)",
        "licence": "TidyTuesday repository CC0-1.0; features originate from the Spotify API",
        "role": "Spotify IDs and audio features for Hot 100 songs",
    },
    "lastfm_tags_1": {
        "file": "lastfm_metadata.zip",
        "url": TAGS + "lastfm_metadata.zip",
        "page": "https://github.com/tranbren/MusicTagsDataset",
        "title": "Music Tags Dataset, Last.fm part 1 (tranbren)",
        "licence": "Repository README: educational, research and non-commercial use with credit",
        "role": "Human-assigned tags",
    },
    "lastfm_tags_2": {
        "file": "lastfm_metadata_2.zip",
        "url": TAGS + "lastfm_metadata_2.zip",
        "page": "https://github.com/tranbren/MusicTagsDataset",
        "title": "Music Tags Dataset, Last.fm part 2 (tranbren)",
        "licence": "Repository README: educational, research and non-commercial use with credit",
        "role": "Human-assigned tags",
    },
    "lastfm_tags_3": {
        "file": "lastfm_metadata_3.zip",
        "url": TAGS + "lastfm_metadata_3.zip",
        "page": "https://github.com/tranbren/MusicTagsDataset",
        "title": "Music Tags Dataset, Last.fm part 3 (tranbren)",
        "licence": "Repository README: educational, research and non-commercial use with credit",
        "role": "Human-assigned tags",
    },
}
