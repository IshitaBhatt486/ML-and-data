# SOUND / SIGNAL

A statistical study of what the measurable properties of music can and cannot tell us about popularity, emotional perception, genre identity, geography and cultural change.

**Central question:** Can we identify measurable patterns in music that explain popularity, emotional perception, genre identity and the evolution of mainstream music?

> `site/index.html` contains an interactive track explorer.

## Main findings

| Topic | Can measurable patterns explain it? | Evidence |
|---|---|---|
| Popularity | Weakly | Audio predicts about 10% of popularity in unseen tracks. The artist's other tracks predict about 63%. An independent sample gives about 4% and 52%. |
| Emotional perception | Partly | Last.fm mood tags align with energy (ε² 0.24) better than valence (ε² 0.14). Agreement on four mood quadrants is weak (κ 0.20). |
| Genre identity | Substantially | Genre explains up to 51% of rank variation in one feature. Audio alone identifies genre about 13 times better than chance, across 112 genres. |
| Evolution | Clearly | The Hot 100 gained about 1 dB of loudness per decade and lost valence and acousticness. Song length rose until the 1990s and then fell. |
| Geography | Clearly | Top 50 overlap between countries fell from 19% (2017-2021) to 8% (2023-2025). Two thirds of charting songs reached only one country's Top 50. |

The project compares three proxies: popularity, chart longevity and affectionate Last.fm tags. All findings are associations, not causes.

## Data sources and licences

| Source | Use | Licence as listed |
|---|---|---|
| [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (M. Pandya) | Genres, popularity, audio features | Open Database License; contents © original authors |
| [Spotify 1.2M+ Songs](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs) (R. Figueroa) | Catalogue baseline by release year | Unknown; used for analysis only, with no rows published |
| [Spotify 1 Million Tracks](https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks) (A. Joshi) | Popularity sample, 2000-2023 | Open Database License; contents © original authors |
| [Spotify Charts](https://www.kaggle.com/datasets/dhruvildave/spotify-charts) (D. Dave) | Daily Top 200 and streams, 2017-2021 | Open Database License; contents © original authors |
| [Top Spotify Songs in 73 Countries](https://www.kaggle.com/datasets/asaniczka/top-spotify-songs-in-73-countries-daily-updated) (asaniczka) | Daily Top 50 and audio features, Oct 2023-Jun 2025 | ODC Attribution License |
| [Billboard Hot 100 and audio features](https://github.com/rfordatascience/tidytuesday/tree/master/data/2021/2021-09-14) (TidyTuesday) | Weekly chart history, 1958-2021 | Repository CC0-1.0; chart facts from Billboard, features from Spotify |
| [Music Tags Dataset](https://github.com/tranbren/MusicTagsDataset) (tranbren) | Last.fm tags | Educational, research and non-commercial use with credit |
| [MusicBrainz](https://musicbrainz.org/) | Artist home countries | Core data CC0 |

## Cleaning and joins

- Removed: missing or out-of-range audio values, zero tempo, all-zero profiles, and tracks shorter than 30 seconds or longer than 15 minutes.
- Comedy and sleep categories and likely spoken-word tracks (speechiness above 0.66) are left out of musical analyses.
- Outliers that are valid recordings are retained and rank-based tests limit their influence.
- Sources join on Spotify track IDs first. Lead artist plus normalised title is a fallback only when all candidate recordings agree within 0.05 on valence and energy. Title alone is never used.
- Popularity scores from different sources are not pooled, because they were measured at different times.