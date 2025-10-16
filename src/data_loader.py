from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/ml-latest-small")

def load_movies() -> pd.DataFrame:
    """Load minimal movie fields for the MVP."""
    movies_path = DATA_DIR / "movies.csv"
    df = pd.read_csv(movies_path)
    df = df[["movieId", "title", "genres"]].copy()
    df["genres"] = df["genres"].fillna("(no genres listed)")
    return df

def quick_stats(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "unique_movies": df["movieId"].nunique(),
        "missing_title": int(df["title"].isna().sum()),
        "missing_genres": int(df["genres"].isna().sum()),
    }

import re
from typing import Tuple

def split_title_year(title: str) -> Tuple[str, str]:
    """
    Extracts (clean_title, year) from titles like 'Se7en (1995)'.
    If no year is found, returns ('title', '').
    """
    m = re.search(r"\s\((\d{4})\)$", str(title))
    year = m.group(1) if m else ""
    clean = re.sub(r"\s\(\d{4}\)$", "", str(title)).strip()
    return clean, year

def build_movie_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a single descriptive text column per movie for embeddings.
    Uses: clean_title + year + genres.
    """
    out = df.copy()
    tt = out["title"].apply(split_title_year)
    out["clean_title"] = tt.apply(lambda x: x[0])
    out["year"] = tt.apply(lambda x: x[1])

    # Normalize genres: 'Action|Adventure|Sci-Fi' -> 'Action, Adventure, Sci-Fi'
    out["genres_norm"] = out["genres"].fillna("").str.replace("|", ", ", regex=False)

    # Final text field we will embed
    out["text_embed"] = (
        out["clean_title"].fillna("") +
        out["year"].apply(lambda y: f" ({y})" if y else "") +
        out["genres_norm"].apply(lambda g: f" — Genres: {g}" if g else "")
    ).str.strip()

    # Keep only what we need for embedding/indexing
    return out[["movieId", "clean_title", "year", "genres_norm", "text_embed"]]
