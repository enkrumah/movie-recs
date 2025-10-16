# src/recommender.py
from __future__ import annotations
from pathlib import Path
import os, json
from typing import List, Dict, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

ART_DIR = Path("artifacts")
VEC_PATH = ART_DIR / "movie_vectors.npy"
IDS_PATH = ART_DIR / "movie_ids.json"
TXT_PATH = ART_DIR / "movie_texts.json"

EMBED_MODEL = "text-embedding-3-large"

# --- module-level caches so we only load/fit once ---
_IDS: List[str] | None = None
_TXT: List[str] | None = None
_VEC: np.ndarray | None = None
_NN: NearestNeighbors | None = None
_CLIENT: OpenAI | None = None


def _get_client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        load_dotenv(Path(".env"))
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in .env")
        _CLIENT = OpenAI(api_key=api_key)
    return _CLIENT


def _load_artifacts() -> Tuple[List[str], List[str], np.ndarray]:
    """Load ids, texts, and normalized vectors from disk (cached)."""
    global _IDS, _TXT, __VEC
    if _IDS is None or _TXT is None or __VEC is None:
        if not (VEC_PATH.exists() and IDS_PATH.exists() and TXT_PATH.exists()):
            raise FileNotFoundError(
                "Artifacts missing. Run embeddings build first to create "
                "artifacts/movie_vectors.npy, movie_ids.json, movie_texts.json"
            )
        _IDS = json.loads(IDS_PATH.read_text())
        _TXT = json.loads(TXT_PATH.read_text())
        vecs = np.load(VEC_PATH)
        # L2-normalize for cosine similarity
        __VEC = normalize(vecs, axis=1)
    return _IDS, _TXT, __VEC


def _fit_index(vecs_norm: np.ndarray) -> NearestNeighbors:
    """Fit a cosine kNN index (cached)."""
    global _NN
    if _NN is None:
        # cosine distance in sklearn => 0 = identical, 1 = orthogonal
        _NN = NearestNeighbors(metric="cosine", algorithm="brute")
        _NN.fit(vecs_norm)
    return _NN


def _embed_query(text: str) -> np.ndarray:
    """Embed and L2-normalize a single query string."""
    client = _get_client()
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    q = np.array(resp.data[0].embedding, dtype=np.float32)[None, :]  # shape (1, d)
    q_norm = normalize(q, axis=1)
    return q_norm


def recommend_movies(query: str, k: int = 5) -> List[Dict]:
    """
    Return top-k recommendations as a list of dicts:
    [{id, text, distance, similarity}]
    """
    if not query or not query.strip():
        return []

    ids, texts, vecs_norm = _load_artifacts()
    nn = _fit_index(vecs_norm)
    q = _embed_query(query.strip())

    distances, indices = nn.kneighbors(q, n_neighbors=min(k, len(ids)))
    distances = distances[0]  # shape (k,)
    indices = indices[0]      # shape (k,)

    results: List[Dict] = []
    for d, idx in zip(distances, indices):
        # cosine distance -> similarity
        sim = float(1.0 - d)
        results.append({
            "movieId": ids[idx],
            "text": texts[idx],        # this is your "title (year) — Genres: ..."
            "distance": float(d),
            "similarity": sim,
        })
    return results
