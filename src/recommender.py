# src/recommender.py
"""Semantic movie retrieval using FAISS and OpenAI embeddings."""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np

from src.client import get_openai_client

ART_DIR = Path("artifacts")
VEC_PATH = ART_DIR / "movie_vectors.npy"
IDS_PATH = ART_DIR / "movie_ids.json"
TXT_PATH = ART_DIR / "movie_texts.json"

EMBED_MODEL = "text-embedding-3-large"

# --- Module-level caches (load once) ---
_IDS: List[str] | None = None
_TXT: List[str] | None = None
_VEC: np.ndarray | None = None
_INDEX: faiss.IndexFlatIP | None = None  # Inner product index for cosine similarity


def _load_artifacts() -> tuple[List[str], List[str], np.ndarray]:
    """Load ids, texts, and L2-normalized vectors from disk (cached)."""
    global _IDS, _TXT, _VEC
    if _IDS is None or _TXT is None or _VEC is None:
        if not (VEC_PATH.exists() and IDS_PATH.exists() and TXT_PATH.exists()):
            raise FileNotFoundError(
                "Artifacts missing. Run `python embed_index.py` first to create "
                "artifacts/movie_vectors.npy, movie_ids.json, movie_texts.json"
            )
        _IDS = json.loads(IDS_PATH.read_text())
        _TXT = json.loads(TXT_PATH.read_text())
        vecs = np.load(VEC_PATH).astype(np.float32)
        # L2-normalize for cosine similarity via inner product
        faiss.normalize_L2(vecs)
        _VEC = vecs
    return _IDS, _TXT, _VEC


def _get_index(vecs_norm: np.ndarray) -> faiss.IndexFlatIP:
    """Build or return cached FAISS index for inner product (cosine) search."""
    global _INDEX
    if _INDEX is None:
        dim = vecs_norm.shape[1]
        _INDEX = faiss.IndexFlatIP(dim)  # Inner product on normalized vecs = cosine
        _INDEX.add(vecs_norm)
    return _INDEX


def _embed_query(text: str) -> np.ndarray:
    """Embed and L2-normalize a single query string."""
    client = get_openai_client()
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    q = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(q)
    return q


def recommend_movies(query: str, k: int = 5) -> List[Dict]:
    """
    Return top-k recommendations as a list of dicts:
    [{movieId, text, similarity}]
    """
    if not query or not query.strip():
        return []

    ids, texts, vecs_norm = _load_artifacts()
    index = _get_index(vecs_norm)
    q = _embed_query(query.strip())

    # FAISS returns (similarities, indices) for IndexFlatIP
    k_actual = min(k, len(ids))
    similarities, indices = index.search(q, k_actual)
    similarities = similarities[0]  # shape (k,)
    indices = indices[0]            # shape (k,)

    results: List[Dict] = []
    for sim, idx in zip(similarities, indices):
        results.append({
            "movieId": ids[idx],
            "text": texts[idx],
            "similarity": float(sim),
        })
    return results
