# embed_index.py
"""Build embedding index for the movie catalog."""
from __future__ import annotations
import json
import math
import time
from pathlib import Path
from typing import List, Dict

import numpy as np

from src.client import get_openai_client
from src.data_loader import load_movies, build_movie_text

# -------------------------------
# CONFIG
# -------------------------------
EMBED_MODEL = "text-embedding-3-large"  # 3072-dim embeddings
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)


# -------------------------------
# BATCH EMBEDDING
# -------------------------------
def _embed_batch(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts using OpenAI."""
    client = get_openai_client()
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in response.data]


# -------------------------------
# MAIN BUILD FUNCTION
# -------------------------------
def build_embeddings(batch_size: int = 256) -> Dict[str, str]:
    """
    Build and save embedding artifacts for all movies.
    
    Creates:
        - artifacts/movie_vectors.npy: NumPy array of embeddings
        - artifacts/movie_ids.json: List of movie IDs
        - artifacts/movie_texts.json: List of movie text descriptions
    
    Args:
        batch_size: Number of texts to embed per API call
    
    Returns:
        Dict with count, dimensions, and output path
    """
    # 1) Load + prepare text data
    movies = load_movies()
    movies_txt = build_movie_text(movies)
    texts = movies_txt["text_embed"].astype(str).tolist()
    ids = movies_txt["movieId"].astype(str).tolist()

    vectors: List[List[float]] = []
    n = len(texts)
    n_batches = math.ceil(n / batch_size)

    for i in range(n_batches):
        s = i * batch_size
        e = min((i + 1) * batch_size, n)
        batch = texts[s:e]
        print(f"Embedding batch {i+1}/{n_batches} ({s}-{e})")

        # Retry logic for robustness
        for attempt in range(3):
            try:
                vecs = _embed_batch(batch)
                vectors.extend(vecs)
                break
            except Exception as err:
                print(f"Error: {err} (attempt {attempt+1}/3)")
                time.sleep(2 * (attempt + 1))

    # 2) Save results
    vecs_np = np.asarray(vectors, dtype=np.float32)
    np.save(ARTIFACTS_DIR / "movie_vectors.npy", vecs_np)
    (ARTIFACTS_DIR / "movie_ids.json").write_text(json.dumps(ids))
    (ARTIFACTS_DIR / "movie_texts.json").write_text(json.dumps(texts))

    print(f"✅ Saved {len(ids)} embeddings to {ARTIFACTS_DIR}/")

    return {
        "count": str(len(ids)),
        "dim": str(vecs_np.shape[1] if vecs_np.size else 0),
        "path": str(ARTIFACTS_DIR / "movie_vectors.npy"),
    }


# -------------------------------
# RUNNER
# -------------------------------
if __name__ == "__main__":
    info = build_embeddings(batch_size=128)
    print("✅ Built embeddings:", info)
