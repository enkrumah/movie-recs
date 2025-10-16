# src/embed_index.py
from __future__ import annotations
import os, json, math, time
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from .data_loader import load_movies, build_movie_text

# -------------------------------
# CONFIG
# -------------------------------
EMBED_MODEL = "text-embedding-3-large"  # 3072-dim embeddings
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)

# -------------------------------
# CLIENT
# -------------------------------
def _get_client() -> OpenAI:
    load_dotenv(Path(".env"))  # explicitly load .env
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Put it in .env")
    return OpenAI(api_key=api_key)

# -------------------------------
# BATCH EMBEDDING
# -------------------------------
def _embed_batch(client: OpenAI, texts: List[str]) -> List[List[float]]:
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in response.data]

# -------------------------------
# MAIN BUILD FUNCTION
# -------------------------------
def build_embeddings(batch_size: int = 256) -> Dict[str, str]:
    client = _get_client()

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

        # retry logic
        for attempt in range(3):
            try:
                vecs = _embed_batch(client, batch)
                vectors.extend(vecs)
                break
            except Exception as err:
                print(f"Error: {err} (attempt {attempt+1}/3)")
                time.sleep(2 * (attempt + 1))

    # 3) Save results
    vecs_np = np.asarray(vectors, dtype=np.float32)
    np.save(ARTIFACTS_DIR / "movie_vectors.npy", vecs_np)
    (ARTIFACTS_DIR / "movie_ids.json").write_text(json.dumps(ids))
    (ARTIFACTS_DIR / "movie_texts.json").write_text(json.dumps(texts))

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
