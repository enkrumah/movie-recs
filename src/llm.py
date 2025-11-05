# src/llm.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple
import os

from dotenv import load_dotenv
from openai import OpenAI

DEFAULT_MODEL = "gpt-4o-mini"
_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        # Try .env (local) and st.secrets (Streamlit Cloud)
        from dotenv import load_dotenv
        import streamlit as st
        load_dotenv(Path(".env"))
        api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment or Streamlit secrets")
        _client = OpenAI(api_key=api_key)
    return _client


def summarize_recommendations(
    query: str,
    recs: List[Dict],
    model: str = DEFAULT_MODEL,
    include_per_movie: bool = False,
) -> Tuple[str, List[str]]:
    if not recs:
        return ("", [])

    movies_compact = [r["text"] for r in recs]

    system = (
        "You are a concise movie concierge. Given a user mood/constraints and a list of recommended movies, "
        "explain the common vibe in 2–3 sentences. Be specific (themes, tone, pacing) and avoid spoilers."
    )
    user = (
        f"User request: {query}\n\n"
        "Recommended movies:\n- " + "\n- ".join(movies_compact) + "\n\n"
        "Write a brief summary for why these picks fit together."
    )

    client = _get_client()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.5,
            max_tokens=220,
        )
        summary = resp.choices[0].message.content.strip()
    except Exception as e:
        summary = f"(LLM summary unavailable: {e})"

    per_movie = []
    if include_per_movie:
        user2 = (
            "For each movie, provide ONE short line why it matches the user's request "
            "(no spoilers, <18 words each).\n\n"
            f"User request: {query}\n\nMovies:\n- " + "\n- ".join(movies_compact)
        )
        try:
            r2 = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": "Be brief and helpful."},
                          {"role": "user", "content": user2}],
                temperature=0.4,
                max_tokens=300,
            )
            raw = r2.choices[0].message.content
            per_movie = [ln.strip("-• ").strip() for ln in raw.splitlines() if ln.strip()]
            per_movie = per_movie[: len(recs)]
        except Exception as e:
            per_movie = [f"(rationale unavailable: {e})"] * len(recs)

    return summary, per_movie
