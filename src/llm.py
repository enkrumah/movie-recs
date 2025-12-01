# src/llm.py
"""LLM-powered explanations and suggestions."""
from __future__ import annotations
from typing import List, Dict, Tuple

from src.client import get_openai_client

DEFAULT_MODEL = "gpt-4o-mini"


# Re-export for backward compatibility (used by app.py for suggestions)
def _get_client():
    """Wrapper for backward compatibility with app.py imports."""
    return get_openai_client()


def summarize_recommendations(
    query: str,
    recs: List[Dict],
    model: str = DEFAULT_MODEL,
    include_per_movie: bool = False,
) -> Tuple[str, List[str]]:
    """
    Generate an LLM summary explaining why the recommendations fit the query.
    
    Args:
        query: User's search query
        recs: List of recommendation dicts with 'text' field
        model: OpenAI model to use
        include_per_movie: If True, also generate per-movie explanations
    
    Returns:
        Tuple of (summary_text, list_of_per_movie_explanations)
    """
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

    client = get_openai_client()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.5,
            max_tokens=220,
        )
        summary = resp.choices[0].message.content.strip()
    except Exception as e:
        summary = f"(LLM summary unavailable: {e})"

    per_movie: List[str] = []
    if include_per_movie:
        user2 = (
            "For each movie, provide ONE short line why it matches the user's request "
            "(no spoilers, <18 words each).\n\n"
            f"User request: {query}\n\nMovies:\n- " + "\n- ".join(movies_compact)
        )
        try:
            r2 = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Be brief and helpful."},
                    {"role": "user", "content": user2},
                ],
                temperature=0.4,
                max_tokens=300,
            )
            raw = r2.choices[0].message.content
            per_movie = [ln.strip("-• ").strip() for ln in raw.splitlines() if ln.strip()]
            per_movie = per_movie[: len(recs)]
        except Exception as e:
            per_movie = [f"(rationale unavailable: {e})"] * len(recs)

    return summary, per_movie
