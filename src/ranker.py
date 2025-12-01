# src/ranker.py
"""Multi-signal ranking layer for movie recommendations."""
from __future__ import annotations
import re
import json
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np


@dataclass
class RankingWeights:
    """Configurable weights for ranking signals."""
    similarity: float = 0.45      # Embedding similarity (core relevance)
    recency: float = 0.10         # Prefer newer movies
    genre_match: float = 0.15     # Genre alignment with query
    keyword_match: float = 0.10   # Lexical overlap with query
    llm_score: float = 0.20       # LLM relevance judgment
    
    def normalize(self) -> "RankingWeights":
        """Normalize weights to sum to 1.0."""
        total = (
            self.similarity + self.recency + self.genre_match + 
            self.keyword_match + self.llm_score
        )
        if total == 0:
            return RankingWeights()
        return RankingWeights(
            similarity=self.similarity / total,
            recency=self.recency / total,
            genre_match=self.genre_match / total,
            keyword_match=self.keyword_match / total,
            llm_score=self.llm_score / total,
        )
    
    @classmethod
    def fast_only(cls) -> "RankingWeights":
        """Weights without LLM (for fast ranking)."""
        return cls(
            similarity=0.50,
            recency=0.15,
            genre_match=0.20,
            keyword_match=0.15,
            llm_score=0.0,
        )


# Genre keywords for matching user queries
GENRE_KEYWORDS = {
    "action": ["action", "fight", "battle", "explosive", "chase"],
    "adventure": ["adventure", "quest", "journey", "epic"],
    "animation": ["animated", "animation", "cartoon", "pixar", "disney"],
    "comedy": ["comedy", "funny", "hilarious", "laugh", "humor", "comedic"],
    "crime": ["crime", "criminal", "heist", "mob", "gangster", "mafia"],
    "documentary": ["documentary", "true story", "real life", "non-fiction"],
    "drama": ["drama", "dramatic", "emotional", "intense", "moving"],
    "fantasy": ["fantasy", "magic", "magical", "mythical", "dragon"],
    "horror": ["horror", "scary", "terrifying", "creepy", "spooky", "gore"],
    "mystery": ["mystery", "detective", "whodunit", "suspense", "clues"],
    "romance": ["romance", "romantic", "love", "relationship", "love story"],
    "sci-fi": ["sci-fi", "science fiction", "space", "future", "alien", "robot"],
    "thriller": ["thriller", "suspense", "tense", "edge of seat", "twist"],
    "war": ["war", "military", "soldier", "battle", "combat"],
    "western": ["western", "cowboy", "frontier", "wild west"],
}


def _extract_year(movie_text: str) -> Optional[int]:
    """Extract year from movie text like 'Movie Title (1999) — Genres: ...'"""
    match = re.search(r"\((\d{4})\)", movie_text)
    if match:
        return int(match.group(1))
    return None


def _extract_genres(movie_text: str) -> List[str]:
    """Extract genres from movie text."""
    match = re.search(r"Genres?:\s*(.+)$", movie_text, re.IGNORECASE)
    if match:
        genres_str = match.group(1)
        return [g.strip().lower() for g in genres_str.split(",")]
    return []


def _compute_recency_score(year: Optional[int], min_year: int = 1920, max_year: int = 2025) -> float:
    """
    Compute recency score (0-1). Newer movies score higher.
    Uses a soft sigmoid to avoid harsh cutoffs.
    """
    if year is None:
        return 0.5  # Neutral score for unknown years
    
    # Normalize to 0-1 range
    normalized = (year - min_year) / (max_year - min_year)
    normalized = max(0.0, min(1.0, normalized))
    
    # Apply sigmoid-like boost for recent movies (post-2000 get extra weight)
    if year >= 2000:
        normalized = 0.6 + 0.4 * ((year - 2000) / 25)
    
    return min(1.0, normalized)


def _compute_genre_match_score(query: str, movie_genres: List[str]) -> float:
    """
    Compute genre alignment score between query and movie genres.
    Returns 0-1 based on keyword matching.
    """
    query_lower = query.lower()
    
    # Find which genres the query implies
    query_genres = set()
    for genre, keywords in GENRE_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            query_genres.add(genre)
    
    if not query_genres:
        return 0.5  # Neutral if no genre detected in query
    
    if not movie_genres:
        return 0.3  # Slight penalty for missing genre info
    
    # Compute Jaccard-like overlap
    movie_genre_set = set(g.lower() for g in movie_genres)
    
    # Direct matches
    matches = len(query_genres & movie_genre_set)
    
    # Partial matches (e.g., "sci-fi" matching "science fiction")
    for qg in query_genres:
        for mg in movie_genre_set:
            if qg in mg or mg in qg:
                matches += 0.5
    
    # Normalize by expected matches
    score = matches / len(query_genres)
    return min(1.0, score)


def _compute_keyword_match_score(query: str, movie_text: str) -> float:
    """
    Compute lexical keyword overlap between query and movie title/text.
    Useful for specific title mentions or unique terms.
    """
    # Extract meaningful words (3+ chars, no stopwords)
    stopwords = {"the", "and", "for", "with", "about", "from", "that", "this", "movie", "film"}
    
    query_words = set(
        w.lower() for w in re.findall(r"\b\w{3,}\b", query.lower())
        if w.lower() not in stopwords
    )
    
    movie_words = set(
        w.lower() for w in re.findall(r"\b\w{3,}\b", movie_text.lower())
        if w.lower() not in stopwords
    )
    
    if not query_words:
        return 0.5
    
    matches = len(query_words & movie_words)
    return min(1.0, matches / len(query_words))


# ============================================================
# LLM Scoring (5th signal)
# ============================================================

def _llm_score_batch(query: str, movies: List[Dict], model: str = "gpt-4o-mini") -> List[float]:
    """
    Score multiple movies for relevance using LLM in a single API call.
    
    Returns list of scores (0.0-1.0) in same order as input movies.
    """
    from src.client import get_openai_client
    
    if not movies:
        return []
    
    # Build movie list for prompt
    movie_list = "\n".join(
        f"{i+1}. {m.get('text', 'Unknown')}" 
        for i, m in enumerate(movies)
    )
    
    system_prompt = """You are a movie recommendation evaluator. 
Given a user's request and a list of candidate movies, rate how well each movie matches the request.

Return ONLY a JSON array of scores from 0.0 to 1.0, one per movie, in order.
- 1.0 = perfect match for the request
- 0.7-0.9 = good match, fits most criteria  
- 0.4-0.6 = partial match, some relevance
- 0.1-0.3 = weak match, tangentially related
- 0.0 = not relevant at all

Consider: themes, tone, mood, genre, era, and any specific constraints in the request.
Output format: [0.8, 0.6, 0.9, ...]"""

    user_prompt = f"""User request: "{query}"

Candidate movies:
{movie_list}

Return JSON array of {len(movies)} scores:"""

    client = get_openai_client()
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        
        raw = resp.choices[0].message.content.strip()
        
        # Parse JSON array from response
        # Handle cases where LLM wraps in markdown code blocks
        if "```" in raw:
            raw = re.search(r"\[[\d\s,.\[\]]+\]", raw)
            raw = raw.group(0) if raw else "[]"
        
        scores = json.loads(raw)
        
        # Validate and normalize
        if not isinstance(scores, list):
            scores = [0.5] * len(movies)
        
        # Ensure we have right number of scores
        while len(scores) < len(movies):
            scores.append(0.5)
        scores = scores[:len(movies)]
        
        # Clamp to 0-1
        scores = [max(0.0, min(1.0, float(s))) for s in scores]
        
        return scores
        
    except Exception as e:
        print(f"LLM scoring error: {e}")
        return [0.5] * len(movies)  # Neutral fallback


# ============================================================
# Core Ranking Functions
# ============================================================

def rerank(
    query: str,
    candidates: List[Dict],
    weights: Optional[RankingWeights] = None,
    top_k: Optional[int] = None,
    use_llm: bool = False,
) -> List[Dict]:
    """
    Re-rank candidate movies using multi-signal scoring.
    
    Args:
        query: User's search query
        candidates: List of movie dicts from retriever (with 'text', 'similarity')
        weights: Optional custom weights (uses defaults if None)
        top_k: Optional limit on returned results
        use_llm: Whether to include LLM scoring (adds latency + cost)
    
    Returns:
        Re-ranked list of movies with added 'rank_score' and signal breakdowns
    """
    if not candidates:
        return []
    
    if weights is None:
        weights = RankingWeights() if use_llm else RankingWeights.fast_only()
    
    weights = weights.normalize()
    
    # Get LLM scores if enabled (batch call for efficiency)
    llm_scores: List[float] = []
    if use_llm and weights.llm_score > 0:
        llm_scores = _llm_score_batch(query, candidates)
    
    ranked = []
    for i, movie in enumerate(candidates):
        movie_text = movie.get("text", "")
        year = _extract_year(movie_text)
        genres = _extract_genres(movie_text)
        
        # Compute individual signals
        signals = {
            "sim_score": movie.get("similarity", 0.0),
            "recency_score": _compute_recency_score(year),
            "genre_score": _compute_genre_match_score(query, genres),
            "keyword_score": _compute_keyword_match_score(query, movie_text),
            "llm_score": llm_scores[i] if llm_scores else 0.5,
        }
        
        # Compute final weighted score
        rank_score = (
            weights.similarity * signals["sim_score"] +
            weights.recency * signals["recency_score"] +
            weights.genre_match * signals["genre_score"] +
            weights.keyword_match * signals["keyword_score"] +
            weights.llm_score * signals["llm_score"]
        )
        
        ranked.append({
            **movie,
            "rank_score": rank_score,
            "signals": signals,
        })
    
    # Sort by rank_score descending
    ranked.sort(key=lambda x: x["rank_score"], reverse=True)
    
    if top_k:
        ranked = ranked[:top_k]
    
    return ranked


def retrieve_and_rank(
    query: str,
    k_retrieve: int = 20,
    k_final: int = 5,
    weights: Optional[RankingWeights] = None,
    use_llm: bool = False,
) -> List[Dict]:
    """
    Two-stage retrieval + ranking pipeline.
    
    Stage 1: Retrieve top-k candidates by embedding similarity
    Stage 2: Re-rank using multi-signal scoring (optionally with LLM)
    
    Args:
        query: User's search query
        k_retrieve: Number of candidates to retrieve (stage 1)
        k_final: Number of results to return (stage 2)
        weights: Optional ranking weights
        use_llm: Whether to use LLM scoring (adds ~500ms-2s latency)
    
    Returns:
        Final ranked list of movies
    """
    from src.recommender import recommend_movies
    
    # Stage 1: Retrieve candidates
    candidates = recommend_movies(query, k=k_retrieve)
    
    # Stage 2: Re-rank
    ranked = rerank(query, candidates, weights=weights, top_k=k_final, use_llm=use_llm)
    
    return ranked


def retrieve_rank_llm(
    query: str,
    k_retrieve: int = 20,
    k_rerank: int = 10,
    k_final: int = 5,
    weights: Optional[RankingWeights] = None,
) -> List[Dict]:
    """
    Three-stage pipeline: Retrieve → Fast Rank → LLM Rerank
    
    This is the most accurate but slowest approach:
    1. Retrieve 20 candidates by embedding similarity
    2. Fast-rank to top 10 using non-LLM signals
    3. LLM-score the top 10, return top 5
    
    Args:
        query: User's search query
        k_retrieve: Candidates from retrieval (default 20)
        k_rerank: Candidates to send to LLM (default 10)
        k_final: Final results to return (default 5)
        weights: Ranking weights (LLM weight used in final stage)
    
    Returns:
        Final ranked list with LLM-enhanced scores
    """
    from src.recommender import recommend_movies
    
    # Stage 1: Retrieve
    candidates = recommend_movies(query, k=k_retrieve)
    
    # Stage 2: Fast rank (no LLM)
    fast_weights = RankingWeights.fast_only()
    fast_ranked = rerank(query, candidates, weights=fast_weights, top_k=k_rerank, use_llm=False)
    
    # Stage 3: LLM rerank on top candidates
    if weights is None:
        weights = RankingWeights()
    final_ranked = rerank(query, fast_ranked, weights=weights, top_k=k_final, use_llm=True)
    
    return final_ranked
