# src/evaluation.py
"""
Evaluation metrics for retrieval and ranking quality.

Implements standard IR/RecSys metrics:
- nDCG@k: Normalized Discounted Cumulative Gain
- Precision@k: Fraction of relevant items in top-k
- Recall@k: Fraction of relevant items retrieved
- Hit Rate@k: Binary success metric
- MRR: Mean Reciprocal Rank
- Coverage: Catalog diversity

Also supports LLM-based relevance evaluation (modern approach).
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Set, Callable, Optional, Tuple
import numpy as np


# ============================================================
# Data Classes
# ============================================================

@dataclass
class EvalResult:
    """Container for evaluation results on a single query."""
    query: str
    ndcg: float
    precision: float
    recall: float
    hit: bool
    reciprocal_rank: float
    retrieved_ids: List[str] = field(default_factory=list)


@dataclass 
class EvalSummary:
    """Aggregated evaluation results across all queries."""
    num_queries: int
    ndcg_mean: float
    ndcg_std: float
    precision_mean: float
    precision_std: float
    recall_mean: float
    recall_std: float
    hit_rate: float
    mrr: float
    coverage: float
    coverage_count: int
    catalog_size: int
    eval_method: str = "ground_truth"  # "ground_truth" or "llm_judge"
    
    def __str__(self) -> str:
        method_label = "🤖 LLM-Judged" if self.eval_method == "llm_judge" else "📋 Ground Truth"
        return (
            f"=== Evaluation Summary ({self.num_queries} queries) [{method_label}] ===\n"
            f"nDCG@k:      {self.ndcg_mean:.4f} ± {self.ndcg_std:.4f}\n"
            f"Precision@k: {self.precision_mean:.4f} ± {self.precision_std:.4f}\n"
            f"Recall@k:    {self.recall_mean:.4f} ± {self.recall_std:.4f}\n"
            f"Hit Rate@k:  {self.hit_rate:.4f}\n"
            f"MRR:         {self.mrr:.4f}\n"
            f"Coverage:    {self.coverage:.4f} ({self.coverage_count}/{self.catalog_size} movies)"
        )


@dataclass
class LLMJudgment:
    """Container for LLM relevance judgments on a single query."""
    query: str
    movies: List[str]
    scores: List[float]  # 0.0-1.0 relevance scores
    reasoning: List[str]  # Optional explanations


# ============================================================
# Metric Functions
# ============================================================

def dcg_at_k(relevances: List[float], k: int) -> float:
    """Compute Discounted Cumulative Gain at k."""
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    
    gains = np.array([(2**rel - 1) for rel in relevances])
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(relevances: List[float], k: int) -> float:
    """Compute Normalized DCG at k."""
    dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


def precision_at_k(relevances: List[float], k: int, threshold: float = 0.5) -> float:
    """Precision@k = |relevant items in top-k| / k"""
    top_k = relevances[:k]
    if not top_k:
        return 0.0
    
    relevant_count = sum(1 for r in top_k if r >= threshold)
    return relevant_count / len(top_k)


def recall_at_k(
    relevances: List[float], 
    k: int, 
    total_relevant: int,
    threshold: float = 0.5,
) -> float:
    """Recall@k = |relevant items in top-k| / |total relevant items|"""
    if total_relevant == 0:
        return 0.0
    
    top_k = relevances[:k]
    relevant_retrieved = sum(1 for r in top_k if r >= threshold)
    return relevant_retrieved / total_relevant


def hit_at_k(relevances: List[float], k: int, threshold: float = 0.5) -> bool:
    """Hit@k = 1 if at least one relevant item in top-k, else 0"""
    top_k = relevances[:k]
    return any(r >= threshold for r in top_k)


def reciprocal_rank(relevances: List[float], threshold: float = 0.5) -> float:
    """RR = 1 / (rank of first relevant item), or 0 if none found"""
    for i, rel in enumerate(relevances):
        if rel >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def coverage(retrieved_ids: Set[str], catalog_size: int) -> float:
    """Coverage = |unique items retrieved| / |catalog|"""
    if catalog_size == 0:
        return 0.0
    return len(retrieved_ids) / catalog_size


# ============================================================
# LLM-Based Relevance Judgment
# ============================================================

def llm_judge_relevance(
    query: str,
    movies: List[Dict],
    model: str = "gpt-4o-mini",
) -> LLMJudgment:
    """
    Use LLM to judge relevance of retrieved movies to the query.
    
    Returns graded relevance scores (0.0-1.0) for each movie.
    """
    from src.client import get_openai_client
    
    if not movies:
        return LLMJudgment(query=query, movies=[], scores=[], reasoning=[])
    
    movie_list = "\n".join(
        f"{i+1}. {m.get('text', 'Unknown')}"
        for i, m in enumerate(movies)
    )
    
    system_prompt = """You are an expert movie relevance evaluator.

Given a user's search query and a list of retrieved movies, judge how relevant each movie is to what the user is looking for.

For each movie, provide:
1. A relevance score from 0.0 to 1.0:
   - 1.0 = Perfect match, exactly what the user wants
   - 0.8-0.9 = Excellent match, fits the request very well
   - 0.6-0.7 = Good match, mostly fits the criteria
   - 0.4-0.5 = Partial match, some relevance but missing key aspects
   - 0.2-0.3 = Weak match, only tangentially related
   - 0.0-0.1 = Not relevant at all

2. A brief reason (5-10 words)

Consider: genre, themes, mood, tone, era, and any specific criteria mentioned.

Return your response as a JSON array of objects:
[{"score": 0.85, "reason": "matches sci-fi thriller criteria"}, ...]"""

    user_prompt = f"""User Query: "{query}"

Retrieved Movies:
{movie_list}

Judge each movie's relevance. Return exactly {len(movies)} judgments as JSON array:"""

    client = get_openai_client()
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=500,
        )
        
        raw = resp.choices[0].message.content.strip()
        
        # Parse JSON from response (handle markdown code blocks)
        if "```" in raw:
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
            if json_match:
                raw = json_match.group(1).strip()
        
        judgments = json.loads(raw)
        
        # Extract scores and reasons
        scores = []
        reasoning = []
        for j in judgments[:len(movies)]:
            score = float(j.get("score", 0.5))
            score = max(0.0, min(1.0, score))  # Clamp to 0-1
            scores.append(score)
            reasoning.append(j.get("reason", ""))
        
        # Pad if needed
        while len(scores) < len(movies):
            scores.append(0.5)
            reasoning.append("(no judgment)")
        
        return LLMJudgment(
            query=query,
            movies=[m.get("text", "") for m in movies],
            scores=scores,
            reasoning=reasoning,
        )
        
    except Exception as e:
        print(f"LLM judgment error: {e}")
        # Return neutral scores on failure
        return LLMJudgment(
            query=query,
            movies=[m.get("text", "") for m in movies],
            scores=[0.5] * len(movies),
            reasoning=[f"(error: {str(e)[:30]})"] * len(movies),
        )


# ============================================================
# Evaluation Functions
# ============================================================

def evaluate_single_query(
    query: str,
    retrieved: List[Dict],
    ground_truth: Set[str],
    k: int = 5,
) -> EvalResult:
    """Evaluate retrieval quality for a single query using ground truth."""
    retrieved_ids = [m.get("movieId", "") for m in retrieved[:k]]
    relevances = [1.0 if mid in ground_truth else 0.0 for mid in retrieved_ids]
    total_relevant = len(ground_truth)
    
    return EvalResult(
        query=query,
        ndcg=ndcg_at_k(relevances, k),
        precision=precision_at_k(relevances, k),
        recall=recall_at_k(relevances, k, total_relevant),
        hit=hit_at_k(relevances, k),
        reciprocal_rank=reciprocal_rank(relevances),
        retrieved_ids=retrieved_ids,
    )


def evaluate_single_query_llm(
    query: str,
    retrieved: List[Dict],
    k: int = 5,
    threshold: float = 0.6,
) -> Tuple[EvalResult, LLMJudgment]:
    """Evaluate retrieval quality for a single query using LLM judgment."""
    retrieved_k = retrieved[:k]
    judgment = llm_judge_relevance(query, retrieved_k)
    
    relevances = judgment.scores
    retrieved_ids = [m.get("movieId", "") for m in retrieved_k]
    
    # For recall, estimate total relevant as sum of scores (graded relevance)
    # This is an approximation since we don't know all relevant items
    total_relevant_estimate = max(1, sum(1 for s in relevances if s >= threshold))
    
    return EvalResult(
        query=query,
        ndcg=ndcg_at_k(relevances, k),
        precision=precision_at_k(relevances, k, threshold=threshold),
        recall=recall_at_k(relevances, k, total_relevant_estimate, threshold=threshold),
        hit=hit_at_k(relevances, k, threshold=threshold),
        reciprocal_rank=reciprocal_rank(relevances, threshold=threshold),
        retrieved_ids=retrieved_ids,
    ), judgment


def evaluate_batch(
    queries: List[str],
    ground_truths: List[Set[str]],
    retriever_fn: Callable[[str, int], List[Dict]],
    k: int = 5,
    catalog_size: Optional[int] = None,
    verbose: bool = False,
) -> EvalSummary:
    """Evaluate retrieval quality using ground truth."""
    results: List[EvalResult] = []
    all_retrieved_ids: Set[str] = set()
    
    for i, (query, gt) in enumerate(zip(queries, ground_truths)):
        if verbose:
            print(f"  [{i+1}/{len(queries)}] {query[:50]}...")
        
        retrieved = retriever_fn(query, k)
        result = evaluate_single_query(query, retrieved, gt, k)
        results.append(result)
        all_retrieved_ids.update(result.retrieved_ids)
        
        if verbose:
            hit_str = "✓" if result.hit else "✗"
            print(f"           nDCG={result.ndcg:.3f} P@k={result.precision:.3f} {hit_str}")
    
    # Aggregate metrics
    ndcgs = [r.ndcg for r in results]
    precisions = [r.precision for r in results]
    recalls = [r.recall for r in results]
    hits = [1.0 if r.hit else 0.0 for r in results]
    rrs = [r.reciprocal_rank for r in results]
    
    # Auto-detect catalog size
    if catalog_size is None:
        try:
            ids_path = Path("artifacts/movie_ids.json")
            if ids_path.exists():
                catalog_size = len(json.loads(ids_path.read_text()))
            else:
                catalog_size = len(all_retrieved_ids)
        except Exception:
            catalog_size = len(all_retrieved_ids)
    
    return EvalSummary(
        num_queries=len(results),
        ndcg_mean=float(np.mean(ndcgs)),
        ndcg_std=float(np.std(ndcgs)),
        precision_mean=float(np.mean(precisions)),
        precision_std=float(np.std(precisions)),
        recall_mean=float(np.mean(recalls)),
        recall_std=float(np.std(recalls)),
        hit_rate=float(np.mean(hits)),
        mrr=float(np.mean(rrs)),
        coverage=coverage(all_retrieved_ids, catalog_size),
        coverage_count=len(all_retrieved_ids),
        catalog_size=catalog_size,
        eval_method="ground_truth",
    )


def evaluate_batch_llm(
    queries: List[str],
    retriever_fn: Callable[[str, int], List[Dict]],
    k: int = 5,
    catalog_size: Optional[int] = None,
    verbose: bool = False,
    threshold: float = 0.6,
) -> Tuple[EvalSummary, List[LLMJudgment]]:
    """
    Evaluate retrieval quality using LLM-based relevance judgment.
    
    Returns both summary metrics and detailed judgments.
    """
    results: List[EvalResult] = []
    judgments: List[LLMJudgment] = []
    all_retrieved_ids: Set[str] = set()
    
    for i, query in enumerate(queries):
        if verbose:
            print(f"  [{i+1}/{len(queries)}] {query[:50]}...")
        
        retrieved = retriever_fn(query, k)
        result, judgment = evaluate_single_query_llm(query, retrieved, k, threshold)
        results.append(result)
        judgments.append(judgment)
        all_retrieved_ids.update(result.retrieved_ids)
        
        if verbose:
            hit_str = "✓" if result.hit else "✗"
            avg_score = np.mean(judgment.scores) if judgment.scores else 0
            print(f"           nDCG={result.ndcg:.3f} P@k={result.precision:.3f} AvgRel={avg_score:.2f} {hit_str}")
    
    # Aggregate metrics
    ndcgs = [r.ndcg for r in results]
    precisions = [r.precision for r in results]
    recalls = [r.recall for r in results]
    hits = [1.0 if r.hit else 0.0 for r in results]
    rrs = [r.reciprocal_rank for r in results]
    
    # Auto-detect catalog size
    if catalog_size is None:
        try:
            ids_path = Path("artifacts/movie_ids.json")
            if ids_path.exists():
                catalog_size = len(json.loads(ids_path.read_text()))
            else:
                catalog_size = len(all_retrieved_ids)
        except Exception:
            catalog_size = len(all_retrieved_ids)
    
    summary = EvalSummary(
        num_queries=len(results),
        ndcg_mean=float(np.mean(ndcgs)),
        ndcg_std=float(np.std(ndcgs)),
        precision_mean=float(np.mean(precisions)),
        precision_std=float(np.std(precisions)),
        recall_mean=float(np.mean(recalls)),
        recall_std=float(np.std(recalls)),
        hit_rate=float(np.mean(hits)),
        mrr=float(np.mean(rrs)),
        coverage=coverage(all_retrieved_ids, catalog_size),
        coverage_count=len(all_retrieved_ids),
        catalog_size=catalog_size,
        eval_method="llm_judge",
    )
    
    return summary, judgments


# ============================================================
# Dataset Loading
# ============================================================

EVAL_DATASET_PATH = Path("data/eval_dataset.json")


def load_eval_dataset(path: Optional[Path] = None) -> List[Dict]:
    """Load evaluation dataset from JSON file."""
    path = path or EVAL_DATASET_PATH
    
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found at {path}. "
            "Please create data/eval_dataset.json with queries and ground truth."
        )
    
    data = json.loads(path.read_text())
    return data.get("queries", [])


def get_eval_queries_and_ground_truth(
    path: Optional[Path] = None,
    categories: Optional[List[str]] = None,
) -> tuple[List[str], List[Set[str]]]:
    """Load queries and ground truth from evaluation dataset."""
    dataset = load_eval_dataset(path)
    
    if categories:
        dataset = [q for q in dataset if q.get("category") in categories]
    
    queries = [item["query"] for item in dataset]
    ground_truths = [set(item["relevant"]) for item in dataset]
    
    return queries, ground_truths


def get_eval_queries(
    path: Optional[Path] = None,
    categories: Optional[List[str]] = None,
) -> List[str]:
    """Load just queries (for LLM-based evaluation)."""
    dataset = load_eval_dataset(path)
    
    if categories:
        dataset = [q for q in dataset if q.get("category") in categories]
    
    return [item["query"] for item in dataset]


# ============================================================
# Runner Functions
# ============================================================

def run_evaluation(
    k: int = 5,
    categories: Optional[List[str]] = None,
    verbose: bool = False,
) -> EvalSummary:
    """Run evaluation using ground truth (retrieval only)."""
    from src.recommender import recommend_movies
    
    queries, ground_truths = get_eval_queries_and_ground_truth(categories=categories)
    
    return evaluate_batch(
        queries=queries,
        ground_truths=ground_truths,
        retriever_fn=recommend_movies,
        k=k,
        verbose=verbose,
    )


def run_ranked_evaluation(
    k: int = 5,
    k_retrieve: int = 20,
    categories: Optional[List[str]] = None,
    verbose: bool = False,
) -> EvalSummary:
    """Run evaluation using ground truth (retrieve + fast rank)."""
    from src.ranker import retrieve_and_rank
    
    queries, ground_truths = get_eval_queries_and_ground_truth(categories=categories)
    
    def retriever_fn(query: str, k: int) -> List[Dict]:
        return retrieve_and_rank(query, k_retrieve=k_retrieve, k_final=k, use_llm=False)
    
    return evaluate_batch(
        queries=queries,
        ground_truths=ground_truths,
        retriever_fn=retriever_fn,
        k=k,
        verbose=verbose,
    )


def run_llm_ranked_evaluation(
    k: int = 5,
    k_retrieve: int = 20,
    k_rerank: int = 10,
    categories: Optional[List[str]] = None,
    verbose: bool = False,
) -> EvalSummary:
    """Run evaluation using ground truth (retrieve + rank + LLM rerank)."""
    from src.ranker import retrieve_rank_llm
    
    queries, ground_truths = get_eval_queries_and_ground_truth(categories=categories)
    
    def retriever_fn(query: str, k: int) -> List[Dict]:
        return retrieve_rank_llm(
            query,
            k_retrieve=k_retrieve,
            k_rerank=k_rerank,
            k_final=k,
        )
    
    return evaluate_batch(
        queries=queries,
        ground_truths=ground_truths,
        retriever_fn=retriever_fn,
        k=k,
        verbose=verbose,
    )


def run_llm_judge_evaluation(
    k: int = 5,
    categories: Optional[List[str]] = None,
    verbose: bool = False,
) -> Tuple[EvalSummary, List[LLMJudgment]]:
    """
    Run evaluation using LLM-based relevance judgment (retrieval only).
    
    This doesn't use ground truth—the LLM judges each retrieved movie's relevance.
    """
    from src.recommender import recommend_movies
    
    queries = get_eval_queries(categories=categories)
    
    return evaluate_batch_llm(
        queries=queries,
        retriever_fn=recommend_movies,
        k=k,
        verbose=verbose,
    )


def run_llm_judge_ranked_evaluation(
    k: int = 5,
    k_retrieve: int = 20,
    categories: Optional[List[str]] = None,
    verbose: bool = False,
) -> Tuple[EvalSummary, List[LLMJudgment]]:
    """Run LLM-judged evaluation (retrieve + fast rank)."""
    from src.ranker import retrieve_and_rank
    
    queries = get_eval_queries(categories=categories)
    
    def retriever_fn(query: str, k: int) -> List[Dict]:
        return retrieve_and_rank(query, k_retrieve=k_retrieve, k_final=k, use_llm=False)
    
    return evaluate_batch_llm(
        queries=queries,
        retriever_fn=retriever_fn,
        k=k,
        verbose=verbose,
    )


def compare_pipelines(
    k: int = 5,
    categories: Optional[List[str]] = None,
    include_llm: bool = False,
    verbose: bool = False,
) -> Dict[str, EvalSummary]:
    """Compare pipelines using ground truth evaluation."""
    results = {}
    
    print("=" * 60)
    print("Pipeline 1: Retrieval Only (embedding similarity)")
    print("=" * 60)
    results["retrieval"] = run_evaluation(k=k, categories=categories, verbose=verbose)
    print(results["retrieval"])
    
    print("\n" + "=" * 60)
    print("Pipeline 2: Retrieve + Fast Rank (4 signals, no LLM)")
    print("=" * 60)
    results["ranked"] = run_ranked_evaluation(k=k, categories=categories, verbose=verbose)
    print(results["ranked"])
    
    if include_llm:
        print("\n" + "=" * 60)
        print("Pipeline 3: Retrieve + Rank + LLM (5 signals)")
        print("=" * 60)
        results["llm_ranked"] = run_llm_ranked_evaluation(k=k, categories=categories, verbose=verbose)
        print(results["llm_ranked"])
    
    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY (Ground Truth)")
    print("=" * 60)
    print(f"{'Pipeline':<25} {'nDCG@k':<12} {'Precision@k':<12} {'Hit Rate':<12} {'MRR':<12}")
    print("-" * 60)
    for name, summary in results.items():
        print(f"{name:<25} {summary.ndcg_mean:.4f}       {summary.precision_mean:.4f}        {summary.hit_rate:.4f}       {summary.mrr:.4f}")
    
    return results


def compare_pipelines_llm_judge(
    k: int = 5,
    categories: Optional[List[str]] = None,
    verbose: bool = False,
) -> Dict[str, EvalSummary]:
    """Compare pipelines using LLM-based relevance judgment."""
    results = {}
    
    print("=" * 60)
    print("Pipeline 1: Retrieval Only [LLM-Judged]")
    print("=" * 60)
    summary, judgments = run_llm_judge_evaluation(k=k, categories=categories, verbose=verbose)
    results["retrieval"] = summary
    print(summary)
    
    print("\n" + "=" * 60)
    print("Pipeline 2: Retrieve + Fast Rank [LLM-Judged]")
    print("=" * 60)
    summary2, judgments2 = run_llm_judge_ranked_evaluation(k=k, categories=categories, verbose=verbose)
    results["ranked"] = summary2
    print(summary2)
    
    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY (LLM-Judged)")
    print("=" * 60)
    print(f"{'Pipeline':<25} {'nDCG@k':<12} {'Precision@k':<12} {'Hit Rate':<12} {'MRR':<12}")
    print("-" * 60)
    for name, summary in results.items():
        print(f"{name:<25} {summary.ndcg_mean:.4f}       {summary.precision_mean:.4f}        {summary.hit_rate:.4f}       {summary.mrr:.4f}")
    
    return results


def show_llm_judgments(judgments: List[LLMJudgment], max_queries: int = 5):
    """Print detailed LLM judgments for inspection."""
    print("\n" + "=" * 60)
    print("DETAILED LLM JUDGMENTS (sample)")
    print("=" * 60)
    
    for judgment in judgments[:max_queries]:
        print(f"\n📝 Query: \"{judgment.query}\"")
        print("-" * 40)
        for i, (movie, score, reason) in enumerate(zip(judgment.movies, judgment.scores, judgment.reasoning)):
            emoji = "✅" if score >= 0.6 else "⚠️" if score >= 0.4 else "❌"
            print(f"  {i+1}. {emoji} [{score:.2f}] {movie[:50]}...")
            if reason:
                print(f"      → {reason}")


# ============================================================
# CLI Runner
# ============================================================

if __name__ == "__main__":
    import sys
    
    include_llm = "--llm" in sys.argv
    llm_judge = "--llm-judge" in sys.argv
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    show_details = "--details" in sys.argv
    
    print("\n🎬 Movie Recommender Evaluation Suite")
    print(f"   Dataset: {EVAL_DATASET_PATH}")
    
    try:
        queries = get_eval_queries()
        print(f"   Queries: {len(queries)}")
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
    
    print()
    
    if llm_judge:
        # LLM-based relevance evaluation
        print("🤖 Using LLM-based relevance judgment (no ground truth)")
        print("   This evaluates whether retrieved movies semantically match queries.\n")
        
        results = compare_pipelines_llm_judge(k=5, verbose=verbose)
        
        if show_details:
            # Run again to get judgments for display
            _, judgments = run_llm_judge_evaluation(k=5, verbose=False)
            show_llm_judgments(judgments)
    else:
        # Traditional ground truth evaluation
        compare_pipelines(k=5, include_llm=include_llm, verbose=verbose)
    
    # Tips
    print()
    if not llm_judge:
        print("💡 Tip: Run with --llm-judge for LLM-based relevance evaluation")
        print("   python -m src.evaluation --llm-judge")
    if not include_llm and not llm_judge:
        print("💡 Tip: Run with --llm flag to include LLM-enhanced ranking")
        print("   python -m src.evaluation --llm")
    if not verbose:
        print("💡 Tip: Run with -v flag for per-query details")
    if llm_judge and not show_details:
        print("💡 Tip: Run with --details to see LLM judgment reasoning")
        print("   python -m src.evaluation --llm-judge --details")
