# 🎬 AI Movie Recommendation System (V1) — Project Report

**Course:** Data Science / Predictive Analytics  
**Author:** Ebenezer Nkrumah Amankwah  
**Date:** December 3, 2025  
**Live Demo:** [movievibesai.streamlit.app](https://movievibesai.streamlit.app)  
**Repository:** [github.com/enkrumah/movie-recs](https://github.com/enkrumah/movie-recs)

---

## 1. Business Understanding

### Problem Statement

Traditional movie recommendation systems rely on collaborative filtering or keyword matching, which often fail to capture the **nuanced preferences** users express naturally:

> *"I want something like Inception but more emotional"*  
> *"A 90s crime thriller with a cat-and-mouse vibe"*

**Business Goal:** Build a recommendation system that understands **natural language queries** describing mood, themes, and vibes—not just genres or titles.

### Why This Matters

- **User Experience:** Eliminate the "browsing paralysis" on streaming platforms
- **Engagement:** Better recommendations → longer sessions → higher retention
- **Competitive Edge:** Semantic understanding differentiates from keyword-based systems

### Connection to Course Fundamentals

This project applies the **k-Nearest Neighbors (k-NN)** concept learned in Weeks 4-5, but extends it to high-dimensional embedding space where "distance" represents semantic similarity rather than Euclidean distance on tabular features.

```
Traditional k-NN:    Feature vectors → Euclidean distance → Find k closest
This Project:        Text embeddings → Cosine similarity → Find k most similar
```

---

## 2. Data Understanding

### Primary Data Source

**MovieLens ml-latest-small Dataset**
- **Records:** 9,742 movies
- **Source:** GroupLens Research (University of Minnesota)
- **Time Period:** Movies from 1902-2018

### Data Fields

| Field | Type | Description | Quality Issues |
|-------|------|-------------|----------------|
| `movieId` | Integer | Unique identifier | None |
| `title` | String | Movie title with year, e.g., "Se7en (1995)" | Year embedded in title |
| `genres` | String | Pipe-separated genres, e.g., "Crime\|Thriller" | Some missing |

### Data Quality Assessment

```python
# Quick stats from data_loader.py
{
    "rows": 9742,
    "unique_movies": 9742,
    "missing_title": 0,
    "missing_genres": 34  # "(no genres listed)"
}
```

### Key Observations

1. **Title Format:** Year embedded in title requires regex extraction
2. **Genre Encoding:** Pipe-delimited, needs normalization
3. **No Plot Data:** Must create meaningful text from limited fields
4. **No Ratings in Model:** Similarity-based, not collaborative filtering

---

## 3. Data Preparation

### Pipeline Overview

```
Raw CSV → Extract Fields → Parse Title/Year → Normalize Genres → Build Text Corpus → Generate Embeddings → L2 Normalize → Index
```

### Step 1: Feature Extraction

Extract only relevant columns:
```python
df = df[["movieId", "title", "genres"]].copy()
```

### Step 2: Title Parsing (Regex)

Separate movie title from year:
```python
def split_title_year(title: str) -> Tuple[str, str]:
    m = re.search(r"\s\((\d{4})\)$", str(title))
    year = m.group(1) if m else ""
    clean = re.sub(r"\s\(\d{4}\)$", "", str(title)).strip()
    return clean, year

# "Se7en (1995)" → ("Se7en", "1995")
```

### Step 3: Genre Normalization

Convert pipe-delimited to comma-separated:
```python
df["genres_norm"] = df["genres"].str.replace("|", ", ", regex=False)
# "Crime|Thriller" → "Crime, Thriller"
```

### Step 4: Text Corpus Construction

Create embedding-ready text for each movie:
```python
text_embed = f"{clean_title} ({year}) — Genres: {genres_norm}"
# Output: "Se7en (1995) — Genres: Crime, Thriller"
```

### Step 5: Embedding Generation

Convert text to 3072-dimensional vectors using OpenAI:
```python
# Model: text-embedding-3-large
response = client.embeddings.create(model="text-embedding-3-large", input=texts)
vectors = [d.embedding for d in response.data]
```

**Why text-embedding-3-large?**
- 3072 dimensions capture nuanced semantic relationships
- State-of-the-art performance on semantic similarity benchmarks
- Handles multi-lingual queries

### Step 6: L2 Normalization

Normalize vectors for cosine similarity via inner product:
```python
faiss.normalize_L2(vectors)  # ||v|| = 1 for all vectors
```

**Mathematical Foundation:**
When vectors are L2-normalized:
```
cosine_similarity(a, b) = dot_product(a, b)
```
This allows FAISS IndexFlatIP (inner product) to compute cosine similarity efficiently.

---

## 4. Modeling

### Core Algorithm: Semantic k-NN via Vector Search

**Relationship to Course k-NN:**
| Traditional k-NN | This Project |
|------------------|--------------|
| Tabular features | Text embeddings |
| Euclidean distance | Cosine similarity |
| scikit-learn | FAISS (Facebook AI) |
| Classification/Regression | Information Retrieval |

### FAISS Index Configuration

```python
index = faiss.IndexFlatIP(3072)  # Inner product, 3072 dimensions
index.add(normalized_vectors)    # Add all movie embeddings
```

### Query Processing

```python
def recommend_movies(query: str, k: int = 5):
    # 1. Embed query using same model
    q = client.embeddings.create(model="text-embedding-3-large", input=query)
    q_vector = np.array(q.data[0].embedding)
    
    # 2. L2-normalize query
    faiss.normalize_L2(q_vector)
    
    # 3. Find k nearest neighbors
    similarities, indices = index.search(q_vector, k)
    
    return movies[indices]
```

### Multi-Signal Ranking (Smart Mode)

**Three-Stage Pipeline:**
```
Retrieve 20 → Fast Rank to 10 → LLM Rerank → Return Top 5
```

**Five Ranking Signals:**
| Signal | Weight | Implementation |
|--------|--------|----------------|
| Similarity | 45% | Cosine similarity from embeddings |
| Recency | 10% | Sigmoid boost for post-2000 movies |
| Genre Match | 15% | Query keywords → genre alignment |
| Keyword Match | 10% | Lexical overlap (stopwords removed) |
| LLM Score | 20% | GPT-4o-mini relevance judgment |

**Final Score Calculation:**
```python
rank_score = (
    0.45 * sim_score +
    0.10 * recency_score +
    0.15 * genre_score +
    0.10 * keyword_score +
    0.20 * llm_score
)
```

### Why This Model Solves the Problem

1. **Semantic Understanding:** Embeddings capture meaning, not just keywords
2. **Flexible Queries:** Users can describe mood, themes, or specific criteria
3. **Multi-Signal Fusion:** Combines semantic with heuristic signals
4. **Explainability:** LLM provides human-readable explanations

---

## 5. Evaluation

### Evaluation Framework

Two complementary approaches:

| Method | What It Measures | Pros | Cons |
|--------|------------------|------|------|
| **Ground Truth** | Exact match with pre-defined relevant movies | Reproducible, free | Misses valid alternatives |
| **LLM-Judge** | Semantic relevance scored by GPT-4o-mini | Comprehensive | Cost, non-deterministic |

### Metrics Implemented

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **nDCG@k** | DCG@k / IDCG@k | Ranking quality (position-sensitive) |
| **Precision@k** | Relevant in top-k / k | Cleanliness of results |
| **Recall@k** | Relevant found / Total relevant | Completeness |
| **Hit Rate@k** | ≥1 relevant in top-k? | Robustness |
| **MRR** | 1 / rank of first relevant | Speed to good result |
| **Coverage** | Unique movies / Catalog size | Diversity |

### Results Summary

```
============================================================
COMPARISON SUMMARY
============================================================

Ground Truth Evaluation:
Pipeline                  nDCG@k    Precision@k  Hit Rate   MRR
------------------------------------------------------------
retrieval                 0.0635    0.0267       0.1333     0.0411
ranked                    0.0606    0.0200       0.1000     0.0483

LLM-Judged Evaluation:
Pipeline                  nDCG@k    Precision@k  Hit Rate   MRR
------------------------------------------------------------
retrieval                 0.9556    0.8800       1.0000     0.9500
ranked                    0.9579    0.8200       1.0000     0.9500
```

### Interpretation

- **Low ground truth scores:** Narrow test coverage, not poor retrieval
- **High LLM-judge scores:** System finds semantically relevant movies that weren't pre-defined
- **Key Insight:** Traditional evaluation underestimates semantic retrieval quality

### Business Case / ROI

For a streaming platform:
- **Metric:** Recommendation click-through rate (CTR)
- **Baseline:** 5% CTR on homepage
- **Expected Improvement:** 15-25% lift with semantic recommendations
- **ROI Calculation:** Higher engagement → reduced churn → increased lifetime value

---

## 6. Deployment

### Deployment Platform

**Streamlit Cloud**
- Free hosting for Python/Streamlit apps
- Direct GitHub integration (auto-deploy on push)
- Built-in secrets management for API keys

**Live URL:** [movievibesai.streamlit.app](https://movievibesai.streamlit.app)

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit Cloud                     │
├─────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │  Streamlit  │  │   FAISS     │  │   OpenAI    │  │
│  │     UI      │→ │   Index     │→ │   API       │  │
│  │  (app.py)   │  │ (in-memory) │  │ (embeddings)│  │
│  └─────────────┘  └─────────────┘  └─────────────┘  │
│                          ↓                          │
│              artifacts/movie_vectors.npy            │
│                    (~120MB, Git LFS)                │
└─────────────────────────────────────────────────────┘
```

### Ethical Considerations

1. **Bias in Embeddings:** OpenAI models trained on internet text may reflect biases
   - *Mitigation:* Monitor recommendation diversity, manual review of edge cases

2. **Filter Bubbles:** Semantic similarity may reinforce existing preferences
   - *Mitigation:* Include "diversity" signal, occasionally surface unexpected picks

3. **Data Privacy:** No user data is collected beyond session state
   - *Mitigation:* Feedback collected via Google Forms (optional, anonymous)

### Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| OpenAI API downtime | Service unavailable | Cache popular queries, graceful error handling |
| API cost overruns | Budget exceeded | Rate limiting, batched embeddings |
| Embedding drift | Degraded relevance | Version artifacts, monitor metrics |
| LFS storage limits | Deployment fails | Compress artifacts, consider cloud storage |

---

## 7. Key Learnings

### From Course to Production

1. **k-NN Fundamentals → Vector Search:** The distance-based intuition from class directly applies to embedding space
2. **Feature Engineering → Text Corpus:** Creating rich text representations is modern "feature engineering"
3. **Evaluation Metrics:** nDCG, Precision, Recall from class apply to retrieval systems
4. **Model Selection → Embedding Model:** Choosing text-embedding-3-large is analogous to algorithm selection

### Technical Skills Developed

- Working with high-dimensional vector spaces
- API integration (OpenAI, FAISS)
- Building evaluation frameworks
- Deploying ML applications

---

## Appendix: Running the Code

```bash
# Clone
git clone https://github.com/enkrumah/movie-recs.git
cd movie-recs

# Install
pip install -r requirements.txt

# Set API key
echo "OPENAI_API_KEY=sk-..." > .env

# Run
streamlit run app.py

# Evaluate
python -m src.evaluation --llm-judge -v
```

---

*Report Date: December 3, 2025*
