# 🎬 AI Movie Recommendation System — From Prototype to Production

**Course:** Data Science / Predictive Analytics  
**Author:** Ebenezer Nkrumah Amankwah  
**Date:** December 3, 2025

| Version | Live Demo | Repository |
|---------|-----------|------------|
| **V1 (Prototype)** | [movievibesai.streamlit.app](https://movievibesai.streamlit.app) | [github.com/enkrumah/movie-recs](https://github.com/enkrumah/movie-recs) |
| **V2 (Production)** | [movie-recs-v2-production.up.railway.app](https://movie-recs-v2-production.up.railway.app) | [github.com/enkrumah/movie-recs-v2](https://github.com/enkrumah/movie-recs-v2) |

---

## 1. Problem & Motivation

### The Challenge: Browsing Paralysis

Traditional movie recommendation systems fail to understand how users actually think:

| What Systems Expect | What Users Actually Say |
|---------------------|------------------------|
| "Action" | *"Something like Inception but more emotional"* |
| "Comedy, 2020" | *"A feel-good movie for a rainy Sunday"* |
| "Thriller" | *"90s crime thriller with a cat-and-mouse vibe"* |

### Business Impact

- **Problem:** Users spend 20+ minutes browsing, often giving up
- **Opportunity:** Semantic understanding → better matches → higher engagement
- **Solution:** Build a system that understands **meaning**, not just keywords

---

## 2. Course Connection — k-NN Fundamentals to Vector Search

### The Foundation: k-Nearest Neighbors (Weeks 4-5)

This project directly applies k-NN principles learned in class:

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRADITIONAL k-NN                              │
│                                                                  │
│   Tabular Data → Euclidean Distance → Find k Closest Points     │
│                                                                  │
│   Example: Loan approval based on income, credit score           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    SAME PRINCIPLE, NEW SPACE
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    SEMANTIC k-NN (This Project)                  │
│                                                                  │
│   Text → Embeddings → Cosine Similarity → Find k Most Similar   │
│                                                                  │
│   Example: "mind-bending sci-fi" → Inception, Matrix, Arrival   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Insight

> **"Similar things are close together"** — this intuition works in 3072-dimensional embedding space just like it works in 2D feature space.

---

## 3. V1: The Prototype

### Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit Cloud                     │
├─────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │  Streamlit  │  │   FAISS     │  │   OpenAI    │  │
│  │     UI      │→ │   Index     │→ │   API       │  │
│  │  (app.py)   │  │ (in-memory) │  │ (embeddings)│  │
│  └─────────────┘  └─────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────┘
```

### Data Source

**MovieLens ml-latest-small Dataset**
- 9,742 movies
- Fields: `movieId`, `title`, `genres`
- Static snapshot (movies up to 2018)

### Data Preparation Pipeline

```
Raw CSV → Extract → Parse → Normalize → Embed → L2 Normalize → Index
```

| Step | Input | Output | Code |
|------|-------|--------|------|
| **1. Extract** | Full CSV | 3 columns | `df[["movieId", "title", "genres"]]` |
| **2. Parse Title** | `"Se7en (1995)"` | title + year | `regex: \((\d{4})\)$` |
| **3. Normalize Genres** | `"Crime\|Thriller"` | `"Crime, Thriller"` | `.str.replace("\|", ", ")` |
| **4. Build Corpus** | Components | Text | `f"{title} ({year}) — Genres: {genres}"` |
| **5. Embed** | Text | 3072-dim vector | `text-embedding-3-large` |
| **6. L2 Normalize** | Raw vector | Unit vector | `faiss.normalize_L2()` |
| **7. Index** | Vectors | Searchable index | `faiss.IndexFlatIP()` |

### Why L2 Normalization?

When vectors are L2-normalized (length = 1):
```
cosine_similarity(a, b) = dot_product(a, b)
```
This allows FAISS inner product index to compute cosine similarity efficiently.

### Search Process

```python
def recommend_movies(query: str, k: int = 5):
    # 1. Embed query with same model
    q_vector = embed(query)  # → 3072 dimensions
    
    # 2. L2-normalize
    normalize(q_vector)      # → length = 1
    
    # 3. Find k nearest neighbors
    similarities, indices = index.search(q_vector, k)
    
    return movies[indices]
```

### Multi-Signal Ranking (Smart Mode)

V1 implements an **ensemble approach** (Week 13-14 concepts):

| Signal | Weight | Description |
|--------|--------|-------------|
| **Similarity** | 45% | Cosine similarity from embeddings |
| **Recency** | 10% | Sigmoid boost for post-2000 movies |
| **Genre Match** | 15% | Query keywords → genre alignment |
| **Keyword Match** | 10% | Lexical overlap (stopwords removed) |
| **LLM Score** | 20% | GPT-4o-mini relevance judgment |

**Three-Stage Pipeline:**
```
Retrieve 20 candidates → Fast Rank to 10 → LLM Rerank → Return Top 5
```

---

## 4. V1 Demo

### Live: [movievibesai.streamlit.app](https://movievibesai.streamlit.app)

**Try These Queries:**

| Query | What It Tests |
|-------|---------------|
| `"mind-bending sci-fi"` | Semantic understanding |
| `"feel-good sports drama, true story"` | Multi-concept parsing |
| `"90s crime thriller, cat-and-mouse"` | Era + mood |

**Features to Show:**
- Quick example buttons
- Results with similarity scores
- "Smart Mode" toggle for LLM reranking
- "Why these match your vibe" explanation

---

## 5. Evolution to V2: From Prototype to Production

### Why Build V2?

| V1 Limitation | Impact | V2 Solution |
|---------------|--------|-------------|
| Static data (2018) | Missing new movies | Live TMDB API |
| In-memory FAISS | Can't scale | pgvector (PostgreSQL) |
| No user accounts | No personalization | Google OAuth |
| Basic Streamlit UI | Limited UX | React + Tailwind |

### Architecture Comparison

```
V1: Streamlit ←→ FAISS (memory) ←→ OpenAI
                    ↑
              Static CSV file

V2: React ←→ FastAPI ←→ PostgreSQL/pgvector ←→ OpenAI
                ↑              ↑
           Live TMDB      Persistent storage
```

### V2 Tech Stack

| Layer | V1 | V2 |
|-------|----|----|
| **Frontend** | Streamlit | React + Tailwind CSS |
| **Backend** | Streamlit | FastAPI + Gunicorn |
| **Vector Store** | FAISS | pgvector (PostgreSQL) |
| **Database** | None | PostgreSQL |
| **Data Source** | MovieLens CSV | TMDB + OMDB APIs |
| **Auth** | None | Google OAuth 2.0 |
| **Hosting** | Streamlit Cloud | Railway |

---

## 6. V2: Production System

### Data Pipeline

```
TMDB API → Fetch Movies → OMDB Enrichment → Build Corpus → Embed → pgvector
```

**Richer Data:**
```python
# V1 corpus (limited)
"Se7en (1995) — Genres: Crime, Thriller"

# V2 corpus (rich)
"Se7en (1995). Two detectives hunt a serial killer who uses the seven 
deadly sins as motives. Genres: Crime, Thriller. Directed by David Fincher. 
Starring Brad Pitt, Morgan Freeman."
```

### Three Search Modes

| Mode | Pipeline | Latency | Best For |
|------|----------|---------|----------|
| **Fast** | Query → Embed → pgvector → Top K | ~200ms | Quick searches |
| **Smart** | + Multi-signal rank + LLM rerank | ~1-2s | Best quality |
| **Hybrid** | Local + Live TMDB → Merge → Rank | ~2-3s | Finding new movies |

### Personalization Layer

**User Taste Vector:**
```python
# Exponential moving average of liked movie embeddings
taste_vector = α * liked_embedding + (1-α) * taste_vector
```

**Genre Weights:**
```python
# Learned from user behavior
{"Action": 1.2, "Horror": 0.5, "Comedy": 1.0}
```

### Additional Features

- 🎭 **Browse by Mood** — Curated mood categories
- 🎲 **Surprise Me** — Random discovery with personalization
- 📋 **Watchlists** — Save movies for later
- 👍👎 **Like/Dislike** — Train preferences
- 🎬 **Trailers** — Play YouTube trailers in-app

---

## 7. V2 Demo

### Live: [movie-recs-v2-production.up.railway.app](https://movie-recs-v2-production.up.railway.app)

**Features to Show:**

| Feature | What It Demonstrates |
|---------|---------------------|
| Hero carousel | Rich TMDB data, Netflix-style UI |
| Search modes | Fast vs Smart vs Hybrid |
| Browse by Mood | Clustering-based discovery |
| Surprise Me | Personalized random |
| Movie details | Trailers, ratings, cast |
| Sign in | Google OAuth, preference sync |

---

## 8. Evaluation

### Metrics (From Course: Week 6-7)

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **nDCG@k** | DCG@k / IDCG@k | Ranking quality (position-weighted) |
| **Precision@k** | Relevant / k | Result cleanliness |
| **Recall@k** | Found / Total relevant | Completeness |
| **Hit Rate@k** | ≥1 relevant? | Robustness |
| **MRR** | 1 / rank of first relevant | Speed to good result |

### V1 Results

```
Ground Truth Evaluation:
Pipeline        nDCG@k    Precision@k  Hit Rate   MRR
------------------------------------------------------------
retrieval       0.0635    0.0267       0.1333     0.0411
ranked          0.0606    0.0200       0.1000     0.0483

LLM-Judged Evaluation:
Pipeline        nDCG@k    Precision@k  Hit Rate   MRR
------------------------------------------------------------
retrieval       0.9556    0.8800       1.0000     0.9500
ranked          0.9579    0.8200       1.0000     0.9500
```

### Key Insight

> **Low ground truth ≠ bad system.** The ground truth is narrow; LLM-judge shows the system finds semantically relevant movies that weren't pre-defined.

### V2 Evaluation Additions

- **Online Analytics:** Click-through rate, dwell time, action rate
- **A/B Testing:** Compare search modes with real users
- **Zero Results Rate:** Track queries returning nothing
- **Query Reformulation:** Track users who search again

---

## 9. Deployment

### V1: Streamlit Cloud

- **Cost:** Free
- **Setup:** Connect GitHub, deploy automatically
- **Limitation:** Embeddings in Git LFS (~120MB)

### V2: Railway

```
┌─────────────────────────────────────────────────────────────┐
│                        Railway Cloud                         │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │    React     │    │   FastAPI    │    │  PostgreSQL  │  │
│  │   Frontend   │←──→│   Backend    │←──→│  + pgvector  │  │
│  │   (Nginx)    │    │  (Gunicorn)  │    │              │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

- **Cost:** ~$5-20/month
- **CI/CD:** GitHub Actions → auto-deploy on push
- **Monitoring:** Sentry for errors, structured logging

### Ethical Considerations

| Concern | Risk | Mitigation |
|---------|------|------------|
| **Bias** | Embeddings may reflect training data bias | Monitor diversity metrics |
| **Filter Bubbles** | Personalization creates echo chambers | "Surprise Me" feature |
| **Privacy** | User preference data is sensitive | OAuth (no passwords), minimal data |
| **Transparency** | Users don't understand recommendations | LLM explanations, help modal |

---

## 10. Key Learnings

### From Course Fundamentals to Production

| Course Concept | Class Example | Project Application |
|----------------|---------------|---------------------|
| **k-NN** | Loan classification | Semantic movie search |
| **Feature Engineering** | Income, credit score | Text corpus construction |
| **Evaluation Metrics** | Accuracy, precision | nDCG, MRR, Hit Rate |
| **Ensemble Methods** | Random Forest | Multi-signal ranking |
| **Textual Analytics** | Bag of words | Neural embeddings |

### Technical Growth

1. **Full-Stack:** React + FastAPI + PostgreSQL
2. **ML Infrastructure:** Vector databases, embedding pipelines
3. **Production Systems:** Docker, CI/CD, monitoring
4. **Product Thinking:** UX, analytics, experimentation

### What I Would Do Differently

1. Start with pgvector instead of migrating from FAISS
2. Implement analytics from day one
3. Build mobile-first responsive design earlier

---

## Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                         PROJECT JOURNEY                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Course: k-NN Fundamentals                                      │
│           ↓                                                      │
│   Idea: "What if k-NN worked on movie descriptions?"             │
│           ↓                                                      │
│   V1: Prototype with FAISS + Streamlit                           │
│       ✓ Proved semantic search works                             │
│       ✓ Built evaluation framework                               │
│           ↓                                                      │
│   V2: Production with pgvector + React + FastAPI                 │
│       ✓ Scalable architecture                                    │
│       ✓ Real-time data                                           │
│       ✓ User personalization                                     │
│       ✓ Netflix-style experience                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Links & Resources

| Resource | URL |
|----------|-----|
| V1 Live Demo | [movievibesai.streamlit.app](https://movievibesai.streamlit.app) |
| V2 Live Demo | [movie-recs-v2-production.up.railway.app](https://movie-recs-v2-production.up.railway.app) |
| V1 GitHub | [github.com/enkrumah/movie-recs](https://github.com/enkrumah/movie-recs) |
| V2 GitHub | [github.com/enkrumah/movie-recs-v2](https://github.com/enkrumah/movie-recs-v2) |

---

*Report Date: December 3, 2025*

