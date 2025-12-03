# 🎬 AI Movie Recommendation System — Presentation Script

**Duration:** 8 minutes  
**Date:** December 3, 2025  
**Presenter:** Ebenezer Nkrumah Amankwah

---

## Slide 1: Title (15 seconds)

> "Hi everyone, I'm Ebenezer, and today I'm presenting my project on building an AI-powered movie recommendation system that understands natural language."

**Live Demos:**
- V1: [movievibesai.streamlit.app](https://movievibesai.streamlit.app)
- V2: [vybemoves.up.railway.app](https://vybemoves.up.railway.app)

---

## Slide 2: The Problem (45 seconds)

### What I'm Solving

> "Have you ever spent 20 minutes scrolling through Netflix, unable to decide what to watch? That's called browsing paralysis."

**The Gap:**
- Traditional systems use keywords: "Action" → action movies
- Users think in vibes: *"Something like Inception but more emotional"*

**My Solution:**
- Build a system that understands **meaning**, not just keywords
- Users describe mood, themes, feelings → get relevant recommendations

---

## Slide 3: Course Connection — k-NN to Vector Search (1 minute)

### The Foundation

> "This project is built directly on the k-NN algorithm we learned in Weeks 4-5, but applied to a different problem."

**Traditional k-NN (What We Learned):**
```
Tabular data → Calculate Euclidean distance → Find k closest neighbors
```

**My Adaptation:**
```
Text → Convert to embeddings → Calculate cosine similarity → Find k most similar
```

**Key Insight:**
- Same principle: "similar things are close together"
- Different space: 3072-dimensional embedding space instead of feature space
- The "distance" now measures **semantic similarity**

---

## Slide 4: Data Pipeline (1 minute)

### From Raw Data to Searchable Index

> "Let me walk you through how I prepared the data."

**Source:** MovieLens dataset (9,742 movies)

**Pipeline:**
```
Raw CSV → Extract → Parse → Normalize → Embed → Index
```

**Step-by-Step:**

1. **Extract:** `movieId`, `title`, `genres`
2. **Parse Title:** `"Se7en (1995)"` → title: `"Se7en"`, year: `"1995"`
3. **Normalize Genres:** `"Crime|Thriller"` → `"Crime, Thriller"`
4. **Build Text:** `"Se7en (1995) — Genres: Crime, Thriller"`
5. **Generate Embeddings:** OpenAI text-embedding-3-large (3072 dimensions)
6. **L2 Normalize:** For cosine similarity via inner product
7. **Index with FAISS:** Facebook AI's vector search library

---

## Slide 5: V1 Demo (2 minutes)

### Live Demo at [movievibesai.streamlit.app](https://movievibesai.streamlit.app)

> "Let me show you the prototype in action."

**Demo Script:**

1. **Basic Query:**
   - Type: `"mind-bending sci-fi"`
   - Show results (Inception, Matrix, etc.)
   - Point out: "Notice how it found thematically similar movies"

2. **Mood-Based Query:**
   - Type: `"feel-good sports drama, true story"`
   - Show results
   - Point out: "It understands 'feel-good' and 'true story' — not just genres"

3. **Smart Mode Toggle:**
   - Enable "✨ Smart Mode"
   - Same query, show improved ranking
   - Explain: "This adds LLM reranking for better results"

4. **Show Explanation:**
   - Point to "Why these match your vibe" section
   - "The system explains its reasoning"

---

## Slide 6: Multi-Signal Ranking (30 seconds)

### Beyond Simple Similarity

> "Smart Mode uses five signals, not just embedding similarity."

| Signal | Weight | What It Does |
|--------|--------|--------------|
| Similarity | 45% | How close in embedding space |
| Recency | 10% | Boost newer movies |
| Genre Match | 15% | Query keywords → genre alignment |
| Keyword Match | 10% | Exact word overlap |
| LLM Score | 20% | GPT judges relevance |

**This is an ensemble approach** — combining signals like we discussed in Week 13-14.

---

## Slide 7: Evolution to V2 (1 minute)

### Why I Built a Second Version

> "V1 proved the concept works. V2 makes it production-ready."

**V1 Limitations:**
- Static data (MovieLens from 2018)
- In-memory only (FAISS)
- No user accounts
- Basic UI (Streamlit)

**V2 Improvements:**

| Aspect | V1 | V2 |
|--------|----|----|
| Data | Static CSV | Live TMDB API |
| Storage | FAISS (memory) | pgvector (PostgreSQL) |
| Frontend | Streamlit | React (Netflix-style) |
| Auth | None | Google OAuth |
| Features | Search only | Personalization, watchlists, mood browse |

---

## Slide 8: V2 Demo (2 minutes)

### Live Demo at [vybemoves.up.railway.app](https://vybemoves.up.railway.app)

> "Now let me show you the production version."

**Demo Script:**

1. **Homepage:**
   - Show Netflix-style hero carousel
   - "Real movie posters from TMDB"

2. **Search Modes:**
   - Fast Search: `"romantic comedy"`
   - Smart Search: Same query, show LLM-enhanced results
   - Point out mode badges and explanations

3. **Mood Browse:**
   - Click "Browse by Mood"
   - Select a mood chip (e.g., "Feel-Good")
   - Show curated results

4. **Surprise Me:**
   - Click "Surprise Me"
   - Show random discovery feature

5. **Sign In (if time):**
   - Show Google OAuth
   - "Users can save preferences, build watchlists"

---

## Slide 9: Evaluation Results (30 seconds)

### How I Know It Works

> "I built a full evaluation framework using metrics from our course."

**Ground Truth vs LLM-Judge:**

| Method | nDCG@5 | Precision@5 | Hit Rate |
|--------|--------|-------------|----------|
| Ground Truth | 0.06 | 0.03 | 0.13 |
| LLM-Judge | 0.96 | 0.88 | 1.00 |

**Key Insight:**
- Low ground truth = narrow test set, not bad retrieval
- High LLM-judge = system finds semantically relevant movies
- Traditional metrics underestimate semantic search quality

---

## Slide 10: Key Learnings (45 seconds)

### What I Learned

> "Here's what I took away from this project."

**1. Fundamentals Transfer:**
- k-NN principles work in embedding space
- "Similar things are close" is universal

**2. Feature Engineering is Text Engineering:**
- Building good text corpus = modern feature engineering
- Rich text → better embeddings → better search

**3. Evaluation is Hard:**
- Traditional metrics miss semantic relevance
- LLM-as-judge is the future for semantic systems

**4. Production is Different:**
- Prototype → Production requires new thinking
- Scalability, user experience, monitoring

---

## Slide 11: Conclusion (15 seconds)

### Thank You

> "In summary, I applied k-NN fundamentals to build a semantic movie search system, then scaled it to production with modern tools."

**Links:**
- V1 Demo: [movievibesai.streamlit.app](https://movievibesai.streamlit.app)
- V2 Demo: [vybemoves.up.railway.app](https://vybemoves.up.railway.app)
- V1 Code: [github.com/enkrumah/movie-recs](https://github.com/enkrumah/movie-recs)
- V2 Code: [github.com/enkrumah/movie-recs-v2](https://github.com/enkrumah/movie-recs-v2)

> "Happy to take questions!"

---

## Backup Slides / Q&A Prep

### Q: Why OpenAI embeddings instead of training your own?

> "State-of-the-art quality without needing millions of training examples. For a movie catalog of 10K, training custom embeddings would likely underperform."

### Q: How does it handle movies not in the database?

> "V2's Hybrid mode queries TMDB live, so it can find movies not in our local index."

### Q: What about cold start for new users?

> "New users get popularity-based recommendations. Once they like/dislike movies, we build their taste vector."

### Q: Cost to run?

> "V1 on Streamlit Cloud: Free. V2 on Railway: ~$5-20/month. OpenAI API: ~$0.001 per search."

---

## Timing Checklist

| Section | Target | Cumulative |
|---------|--------|------------|
| Title | 0:15 | 0:15 |
| Problem | 0:45 | 1:00 |
| Course Connection | 1:00 | 2:00 |
| Data Pipeline | 1:00 | 3:00 |
| V1 Demo | 2:00 | 5:00 |
| Multi-Signal | 0:30 | 5:30 |
| V2 Evolution | 1:00 | 6:30 |
| V2 Demo | 2:00 | 8:30 |
| Evaluation | 0:30 | 9:00 |
| Learnings | 0:45 | 9:45 |
| Conclusion | 0:15 | 10:00 |

**Note:** Demos can be shortened if running over. Skip V2 demo details if needed.

---

*Good luck! You've got this! 🎬*

