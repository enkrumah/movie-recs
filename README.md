# 🎬 AI Movie Recommendation System (MVP)

An AI-based movie recommender built with **OpenAI embeddings**, **FAISS vector search**, **multi-signal ranking**, and **Streamlit**.  
Users describe the kind of movie they want to watch, and the system retrieves semantically relevant films — then explains _why_ they match.

---

# 🧠 System Overview

```
User Query → Embedding → FAISS Retrieval → Multi-Signal Ranking → LLM Explanation → Results
```

**Example Query:**  
"romantic drama about memory" →  
Eternal Sunshine of the Spotless Mind, Remember Me, The Romantics.

**Smart Mode Pipeline:**

```
Retrieve 20 candidates → Fast Rank (4 signals) → LLM Rerank (5 signals) → Return Top 5
```

---

# 🗂 Project Structure

```
movie-recs/
│
├── app.py                     # Streamlit UI (main app)
├── embed_index.py             # Embedding builder script
│
├── artifacts/                 # Embeddings (tracked via Git LFS)
│   ├── movie_vectors.npy
│   ├── movie_ids.json
│   └── movie_texts.json
│
├── data/
│   ├── ml-latest-small/       # MovieLens dataset
│   ├── ml-latest-small.zip
│   └── eval_dataset.json      # Evaluation queries + ground truth
│
├── src/
│   ├── client.py              # Shared OpenAI client singleton
│   ├── data_loader.py         # Loads dataset + metadata
│   ├── recommender.py         # FAISS similarity search
│   ├── ranker.py              # Multi-signal ranking layer
│   ├── evaluation.py          # Retrieval metrics (nDCG, MRR, etc.)
│   └── llm.py                 # Explanation + suggestions (OpenAI)
│
├── .streamlit/
│   └── config.toml            # Streamlit theme configuration
│
├── README.md
└── requirements.txt
```

---

# 📌 Project Milestones

| #   | Milestone                | Description                                          | Status |
| --- | ------------------------ | ---------------------------------------------------- | ------ |
| M1  | Environment Setup        | Conda env, folder structure, dependencies.           | ✅     |
| M2  | Data Loading             | Loaded MovieLens dataset and extracted metadata.     | ✅     |
| M3  | Data Assembly            | Generated `movie_texts.json` as embedding input.     | ✅     |
| M4  | Embedding Index          | Produced `movie_vectors.npy`, `movie_ids.json`.      | ✅     |
| M5  | UI Integration           | Streamlit interface + quick examples.                | ✅     |
| M6  | LLM Reasoning Layer      | Explanations + suggestions via OpenAI.               | ✅     |
| M7  | Repo Hygiene             | Cleaned repo, added `.gitignore`, removed artifacts. | ✅     |
| M8  | **Ranking + Evaluation** | Multi-signal ranker + retrieval metrics.             | ✅     |
| M9  | **Deployment**           | Streamlit Cloud or HuggingFace Spaces.               | 🔜     |

---

# 🔐 API Key Configuration

Your OpenAI key is **not included** in the repo.

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-xxxx
```

This is loaded automatically via `src/client.py`.

---

# 🧩 Embedding Artifacts

Artifacts are stored via **Git LFS** (Large File Storage) and will download automatically when you clone.

If you need to regenerate embeddings:

```bash
python embed_index.py
```

This creates:

```
artifacts/
    movie_vectors.npy    # 9,742 movies × 3,072 dimensions (~120MB)
    movie_ids.json
    movie_texts.json
```

**Note:** The `.npy` file is tracked via Git LFS. Run `git lfs install` before cloning if you don't have LFS set up.

---

# ▶️ Running the Application

```bash
conda activate movie-recs
streamlit run app.py
```

A local browser window will open automatically.

### Smart Mode Toggle

The UI includes a **✨ Smart Mode** checkbox:

- **Off (default):** Fast retrieval by embedding similarity (~200ms)
- **On:** Three-stage ranking with LLM scoring (~1-2s, more accurate)

---

# ✨ Current Features

### ✅ Natural Language Search

Users describe the movie they want — the system retrieves and ranks matches.

### ✅ FAISS Vector Search

Efficient nearest-neighbor search over normalized embeddings using Facebook's FAISS library.

### ✅ Multi-Signal Ranking (Smart Mode)

Five ranking signals for improved relevance:

| Signal        | Weight | Description                    |
| ------------- | ------ | ------------------------------ |
| Similarity    | 45%    | Embedding cosine similarity    |
| Genre Match   | 15%    | Query-genre keyword alignment  |
| Recency       | 10%    | Newer movies boosted           |
| Keyword Match | 10%    | Lexical overlap with query     |
| LLM Score     | 20%    | GPT-4o-mini relevance judgment |

### ✅ LLM "Why This Fits Your Vibe"

Short explanations describing why the movies match the user's request.

### ✅ AI Prompt Suggestions

If users type a short fragment, the system suggests full search prompts.

### ✅ Clean, Responsive UI

Two-column movie grid, quick examples, and session-state behavior.

### ✅ Latency Tracking

Real-time performance metrics displayed after each search:
```
⚡ Found 5 movies in 0.43s (Search: 0.31s → Summary: 0.12s)
```

### ✅ User Feedback Collection

Embedded Google Form for collecting user feedback, with responses automatically saved to Google Sheets.

---

# 📊 Evaluation Suite

The project includes a complete offline evaluation suite to measure retrieval and ranking quality.

## Running Evaluations

```bash
# Ground truth evaluation (compares against known relevant movies)
python -m src.evaluation

# With verbose per-query details
python -m src.evaluation -v

# Include LLM-enhanced ranking in comparison
python -m src.evaluation --llm

# LLM-based relevance evaluation (no ground truth needed)
python -m src.evaluation --llm-judge

# See detailed LLM judgments with reasoning
python -m src.evaluation --llm-judge --details
```

## Evaluation Methods

### 📋 Ground Truth Evaluation

Compares retrieved movies against a **pre-defined list** of relevant movies per query.

```
Query: "mind-bending sci-fi"
Ground Truth: [Matrix, Memento, Inception]
Retrieved: [Arrival, Limitless, Matrix, ...]
Match: 1/3 → Precision = 0.33
```

**Pros:** Reproducible, fast, free  
**Cons:** Misses valid alternatives not in the list

### 🤖 LLM-Based Relevance Evaluation

Uses GPT-4o-mini to **judge whether each retrieved movie is relevant** to the query.

```
Query: "mind-bending sci-fi"
Retrieved: [Arrival, Limitless, Matrix, ...]

LLM Judgment:
  ✅ Arrival [0.85] — "cerebral sci-fi about time"
  ✅ Limitless [0.70] — "mind-altering premise"
  ✅ Matrix [0.95] — "iconic mind-bending film"
```

**Pros:** Comprehensive, semantic understanding, graded relevance  
**Cons:** Costs ~$0.002/query, non-deterministic

## Evaluation Metrics

| Metric          | What It Measures                     | Example                                  |
| --------------- | ------------------------------------ | ---------------------------------------- |
| **nDCG@k**      | Ranking quality (position-sensitive) | Are the best movies at the top?          |
| **Precision@k** | Cleanliness of top-k                 | What % of top-5 are relevant?            |
| **Recall@k**    | Retrieval completeness               | What % of relevant movies did we find?   |
| **Hit Rate@k**  | Robustness                           | Did we return at least 1 relevant movie? |
| **MRR**         | First relevant result                | How quickly does a good movie appear?    |
| **Coverage**    | Catalog diversity                    | What % of catalog ever gets recommended? |

## Sample Results

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

**Key Insight:** Low ground truth scores reflect narrow test coverage, not poor retrieval. LLM evaluation shows the system finds **semantically relevant movies** that weren't in the pre-defined ground truth.

---

# 🧮 Ranking Architecture

## Two-Stage Pipeline (Default)

```
User Query → FAISS Retrieve 20 → Return Top 5
```

## Three-Stage Pipeline (Smart Mode)

```
User Query → FAISS Retrieve 20 → Fast Rank to 10 → LLM Score 10 → Return Top 5
```

### Ranking Signals

| Signal            | Implementation                               | Weight |
| ----------------- | -------------------------------------------- | ------ |
| **Similarity**    | Cosine similarity from embeddings            | 45%    |
| **Recency**       | Year extraction, sigmoid boost for post-2000 | 10%    |
| **Genre Match**   | Keyword detection → genre alignment          | 15%    |
| **Keyword Match** | Lexical overlap (stopwords removed)          | 10%    |
| **LLM Score**     | GPT-4o-mini relevance judgment (batched)     | 20%    |

---

# 🚀 Deployment (M9)

Two recommended options:

### 1. **Streamlit Cloud** (Recommended)

- Free, fast, perfect for demos
- Direct GitHub integration
- Store API key in Streamlit Secrets

### 2. **HuggingFace Spaces**

- Good compute options
- Supports Streamlit SDK

**Note:** Use Git LFS for the embeddings file (`movie_vectors.npy` is ~120MB).

---

# 🧰 Tech Stack

| Layer             | Technology                                             |
| ----------------- | ------------------------------------------------------ |
| **Core**          | Python, NumPy, Pandas                                  |
| **Vector Search** | FAISS (`faiss-cpu`)                                    |
| **Embeddings**    | OpenAI `text-embedding-3-large` (3072-dim)             |
| **LLM**           | OpenAI `gpt-4o-mini`                                   |
| **UI**            | Streamlit                                              |
| **Future**        | FastAPI, React, Pinecone, two-stage retrieval at scale |

---

# 🎯 Learning Goals

- Build a semantic retrieval system from scratch
- Understand embeddings, vector search, and ranking signals
- Implement evaluation metrics used in real-world ranking teams
- Compare ground truth vs LLM-based evaluation approaches
- Build an end-to-end AI product from dataset → UI → reasoning layer

---

# 👤 Author

**Ebenezer Nkrumah Amankwah**  
MBA Candidate @ Emory Goizueta  
Product & AI Systems Builder  
GitHub: **@enkrumah**
