# 🎬 AI Movie Recommendation System (MVP)

An AI-based movie recommender built with **OpenAI embeddings**, **cosine similarity retrieval**, and **Streamlit**.  
Users describe the kind of movie they want to watch, and the system retrieves semantically relevant films — then explains _why_ they match.

---

# 🧠 System Overview

**User Input → Embedding Generation → Cosine Similarity Search → LLM Explanation → Ranked Output**

Example Query:  
“romantic drama about memory” →  
Eternal Sunshine of the Spotless Mind, Remember Me, The Romantics.

---

# 🗂 Project Structure

```
movie-recs/
│
├── app.py                     # Streamlit UI (main app)
│
├── artifacts/                 # Generated artifacts (not tracked in Git)
│   ├── movie_vectors.npy
│   ├── movie_ids.json
│   └── movie_texts.json
│
├── data/
│   ├── ml-latest-small/       # MovieLens dataset
│   ├── ml-latest-small.zip
│   └── embed_index.py         # Embedding builder script
│
├── src/
│   ├── data_loader.py         # Loads dataset + metadata
│   ├── recommender.py         # Similarity search
│   └── llm.py                 # Explanation + suggestions (OpenAI)
│
├── README.md
└── requirements.txt
```

---

# 📌 Project Milestones

| #   | Milestone                | Description                                          | Status             |
| --- | ------------------------ | ---------------------------------------------------- | ------------------ |
| M1  | Environment Setup        | Conda env, folder structure, dependencies.           | ✅                 |
| M2  | Data Loading             | Loaded MovieLens dataset and extracted metadata.     | ✅                 |
| M3  | Data Assembly            | Generated `movie_texts.json` as embedding input.     | ✅                 |
| M4  | Embedding Index          | Produced `movie_vectors.npy`, `movie_ids.json`.      | ✅                 |
| M5  | UI Integration           | Streamlit interface + quick examples.                | ✅                 |
| M6  | LLM Reasoning Layer      | Explanations + suggestions via OpenAI.               | ✅                 |
| M7  | Repo Hygiene             | Cleaned repo, added `.gitignore`, removed artifacts. | ✅                 |
| M8  | **Ranking + Evaluation** | Multi-signal ranker + retrieval metrics.             | 🔜 (current focus) |
| M9  | **Deployment**           | Streamlit Cloud or HuggingFace Spaces.               | 🔜                 |

---

# 🔐 API Key Configuration

Your OpenAI key is **not included** in the repo.

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-xxxx
```

This is loaded automatically via `src/llm.py`.

Without this, prompts, suggestions, and explanations will fail.

---

# 🧩 Generating Embedding Artifacts

Artifacts are **not stored in Git** and must be generated locally.

Run:

```
python data/embed_index.py
```

This creates:

```
artifacts/
    movie_vectors.npy
    movie_ids.json
    movie_texts.json
```

These files are required before running `app.py`.

---

# ▶️ Running the Application

```
conda activate movie-recs
streamlit run app.py
```

A local browser window will open automatically.

---

# ✨ Current Features

### ✅ Natural Language Search

Users describe the movie they want — the system retrieves and ranks matches.

### ✅ Cosine Similarity Retrieval

Efficient nearest-neighbor search over normalized embeddings.

### ✅ LLM “Why This Fits Your Vibe”

Short explanations describing why the movies match the user's request.

### ✅ AI Prompt Suggestions

If users type a short fragment, the system suggests full search prompts.

### ✅ Clean, Responsive UI

Two-column movie grid, quick examples, and session-state behavior.

---

# 🧮 Upcoming Feature: Ranking & Evaluation (M8)

The current system ranks solely by **similarity**.  
M8 introduces a **multi-signal ranking layer**:

### Ranking Signals

| Signal             | Description                                | Why it matters                                    |
| ------------------ | ------------------------------------------ | ------------------------------------------------- |
| Similarity         | Embedding-based relevance                  | Core relevance driver                             |
| Recency            | Extracted from movie release year          | Users prefer modern content                       |
| Genre Match        | Genre alignment with user query            | Prevents semantically-close but genre-wrong films |
| Keyword Match      | Lexical match between query + title/genres | Useful for short or ambiguous queries             |
| LLM Semantic Score | Quality of explanation alignment           | Adds interpretability & nuance                    |

The output becomes a **weighted ranking score**, not just cosine similarity.

---

# 📊 Evaluation Suite (M8)

These metrics will be added:

- **nDCG@k** – industry standard for ranking quality
- **Precision@k / Recall@k**
- **Hit Rate@k**
- **MRR** (optional)
- **Coverage** (optional)

These align with interview expectations for retrieval/ranking roles.

---

# 🚀 Deployment (M9)

Two recommended options:

### 1. **Streamlit Cloud**

Pros: fast, free, perfect for demos.  
Cons: limited compute.

### 2. **HuggingFace Spaces**

Pros: good GPU/CPU options, clean UI hosting.  
Cons: slightly more setup.

Both work with local artifacts — or with a future MCP-powered remote database.

---

# 🧰 Tech Stack

**Core:** Python, NumPy, Pandas  
**Retrieval:** OpenAI Embeddings, cosine similarity  
**LLM Reasoning:** OpenAI Responses API  
**UI:** Streamlit  
**Future:** FastAPI, React, Pinecone, MCP Connectors, FAISS, two-stage retrieval

---

# 🎯 Learning Goals

- Build a semantic retrieval system from scratch.
- Understand embeddings, vector search, and ranking signals.
- Implement evaluation metrics used in real-world ranking teams.
- Build an end-to-end AI product from dataset → UI → reasoning layer.
- Prepare for a scalable V2 architecture with proper layering.

---

# 👤 Author

**Ebenezer Nkrumah Amankwah**  
MBA Candidate @ Emory Goizueta  
Product & AI Systems Builder  
GitHub: **@enkrumah**
