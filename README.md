# 🎬 AI Movie Recommender (MVP)

A small educational project to learn how LLMs and embeddings power recommendation systems.
Built in **Python** with **Streamlit**, **OpenAI Embeddings**, and **k-Nearest Neighbors (kNN)**.

---

## 🔧 Setup Steps Done

1. **Environment**
   - Created `conda` env `movie-recs`
   - Installed dependencies from `requirements.txt`
   - Verified Streamlit runs (`streamlit run app.py`)

2. **Dataset**
   - Downloaded MovieLens *latest-small* dataset → `data/ml-latest-small/`
   - Loaded CSVs with Pandas for exploration.

3. **Preprocessing**
   - Built a unified `text_embed` field combining movie title + genres.

4. **Secrets**
   - Added `.env` with `OPENAI_API_KEY=sk-...`
   - Tested with `load_dotenv()`.

5. **Embeddings**
   - Generated OpenAI embeddings (`text-embedding-3-large`)
   - Saved outputs to `artifacts/`:
     - `movie_vectors.npy`
     - `movie_ids.json`
     - `movie_texts.json`
   - Sanity-checked dimensions and sample IDs.

---

## 🚀 Next Steps

**M5 – Retrieval**
- Load vectors and fit a `NearestNeighbors(metric="cosine")`.
- Test sample queries in console.

**M6 – UI Integration**
- Connect kNN to Streamlit text box.
- Display top-k movie recommendations.

**M7 – LLM Enhancement**
- Parse mood / constraints.
- Generate “why this fits” explanations via GPT.

**M8 – Logging & Feedback**
- Save user queries and results.
- Prepare for small Streamlit Cloud deploy.

---

## 🧠 Learning Goals
- Practice building a multimodal recommendation workflow.
- Use embeddings + KNN for semantic search.
- Integrate OpenAI models into a small full-stack prototype.

---

_Progress: ~60% complete (Embeddings done; Retrieval next)._
