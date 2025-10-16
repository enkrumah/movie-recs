# 🎬 AI Movie Recommendation System (MVP)

This project builds an **AI-powered movie recommender** using OpenAI embeddings, k-Nearest Neighbors, and Streamlit.  
The goal is to demonstrate how natural language can drive movie discovery — users can type a description or title, and the model retrieves similar films based on semantic similarity.

---

## 🧱 Project Summary

This app is designed to:
- Use **OpenAI embeddings** to numerically represent movie descriptions.
- Store and search those embeddings efficiently using **vector similarity**.
- Serve recommendations through an interactive **Streamlit UI**.
- Be modular enough for future extensions (e.g., RAG pipelines, user-based personalization).

---

## 🧩 Milestones & Summary (Completed so far)

| Stage | What We Did | Why It Mattered |
|-------|--------------|----------------|
| **M1 – Environment Setup** | Created virtual environment, folder structure (`src/`, `data/`, `artifacts/`), and installed dependencies | Establishes a clean, modular workspace for reproducibility |
| **M2 – Dependencies & Data** | Installed `openai`, `pandas`, `numpy`, `scikit-learn`, and `streamlit`; loaded and cleaned movie dataset | Ensures the environment can support embeddings, vector search, and a web app |
| **M3 – Data Assembly** | Built `movie_texts.json` combining movie title, year, and genre | Provides natural language descriptions for the embedding model |
| **M4 – Embedding Index** | Generated embeddings via OpenAI API, handled API quota/rate errors, stored in `artifacts/movie_vectors.npy` | Converts each movie into a numerical vector for similarity search |
| **Repo Hygiene & Git Setup** | Configured `.env`, `.gitignore`, removed large files, and pushed a clean repo to GitHub | Prevents leaking secrets, large files, or intermediate artifacts |
| **M5 – Retrieval System (kNN)** | Load vectors and metadata to compute top-5 similar movies per query |
---

## Updates
## 🏁 Milestone 5 – Interactive Streamlit App

We’ve successfully built and tested the **interactive front-end** for the AI Movie Recommender.

### 🔹 Key Achievements
- Integrated **Streamlit UI** with the embedding retrieval system.
- Implemented **Cosine KNN** similarity search for movie embeddings.
- Added **Quick Examples** and state management using `st.session_state`.
- Results are now dynamically displayed in a clean, responsive grid.
- End-to-end recommendation pipeline works locally.

### 🧩 Current Project Flow
**User Input → Embedding Generation → Cosine Similarity Search → Top Movie Results (Streamlit Display)**

### 🎬 Example Query
> “romantic drama about memory”  
✅ Output: *Remember Me (2010), Eternal Sunshine of the Spotless Mind (2004), The Romantics (2010)*

---

Next up: **Milestone 6 — LLM Summarization Layer**  
We’ll use an OpenAI model to summarize retrieved movies into a single, human-like movie mood description.


## 🧭 Next Steps

| Upcoming Stage | Goal |
|----------------|------|
| **M6 – Streamlit UI** | Connect retrieval logic to an app interface |
| **M7 – LLM Integration** | Add natural-language explanations for recommendations |
| **M8 – Deployment** | Host app on Streamlit Cloud or GitHub Pages with a public demo link |

---

## 🚀 How to Run (update as you go)

```bash
# Activate environment
conda activate movie-recs

# Run Streamlit app
streamlit run app.py


---

| Date       | Update                                                           |
| ---------- | ---------------------------------------------------------------- |
| 2025-10-16 | Finished M4, cleaned repo, added .gitignore, committed to GitHub |
| 2025-10-15 | Dataset finalized, initial environment setup complete            |
| 2025-10-14 | Repository initialized and directory structure created           |


Tech Stack
Core: Python, Streamlit, OpenAI API, scikit-learn
Data Handling: Pandas, NumPy
Infrastructure: Git, Conda
Future Additions: Pinecone / FAISS (vector DB), LangChain integration




## 🧠 Learning Goals
- Practice building a multimodal recommendation workflow.
- Use embeddings + KNN for semantic search.
- Integrate OpenAI models into a small full-stack prototype.

This project isn’t just about building a recommender — it’s about understanding how semantic similarity, vector representations, and retrieval pipelines can evolve into full-scale AI products.
Every milestone mirrors the real-world process of building AI-driven apps: from prototype to deployment with transparency, modularity, and best practices.

---

## Author
Ebenezer Nkrumah Amankwah
MBA Candidate @ Emory Goizueta | Product & AI Systems Builder
GitHub: @enkrumah
