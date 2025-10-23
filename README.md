# 🎬 AI Movie Recommendation System (MVP)

An **AI-powered movie recommender** built with OpenAI embeddings, cosine-similarity retrieval, and Streamlit.  
This project demonstrates how natural language understanding can drive movie discovery — users simply describe what they’re in the mood for, and the model retrieves semantically similar films.

---

## 🧱 Project Summary

This app:
- Uses **OpenAI embeddings** to numerically represent movie descriptions.
- Performs efficient **vector similarity search** via cosine distance.
- Serves recommendations through an interactive **Streamlit UI**.
- Is modular for future extensions (e.g., RAG pipelines, personalization, or live databases).

---

### 🧩 System Flow
**User Input → Embedding Generation → Cosine Similarity Search → LLM Summary → Top Movie Recommendations**

### 🎥 Example Query
> “romantic drama about memory”  
✅ Output: *Remember Me (2010), Eternal Sunshine of the Spotless Mind (2004), The Romantics (2010)*

---

## 🚀 Development Milestones

| # | Milestone | Description | Status |
|---|------------|-------------|--------|
| **M1** | Environment Setup | Created virtual environment, folder structure (`src/`, `data/`, `artifacts/`), installed dependencies. | ✅ |
| **M2** | Dependencies & Data | Installed `openai`, `pandas`, `numpy`, `scikit-learn`, and `streamlit`; loaded and cleaned movie dataset. | ✅ |
| **M3** | Data Assembly | Built `movie_texts.json` combining movie title, year, and genre for embedding input. | ✅ |
| **M4** | Embedding Index | Generated and stored embeddings via OpenAI API (`artifacts/movie_vectors.npy`); handled rate limits. | ✅ |
| **M5** | Interactive Streamlit App | Integrated front-end with retrieval engine using cosine similarity and KNN search. Added quick examples and responsive layout. | ✅ |
| **M6** | LLM Reasoning Layer | Added `src/llm.py` to generate natural-language summaries explaining *why* the recommendations fit. Integrated results directly above movie cards in the UI. | ✅ |
| **M7** | Repo Hygiene & Git Setup | Configured `.env`, `.gitignore`, cleaned repo, and removed large artifacts. | ✅ |
| **M8** | Deployment (Next) | Host on Streamlit Cloud or Hugging Face Spaces for a public demo. | 🔜 |

---

## Key Updates (M5–M6)

### 🏁 Interactive Streamlit App
- Integrated **Streamlit UI** with the retrieval pipeline.  
- Added **Quick Examples** and `st.session_state` for dynamic interaction.  
- Results display in a clean, responsive grid.  
- End-to-end recommendation pipeline now works locally.

### LLM Reasoning Layer
- Added `src/llm.py` to generate short GPT-based explanations.  
- Integrated reasoning output above recommendations in `app.py`.  
- Summaries explain *why* retrieved movies match the user’s vibe.

---

## Next Steps

| Upcoming Stage | Goal |
|----------------|------|
| **M8 – Deployment** | Host app on Streamlit Cloud or Hugging Face Spaces for a public demo. |
| **Future Iterations** | Explore MCP connectors to query live databases; add personalization, per-movie insights, and analytics dashboards. |

---

## How to Run

```bash
# 1. Activate your environment
conda activate movie-recs

# 2. Run the Streamlit app
streamlit run app.py

---

| Date       | Update                                                           |
| ---------- | ---------------------------------------------------------------- |
| 2025-10-16 | Finished M4, cleaned repo, added .gitignore, committed to GitHub |
| 2025-10-15 | Dataset finalized, initial environment setup complete            |
| 2025-10-14 | Repository initialized and directory structure created           |


## Tech Stack
Core: Python, Streamlit, OpenAI API, scikit-learn
Data Handling: Pandas, NumPy
Version Control: Git, Conda
Future Additions: FAISS / Pinecone (vector DB), LangChain, MCP connectors


## Learning Goals
- Build a semantic recommender using embeddings + cosine KNN.
- Connect retrieval and reasoning layers (RAG-like architecture).
- Translate abstract AI concepts — embeddings, distance metrics, interpretability — into a functional product.

This project mirrors real-world AI development: structured milestones, modular design, and a focus on explainability and user experience.

---

## Author
Ebenezer Nkrumah Amankwah
MBA Candidate @ Emory Goizueta | Product & AI Systems Builder
GitHub: @enkrumah
