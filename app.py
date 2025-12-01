# -------------------------------
# 🎬 AI Movie Recommender (MVP)
# -------------------------------

import time
import streamlit as st
from src.data_loader import load_movies
from src.recommender import recommend_movies
from src.ranker import retrieve_rank_llm
from src.llm import summarize_recommendations


# ---- Page Config ----
st.set_page_config(page_title="AI Movie Recommender", page_icon="🎬", layout="wide")

# ---- Minimal Custom CSS ----
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        .info-box {
            background: #1e293b;
            border-left: 4px solid #e94560;
            border-radius: 8px;
            padding: 16px 20px;
            margin: 16px 0;
        }
        .info-box .info-title {
            font-weight: 600;
            font-size: 0.9rem;
            color: #e94560;
            margin-bottom: 8px;
        }
        .info-box .info-content {
            font-size: 0.9rem;
            color: #cbd5e1;
            line-height: 1.6;
        }
        
        .summary-box {
            background: #1e293b;
            border-radius: 10px;
            padding: 16px 20px;
            margin: 16px 0;
        }
        .summary-box .summary-title {
            font-weight: 600;
            font-size: 0.9rem;
            color: #f8fafc;
            margin-bottom: 8px;
        }
        .summary-box .summary-content {
            font-size: 0.9rem;
            color: #cbd5e1;
            line-height: 1.6;
        }
        
        .movie-card {
            background: #1e293b;
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 12px;
        }
        .movie-card .movie-title {
            font-weight: 600;
            font-size: 1rem;
            color: #f8fafc;
            margin-bottom: 6px;
            line-height: 1.4;
        }
        .movie-card .movie-score {
            font-size: 0.85rem;
            color: #94a3b8;
        }
        .movie-card .movie-details {
            font-size: 0.75rem;
            color: #64748b;
            margin-top: 6px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Main Content ----
st.title("🎬 AI Movie Recommender")
st.caption("Describe what you want to watch — by mood, theme, or vibe.")

# How it works box
st.markdown(
    """
    <div class="info-box">
        <div class="info-title">✨ How it works</div>
        <div class="info-content">
            This AI-powered tool uses <strong>semantic search</strong> to find movies that match your description — 
            not just keywords, but the actual meaning and vibe you're looking for. 
            It then explains why each movie fits your request.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Load movie data ----
@st.cache_data(show_spinner=False)
def get_movies_df():
    return load_movies()

movies_df = get_movies_df()
st.caption(f"✅ {len(movies_df):,} movies loaded")

# ---- Input controls ----
st.markdown("### Try it out")

if "query_text" not in st.session_state:
    st.session_state.query_text = ""
if "auto_search" not in st.session_state:
    st.session_state.auto_search = False

# Quick examples
st.caption("Quick examples:")
cols = st.columns(3)
examples = [
    "mind-bending sci-fi heist, nolan vibe",
    "feel-good sports drama, true story",
    "90s crime thriller, cat-and-mouse",
]
for i, ex in enumerate(examples):
    if cols[i].button(ex.title(), key=f"ex_{i}"):
        st.session_state.query_text = ex
        st.session_state.auto_search = True
        st.rerun()

# Text area
if "pending_text" in st.session_state:
    st.session_state.query_text = st.session_state.pop("pending_text")

user_text = st.text_area(
    "Describe what you're in the mood for:",
    height=100,
    placeholder="e.g., romantic drama about memory, atmospheric horror, uplifting documentary...",
    key="query_text",
)

# Controls
col1, col2 = st.columns([2, 1])
with col1:
    k = st.slider("Results", min_value=3, max_value=10, value=5, step=1)
with col2:
    smart_mode = st.checkbox("✨ Smart Mode", value=False, help="AI-powered re-ranking for better results")

# Search
@st.cache_data(show_spinner=False, ttl=600)
def _cached_summary(query: str, movie_texts: tuple, per_movie: bool):
    recs_for_llm = [{"text": t} for t in movie_texts]
    return summarize_recommendations(query, recs_for_llm, include_per_movie=per_movie)

do_search = st.button("Find Movies", type="primary") or st.session_state.auto_search

if do_search:
    st.session_state.auto_search = False
    query = st.session_state.query_text.strip()

    if not query:
        st.warning("Please describe what you're looking for.")
    else:
        retrieval_time = 0.0
        summary_time = 0.0
        
        if smart_mode:
            with st.spinner("Finding movies..."):
                try:
                    t_start = time.perf_counter()
                    recs = retrieve_rank_llm(query, k_retrieve=20, k_rerank=10, k_final=k)
                    retrieval_time = time.perf_counter() - t_start
                except Exception as e:
                    st.error(f"Error: {e}")
                    recs = []
        else:
            with st.spinner("Finding movies..."):
                try:
                    t_start = time.perf_counter()
                    recs = recommend_movies(query, k=k)
                    retrieval_time = time.perf_counter() - t_start
                except Exception as e:
                    st.error(f"Error: {e}")
                    recs = []

        if not recs:
            st.info("No results found. Try a different description.")
        else:
            movie_texts = tuple(r["text"] for r in recs)
            with st.spinner(""):
                t_start = time.perf_counter()
                summary_text, _ = _cached_summary(query, movie_texts, per_movie=False)
                summary_time = time.perf_counter() - t_start
            
            total_time = retrieval_time + summary_time
            st.caption(f"⚡ {len(recs)} movies in {total_time:.2f}s")
            
            # Summary
            if summary_text:
                st.markdown(
                    f"""
                    <div class="summary-box">
                        <div class="summary-title">🎯 Why these match your vibe</div>
                        <div class="summary-content">{summary_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Results
            st.markdown("### Top Picks")
            cols = st.columns(2)
            
            for idx, r in enumerate(recs):
                with cols[idx % 2]:
                    if smart_mode and "rank_score" in r:
                        score_label = "Score"
                        score_value = r["rank_score"]
                        signals = r.get("signals", {})
                        signal_details = f"Sim: {signals.get('sim_score', 0):.2f} | Genre: {signals.get('genre_score', 0):.2f} | LLM: {signals.get('llm_score', 0):.2f}"
                    else:
                        score_label = "Match"
                        score_value = r.get("similarity", 0)
                        signal_details = ""
                    
                    details_html = f'<div class="movie-details">{signal_details}</div>' if signal_details else ''
                    st.markdown(
                        f"""
                        <div class="movie-card">
                            <div class="movie-title">{r['text']}</div>
                            <div class="movie-score">{score_label}: {score_value:.2f}</div>
                            {details_html}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# ---- Feedback ----
st.markdown("---")
with st.expander("📝 Share Feedback"):
    st.markdown(
        """
        <iframe 
            src="https://docs.google.com/forms/d/e/1FAIpQLSfguNON5rWmnOfwI4wxX5yzGuXkIB40v84-wycYgk3WIrH8DQ/viewform?embedded=true" 
            width="100%" 
            height="500" 
            frameborder="0" 
            marginheight="0" 
            marginwidth="0"
            style="border-radius: 8px;">
            Loading…
        </iframe>
        """,
        unsafe_allow_html=True,
    )

# ---- Footer ----
st.markdown("---")
st.caption("Built with OpenAI embeddings and Streamlit")
