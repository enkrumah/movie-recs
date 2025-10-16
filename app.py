# -------------------------------
# 🎬 AI Movie Recommender (MVP)
# -------------------------------

import streamlit as st
from src.data_loader import load_movies
from src.recommender import recommend_movies
from src.llm import summarize_recommendations


# ---- Streamlit page setup ----
st.set_page_config(page_title="AI Movie Recommender", page_icon="🎬", layout="wide")

st.title("🎬 AI Movie Recommender (MVP)")
st.caption("Describe what you want to watch — by mood, theme, or vibe. \
Example: _smart sci-fi, heist, no gore, under 2 hours_")

# ---- Load movie data (cached) ----
@st.cache_data(show_spinner=False)
def get_movies_df():
    return load_movies()

movies_df = get_movies_df()
st.caption(f"✅ Loaded **{len(movies_df):,}** movies. Embeddings are ready for search.")

# ---- Input controls ----
st.header("Try it out 🎥")

# 1) Init session state keys before any widgets that use them
if "query_text" not in st.session_state:
    st.session_state.query_text = ""
if "auto_search" not in st.session_state:
    st.session_state.auto_search = False

# 2) Quick examples FIRST (so we can set state before the text_area is created)
st.write("Quick examples:")
cols = st.columns(3)
examples = [
    "mind-bending sci-fi heist, nolan vibe",
    "feel-good sports drama based on a true story",
    "classic crime thriller, 90s tone, cat-and-mouse",
]
for i, ex in enumerate(examples):
    if cols[i].button(ex.title()):
        # Set the value BEFORE the text_area exists in this run
        st.session_state.query_text = ex
        st.session_state.auto_search = True
        if hasattr(st, "rerun"):
            st.rerun()

# 3) Now create the text area bound to the same key
user_text = st.text_area(
    "Describe what you're in the mood for:",
    height=120,
    placeholder="romantic drama about memory",
    key="query_text",  # uses the value we set above (if any)
)

# 4) Other controls
k = st.slider("Number of results", min_value=3, max_value=10, value=5, step=1)

# 5) Action: manual click OR auto-search from example

@st.cache_data(show_spinner=False, ttl=600)
def _cached_summary(query: str, movie_texts: tuple[str, ...], per_movie: bool):
    recs_for_llm = [{"text": t} for t in movie_texts]
    return summarize_recommendations(query, recs_for_llm, include_per_movie=per_movie)

do_search = st.button("Recommend") or st.session_state.auto_search
if do_search:
    # Clear the auto flag so we don’t loop
    st.session_state.auto_search = False
    query = st.session_state.query_text.strip()

    if not query:
        st.warning("Please enter a description to get recommendations.")
    else:
        with st.spinner("Finding matching movies..."):
            try:
                recs = recommend_movies(query, k=k)
            except Exception as e:
                st.error(f"⚠️ Error during recommendation: {e}")
                recs = []

        if not recs:
            st.info("No results found. Try a different description.")
        else:
            # ---- LLM summary (add this) ----
            movie_texts = tuple(r["text"] for r in recs)
            with st.spinner("Summarizing the vibe…"):
                summary_text, per_lines = _cached_summary(query, movie_texts, per_movie=False)
            if summary_text:
                st.markdown(
                    f"""
                    <div style="background:#0f172a;border:1px solid #23324d;
                                padding:14px 16px;border-radius:14px;margin:16px 0;">
                    <div style="font-weight:700;margin-bottom:8px;">🧠 Why these picks fit your vibe</div>
                    <div style="color:#cbd5e1;">{summary_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            # ---- end LLM summary ----

            st.subheader("🎯 Top Picks for You")
            cols = st.columns(2)
            for idx, r in enumerate(recs):
                with cols[idx % 2]:
                    st.markdown(
                        f"""
                        <div style="background:#111;border:1px solid #2a2a2a;
                                    padding:14px 16px;border-radius:14px;margin-bottom:12px;">
                        <div style="font-weight:700;font-size:1.05rem;margin-bottom:6px;">
                            {r['text']}
                        </div>
                        <div style="color:#9aa0a6;">Similarity: {r['similarity']:.3f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )



# ---- Footer ----
st.markdown("---")
st.caption("Built with 🧠 OpenAI embeddings, cosine similarity, and Streamlit.")
