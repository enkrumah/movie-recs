import streamlit as st

from src.data_loader import load_movies
import streamlit as st

@st.cache_data(show_spinner=False)
def get_movies_df():
    return load_movies()

movies_df = get_movies_df()
st.caption(f"Loaded {len(movies_df):,} movies.")


st.set_page_config(page_title="AI Movie Recommender", page_icon="🎬", layout="wide")
st.title("🎬 AI Movie Recommender (MVP)")
st.caption("Setup complete. Next: Data → Embeddings → kNN → LLM.")

st.header("Try it")
user_text = st.text_area(
    "Describe what you're in the mood for:",
    height=120,
    placeholder="e.g., 3 friends, smart sci-fi, under 2 hours, no gore"
)

if st.button("Recommend"):
    st.info("Retrieval pipeline not wired yet — we’ll build it step by step.")
