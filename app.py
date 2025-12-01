# -------------------------------
# 🎬 AI Movie Recommender (MVP)
# -------------------------------

import time
from datetime import datetime
import streamlit as st
from src.data_loader import load_movies
from src.recommender import recommend_movies
from src.ranker import retrieve_rank_llm, RankingWeights
from src.llm import summarize_recommendations
from src.llm import _get_client


# ---- Page Config ----
st.set_page_config(page_title="AI Movie Recommender", page_icon="🎬", layout="wide")

# ---- Theme Management ----
current_hour = datetime.now().hour
is_daytime = 7 <= current_hour < 19

if "theme" not in st.session_state:
    st.session_state.theme = "light" if is_daytime else "dark"

# Theme toggle in sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    theme_options = ["🌙 Dark", "☀️ Light", "🔄 Auto"]
    theme_choice = st.radio(
        "Theme",
        theme_options,
        index=0 if st.session_state.theme == "dark" else (1 if st.session_state.theme == "light" else 2),
        horizontal=True,
    )
    
    if theme_choice == "🌙 Dark":
        st.session_state.theme = "dark"
    elif theme_choice == "☀️ Light":
        st.session_state.theme = "light"
    else:
        st.session_state.theme = "light" if is_daytime else "dark"
    
    st.caption(f"🕐 {datetime.now().strftime('%I:%M %p')}")

is_light = st.session_state.theme == "light"

# ---- Premium Design System CSS ----
if is_light:
    # LIGHT MODE - Premium Minimal
    theme_css = """
    <style>
        /* Base */
        .stApp {
            background: linear-gradient(180deg, #FAFAFC 0%, #F1F1F4 100%);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        /* Hide default Streamlit elements for cleaner look */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            font-weight: 600 !important;
            letter-spacing: 0.3px !important;
            color: #1C1C1E !important;
        }
        
        p, span, label, .stMarkdown {
            color: #1C1C1E;
        }
        
        /* Buttons */
        .stButton > button {
            background: #FF6B4A !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 8px 16px !important;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button:hover {
            background: #E55A3A !important;
            box-shadow: 0 4px 12px rgba(255, 107, 74, 0.25) !important;
        }
        .stButton > button:active {
            transform: scale(0.98) !important;
        }
        
        /* Slider */
        .stSlider > div > div > div > div {
            background-color: #FF6B4A !important;
        }
        .stSlider > div > div > div > div > div {
            background-color: #FF6B4A !important;
        }
        
        /* Text input / Text area */
        .stTextArea textarea, .stTextInput input {
            background: #FFFFFF !important;
            border: 1px solid #E5E5EA !important;
            border-radius: 8px !important;
            color: #1C1C1E !important;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        }
        .stTextArea textarea:focus, .stTextInput input:focus {
            border-color: #FF6B4A !important;
            box-shadow: 0 0 0 2px rgba(255, 107, 74, 0.15) !important;
        }
        
        /* Checkbox */
        .stCheckbox label span {
            color: #1C1C1E !important;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background: #FFFFFF !important;
            border: 1px solid #E5E5EA !important;
            border-radius: 8px !important;
            color: #1C1C1E !important;
        }
        
        /* Cards - Light */
        .card-light {
            background: #FFFFFF;
            border: 1px solid #E5E5EA;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 16px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        .card-light .card-title {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-weight: 600;
            font-size: 1rem;
            color: #1C1C1E;
            margin-bottom: 8px;
            line-height: 1.4;
        }
        .card-light .card-score {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 0.875rem;
            color: #4A4A4A;
        }
        .card-light .card-details {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 0.75rem;
            color: #6B6B6B;
            margin-top: 8px;
        }
        
        /* Info Box - Light */
        .info-box-light {
            background: #FFFFFF;
            border: 1px solid #E5E5EA;
            border-left: 4px solid #FF6B4A;
            border-radius: 8px;
            padding: 16px 20px;
            margin: 16px 0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        .info-box-light .info-title {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-weight: 600;
            font-size: 0.875rem;
            color: #FF6B4A;
            margin-bottom: 8px;
            letter-spacing: 0.3px;
        }
        .info-box-light .info-content {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 0.9rem;
            color: #4A4A4A;
            line-height: 1.6;
        }
        
        /* Summary Box - Light */
        .summary-box-light {
            background: #FFFFFF;
            border: 1px solid #E5E5EA;
            border-radius: 12px;
            padding: 16px 20px;
            margin: 16px 0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        .summary-box-light .summary-title {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-weight: 600;
            font-size: 0.9rem;
            color: #1C1C1E;
            margin-bottom: 8px;
        }
        .summary-box-light .summary-content {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 0.9rem;
            color: #4A4A4A;
            line-height: 1.6;
        }
        
        /* Captions */
        .stCaption, small {
            color: #6B6B6B !important;
        }
    </style>
    """
else:
    # DARK MODE - Cinematic Vibes
    theme_css = """
    <style>
        /* Base */
        .stApp {
            background: #0E0F13;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        /* Vignette effect */
        .stApp::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(ellipse at center, transparent 0%, rgba(0,0,0,0.15) 100%);
            pointer-events: none;
            z-index: 0;
        }
        
        /* Hide default Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            font-weight: 600 !important;
            letter-spacing: 0.3px !important;
            color: #F5F5F7 !important;
        }
        
        p, span, label, .stMarkdown {
            color: #F5F5F7;
        }
        
        /* Buttons */
        .stButton > button {
            background: #FF6B4A !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 8px 16px !important;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button:hover {
            background: #E55A3A !important;
            box-shadow: 0 4px 16px rgba(255, 107, 74, 0.35) !important;
        }
        .stButton > button:active {
            transform: scale(0.98) !important;
        }
        
        /* Slider */
        .stSlider > div > div > div > div {
            background-color: #FF6B4A !important;
        }
        .stSlider > div > div > div > div > div {
            background-color: #FF6B4A !important;
        }
        
        /* Text input / Text area */
        .stTextArea textarea, .stTextInput input {
            background: #1A1B20 !important;
            border: 1px solid #2A2B33 !important;
            border-radius: 8px !important;
            color: #F5F5F7 !important;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        }
        .stTextArea textarea:focus, .stTextInput input:focus {
            border-color: #FF6B4A !important;
            box-shadow: 0 0 0 2px rgba(255, 107, 74, 0.2) !important;
        }
        .stTextArea textarea::placeholder {
            color: #6B6B73 !important;
        }
        
        /* Checkbox */
        .stCheckbox label span {
            color: #F5F5F7 !important;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background: #1A1B20 !important;
            border: 1px solid #2A2B33 !important;
            border-radius: 8px !important;
            color: #F5F5F7 !important;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: #0E0F13 !important;
            border-right: 1px solid #2A2B33 !important;
        }
        
        /* Cards - Dark */
        .card-dark {
            background: #1A1B20;
            border: 1px solid #2A2B33;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 16px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
        }
        .card-dark .card-title {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-weight: 600;
            font-size: 1rem;
            color: #F5F5F7;
            margin-bottom: 8px;
            line-height: 1.4;
        }
        .card-dark .card-score {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 0.875rem;
            color: #A9ABB6;
        }
        .card-dark .card-details {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 0.75rem;
            color: #6B6B73;
            margin-top: 8px;
        }
        
        /* Info Box - Dark */
        .info-box-dark {
            background: #1A1B20;
            border: 1px solid #2A2B33;
            border-left: 4px solid #FF6B4A;
            border-radius: 8px;
            padding: 16px 20px;
            margin: 16px 0;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
        }
        .info-box-dark .info-title {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-weight: 600;
            font-size: 0.875rem;
            color: #FF6B4A;
            margin-bottom: 8px;
            letter-spacing: 0.3px;
        }
        .info-box-dark .info-content {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 0.9rem;
            color: #A9ABB6;
            line-height: 1.6;
        }
        
        /* Summary Box - Dark */
        .summary-box-dark {
            background: #1A1B20;
            border: 1px solid #2A2B33;
            border-radius: 12px;
            padding: 16px 20px;
            margin: 16px 0;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
        }
        .summary-box-dark .summary-title {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-weight: 600;
            font-size: 0.9rem;
            color: #F5F5F7;
            margin-bottom: 8px;
        }
        .summary-box-dark .summary-content {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 0.9rem;
            color: #A9ABB6;
            line-height: 1.6;
        }
        
        /* Captions */
        .stCaption, small {
            color: #6B6B73 !important;
        }
        
        /* Radio buttons in sidebar */
        [data-testid="stSidebar"] .stRadio label {
            color: #A9ABB6 !important;
        }
    </style>
    """

st.markdown(theme_css, unsafe_allow_html=True)

# ---- Main Content ----
st.title("🎬 AI Movie Recommender")
st.caption("Describe what you want to watch — by mood, theme, or vibe.")

# How it works box
box_class = "info-box-light" if is_light else "info-box-dark"
st.markdown(
    f"""
    <div class="{box_class}">
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

# AI suggestions
if user_text:
    if "last_prefix" not in st.session_state or st.session_state["last_prefix"] != user_text:
        with st.spinner(""):
            try:
                client = _get_client()
                suggestion_resp = client.responses.create(
                    model="gpt-4o-mini",
                    input=(
                        f"Suggest 3 short, distinct movie-related prompts that could complete "
                        f"or expand the phrase: '{user_text}'. Keep them under 8 words each and movie-focused."
                    ),
                    temperature=0.7,
                )
                raw_text = suggestion_resp.output_text.strip()
                suggestions = [s.strip(" -•1234567890.") for s in raw_text.split("\n") if s.strip()][:3]
            except Exception:
                suggestions = []
        st.session_state["last_suggestions"] = suggestions
        st.session_state["last_prefix"] = user_text
    else:
        suggestions = st.session_state.get("last_suggestions", [])

    if suggestions:
        st.caption("💡 Suggestions:")
        cols = st.columns(len(suggestions))
        for i, s in enumerate(suggestions):
            if cols[i].button(s, key=f"sug_{i}_{hash(s)}"):
                st.session_state["pending_text"] = s
            st.session_state["auto_search"] = True
            st.rerun()

# Controls
col1, col2 = st.columns([2, 1])
with col1:
    k = st.slider("Results", min_value=3, max_value=10, value=5, step=1)
with col2:
    smart_mode = st.checkbox("✨ Smart Mode", value=False, help="AI-powered re-ranking for better results")

# Search
@st.cache_data(show_spinner=False, ttl=600)
def _cached_summary(query: str, movie_texts: tuple[str, ...], per_movie: bool):
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
                summary_class = "summary-box-light" if is_light else "summary-box-dark"
                st.markdown(
                    f"""
                    <div class="{summary_class}">
                        <div class="summary-title">🎯 Why these match your vibe</div>
                        <div class="summary-content">{summary_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Results
            st.markdown("### Top Picks")
            card_class = "card-light" if is_light else "card-dark"
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
                    
                    details_html = f'<div class="card-details">{signal_details}</div>' if signal_details else ''
                    st.markdown(
                        f"""
                        <div class="{card_class}">
                            <div class="card-title">{r['text']}</div>
                            <div class="card-score">{score_label}: {score_value:.2f}</div>
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
