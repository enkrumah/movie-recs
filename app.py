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


# ---- Streamlit page setup ----
st.set_page_config(page_title="AI Movie Recommender", page_icon="🎬", layout="wide")

# ---- Theme Management ----
# Determine if it's daytime (7am-7pm) for auto theme
current_hour = datetime.now().hour
is_daytime = 7 <= current_hour < 19

# Initialize theme in session state (auto-detect based on time)
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
    else:  # Auto
        st.session_state.theme = "light" if is_daytime else "dark"
    
    st.caption(f"Current time: {datetime.now().strftime('%I:%M %p')}")
    if theme_choice == "🔄 Auto":
        st.caption("☀️ Light: 7am-7pm | 🌙 Dark: 7pm-7am")

# Apply theme-specific CSS
is_light = st.session_state.theme == "light"

if is_light:
    # Light mode CSS
    theme_css = """
    <style>
        /* Light mode overrides */
        .stApp {
            background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
        }
        
        /* Accent color - warm coral */
        .stButton > button {
            background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #ea580c 0%, #c2410c 100%);
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(249, 115, 22, 0.3);
        }
        
        /* Slider accent */
        .stSlider > div > div > div > div {
            background-color: #f97316 !important;
        }
        
        /* Info box styling */
        .info-box-light {
            background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%);
            border: 1px solid #fed7aa;
            border-radius: 12px;
            padding: 16px 20px;
            margin: 12px 0;
        }
        .info-box-light .title {
            color: #ea580c;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .info-box-light .content {
            color: #431407;
            font-size: 0.95rem;
            line-height: 1.5;
        }
        
        /* Movie card light */
        .movie-card-light {
            background: white;
            border: 1px solid #e2e8f0;
            padding: 14px 16px;
            border-radius: 14px;
            margin-bottom: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .movie-card-light .title {
            font-weight: 700;
            font-size: 1.05rem;
            margin-bottom: 6px;
            color: #1e293b;
        }
        .movie-card-light .score {
            color: #64748b;
        }
        .movie-card-light .details {
            color: #94a3b8;
            font-size: 0.8rem;
            margin-top: 4px;
        }
        
        /* Summary box light */
        .summary-box-light {
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            border: 1px solid #fcd34d;
            padding: 14px 16px;
            border-radius: 14px;
            margin: 16px 0;
        }
        .summary-box-light .title {
            font-weight: 700;
            margin-bottom: 8px;
            color: #92400e;
        }
        .summary-box-light .content {
            color: #78350f;
        }
    </style>
    """
else:
    # Dark mode CSS
    theme_css = """
    <style>
        /* Dark mode - refined palette */
        .stApp {
            background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
        }
        
        /* Accent color - vibrant coral/pink */
        .stButton > button {
            background: linear-gradient(135deg, #e94560 0%, #c73e54 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #ff6b6b 0%, #e94560 100%);
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(233, 69, 96, 0.4);
        }
        
        /* Slider accent - match coral */
        .stSlider > div > div > div > div {
            background-color: #e94560 !important;
        }
        
        /* Info box styling */
        .info-box-dark {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 1px solid #0f3460;
            border-radius: 12px;
            padding: 16px 20px;
            margin: 12px 0;
        }
        .info-box-dark .title {
            color: #e94560;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .info-box-dark .content {
            color: #eaeaea;
            font-size: 0.95rem;
            line-height: 1.5;
        }
        
        /* Movie card dark */
        .movie-card-dark {
            background: linear-gradient(135deg, #1a1a2e 0%, #0f0f1a 100%);
            border: 1px solid #2a2a4a;
            padding: 14px 16px;
            border-radius: 14px;
            margin-bottom: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        .movie-card-dark .title {
            font-weight: 700;
            font-size: 1.05rem;
            margin-bottom: 6px;
            color: #ffffff;
        }
        .movie-card-dark .score {
            color: #9aa0a6;
        }
        .movie-card-dark .details {
            color: #6b7280;
            font-size: 0.8rem;
            margin-top: 4px;
        }
        
        /* Summary box dark */
        .summary-box-dark {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            border: 1px solid #312e81;
            padding: 14px 16px;
            border-radius: 14px;
            margin: 16px 0;
        }
        .summary-box-dark .title {
            font-weight: 700;
            margin-bottom: 8px;
            color: #a5b4fc;
        }
        .summary-box-dark .content {
            color: #cbd5e1;
        }
    </style>
    """

st.markdown(theme_css, unsafe_allow_html=True)

# ---- Main Content ----
st.title("🎬 AI Movie Recommender")
st.caption("Describe what you want to watch — by mood, theme, or vibe. \
Example: _smart sci-fi, heist, no gore, under 2 hours_")

# How it works box (theme-aware)
box_class = "info-box-light" if is_light else "info-box-dark"
st.markdown(
    f"""
    <div class="{box_class}">
        <div class="title">✨ How it works</div>
        <div class="content">
            This AI-powered tool uses <strong>semantic search</strong> to find movies that match your description — 
            not just keywords, but the actual <em>meaning</em> and <em>vibe</em> you're looking for. 
            It then explains <strong>why</strong> each movie fits your request.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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
        st.session_state.query_text = ex
        st.session_state.auto_search = True
        if hasattr(st, "rerun"):
            st.rerun()


# 3) Now create the text area
if "pending_text" in st.session_state:
    st.session_state.query_text = st.session_state.pop("pending_text")

user_text = st.text_area(
    "Describe what you're in the mood for:",
    height=120,
    placeholder="romantic drama about memory",
    key="query_text",
)


# ---------------------------------------------
# AI-powered prompt suggestions (Responses API)
# ---------------------------------------------
if user_text:
    if "last_prefix" not in st.session_state or st.session_state["last_prefix"] != user_text:
        with st.spinner("Thinking of suggestions..."):
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
                suggestions = [s.strip(" -•") for s in raw_text.split("\n") if s.strip()]
            except Exception as e:
                st.error(f"Error generating suggestions: {e}")
                suggestions = []

        st.session_state["last_suggestions"] = suggestions
        st.session_state["last_prefix"] = user_text
    else:
        suggestions = st.session_state.get("last_suggestions", [])

    if suggestions:
        st.caption("💡 AI suggestions:")
        cols = st.columns(len(suggestions))
        clicked = None
        for i, s in enumerate(suggestions):
            if cols[i].button(s, key=f"suggestion_{i}_{hash(s)}"):
                clicked = s
        if clicked:
            st.session_state["pending_text"] = clicked
            st.session_state["auto_search"] = True
            st.rerun()



# 4) Other controls
col1, col2 = st.columns([2, 1])
with col1:
    k = st.slider("Number of results", min_value=3, max_value=10, value=5, step=1)
with col2:
    smart_mode = st.checkbox(
        "✨ Smart Mode",
        value=False,
        help="Uses AI to re-rank results by relevance, genre fit, and recency. Slower but more accurate.",
    )

# 5) Action: manual click OR auto-search from example

@st.cache_data(show_spinner=False, ttl=600)
def _cached_summary(query: str, movie_texts: tuple[str, ...], per_movie: bool):
    recs_for_llm = [{"text": t} for t in movie_texts]
    return summarize_recommendations(query, recs_for_llm, include_per_movie=per_movie)

do_search = st.button("Recommend") or st.session_state.auto_search
if do_search:
    st.session_state.auto_search = False
    query = st.session_state.query_text.strip()

    if not query:
        st.warning("Please enter a description to get recommendations.")
    else:
        retrieval_time = 0.0
        summary_time = 0.0
        
        if smart_mode:
            with st.spinner("🧠 Smart ranking in progress... (retrieving → ranking → AI scoring)"):
                try:
                    t_start = time.perf_counter()
                    recs = retrieve_rank_llm(
                        query,
                        k_retrieve=20,
                        k_rerank=10,
                        k_final=k,
                    )
                    retrieval_time = time.perf_counter() - t_start
                except Exception as e:
                    st.error(f"⚠️ Error during smart ranking: {e}")
                    recs = []
        else:
            with st.spinner("Finding matching movies..."):
                try:
                    t_start = time.perf_counter()
                    recs = recommend_movies(query, k=k)
                    retrieval_time = time.perf_counter() - t_start
                except Exception as e:
                    st.error(f"⚠️ Error during recommendation: {e}")
                    recs = []

        if not recs:
            st.info("No results found. Try a different description.")
        else:
            # ---- LLM summary ----
            movie_texts = tuple(r["text"] for r in recs)
            with st.spinner("Summarizing the vibe…"):
                t_start = time.perf_counter()
                summary_text, per_lines = _cached_summary(query, movie_texts, per_movie=False)
                summary_time = time.perf_counter() - t_start
            
            # ---- Latency display ----
            total_time = retrieval_time + summary_time
            if smart_mode:
                latency_text = f"⚡ Found {len(recs)} movies in **{total_time:.2f}s** (Retrieve+Rank: {retrieval_time:.2f}s → Summary: {summary_time:.2f}s)"
            else:
                latency_text = f"⚡ Found {len(recs)} movies in **{total_time:.2f}s** (Search: {retrieval_time:.2f}s → Summary: {summary_time:.2f}s)"
            
            st.caption(latency_text)
            
            # Summary box (theme-aware)
            if summary_text:
                summary_class = "summary-box-light" if is_light else "summary-box-dark"
                st.markdown(
                    f"""
                    <div class="{summary_class}">
                        <div class="title">🧠 Why these picks fit your vibe</div>
                        <div class="content">{summary_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.subheader("🎯 Top Picks for You")
            cols = st.columns(2)
            card_class = "movie-card-light" if is_light else "movie-card-dark"
            
            for idx, r in enumerate(recs):
                with cols[idx % 2]:
                    if smart_mode and "rank_score" in r:
                        score_label = "Rank Score"
                        score_value = r["rank_score"]
                        signals = r.get("signals", {})
                        signal_details = (
                            f"Similarity: {signals.get('sim_score', 0):.2f} | "
                            f"Genre: {signals.get('genre_score', 0):.2f} | "
                            f"Recency: {signals.get('recency_score', 0):.2f} | "
                            f"LLM: {signals.get('llm_score', 0):.2f}"
                        )
                    else:
                        score_label = "Similarity"
                        score_value = r.get("similarity", 0)
                        signal_details = ""
                    
                    st.markdown(
                        f"""
                        <div class="{card_class}">
                            <div class="title">{r['text']}</div>
                            <div class="score">{score_label}: {score_value:.3f}</div>
                            {f'<div class="details">{signal_details}</div>' if signal_details else ''}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )



# ---- Feedback Section ----
st.markdown("---")
with st.expander("📝 **Share Your Feedback** — Help improve this tool!", expanded=False):
    st.markdown(
        """
        <iframe 
            src="https://docs.google.com/forms/d/e/1FAIpQLSfguNON5rWmnOfwI4wxX5yzGuXkIB40v84-wycYgk3WIrH8DQ/viewform?embedded=true" 
            width="100%" 
            height="600" 
            frameborder="0" 
            marginheight="0" 
            marginwidth="0"
            style="border-radius: 10px;">
            Loading…
        </iframe>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Your feedback helps make this tool better. Thank you! 🙏")

# ---- Footer ----
st.markdown("---")
if smart_mode:
    st.caption("Built with 🧠 OpenAI embeddings, multi-signal ranking, LLM re-ranking, and Streamlit.")
else:
    st.caption("Built with 🧠 OpenAI embeddings, cosine similarity, and Streamlit.")
