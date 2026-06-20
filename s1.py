import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="The Picture House — Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Data + model (cached so this doesn't re-run on every click/selection)
# ----------------------------------------------------------------------------
@st.cache_data
def load_data_and_similarity():
    data = pd.read_csv("final_data.csv")
    cv = CountVectorizer(max_features=2000, stop_words="english")
    vectors = cv.fit_transform(data["tags_x"]).toarray()
    sim = cosine_similarity(vectors)
    return data, sim


data1, similarity = load_data_and_similarity()

PLACEHOLDER_POSTER = "https://placehold.co/300x445/151922/E8B84B?text=No+Poster&font=montserrat"


def recommend(movie):
    index = data1[data1["original_title_x"] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    names, posters, links = [], [], []
    for i in distances[1:6]:
        row = data1.iloc[i[0]]
        names.append(row.original_title_x)
        poster = row.image_url_y
        posters.append(poster if isinstance(poster, str) and poster.strip() else PLACEHOLDER_POSTER)
        links.append(row.wiki_link_x if isinstance(row.wiki_link_x, str) else "")
    return names, posters, links


# ----------------------------------------------------------------------------
# Theme — cinema marquee: charcoal-navy ground, marquee gold + velvet red accents
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg: #0B0E14;
        --panel: #151922;
        --gold: #E8B84B;
        --velvet: #C73E3E;
        --text: #F2EFE9;
        --muted: #8B92A6;
    }

    .stApp { background: var(--bg); }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--text); }

    #MainMenu, footer, header { visibility: hidden; }

    /* ---- Marquee header ---- */
    .marquee-wrap {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
        border-bottom: 1px solid rgba(232, 184, 75, 0.25);
        margin-bottom: 1.75rem;
    }
    .marquee-eyebrow {
        font-size: 0.72rem;
        letter-spacing: 0.35em;
        text-transform: uppercase;
        color: var(--velvet);
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .marquee-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 4rem;
        line-height: 1;
        letter-spacing: 0.08em;
        color: var(--gold);
        text-shadow: 0 0 18px rgba(232, 184, 75, 0.35), 0 0 2px rgba(232, 184, 75, 0.6);
        margin: 0;
    }
    .marquee-sub {
        color: var(--muted);
        font-size: 0.95rem;
        margin-top: 0.5rem;
        font-style: italic;
    }

    /* ---- Picker ---- */
    .picker-label {
        font-size: 0.72rem;
        letter-spacing: 0.25em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.35rem;
    }
    div[data-baseweb="select"] > div {
        background-color: var(--panel) !important;
        border: 1px solid rgba(232, 184, 75, 0.3) !important;
        border-radius: 4px !important;
    }

    /* ---- Ticket-stub button ---- */
    .stButton > button {
        background: linear-gradient(180deg, #F2CE73, var(--gold));
        color: #0B0E14;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        font-size: 0.8rem;
        border: none;
        border-radius: 4px;
        padding: 0.6rem 1.6rem;
        box-shadow: 0 4px 14px rgba(232, 184, 75, 0.25);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(232, 184, 75, 0.4);
        color: #0B0E14;
    }

    /* ---- Now showing strip ---- */
    .now-showing {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 2rem 0 1.25rem 0;
    }
    .now-showing span {
        font-size: 0.72rem;
        letter-spacing: 0.3em;
        text-transform: uppercase;
        color: var(--gold);
        white-space: nowrap;
        font-weight: 600;
    }
    .now-showing hr {
        flex: 1;
        border: none;
        border-top: 1px solid rgba(232, 184, 75, 0.25);
        margin: 0;
    }

    /* ---- Movie grid ---- */
    .movie-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 1.1rem;
    }
    @media (max-width: 900px) {
        .movie-grid { grid-template-columns: repeat(2, 1fr); }
    }
    .movie-card {
        background: var(--panel);
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 6px 16px rgba(0,0,0,0.4);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        opacity: 0;
        animation: riseIn 0.5s ease forwards;
    }
    .movie-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 14px 30px rgba(232, 184, 75, 0.25);
    }
    .movie-card img {
        width: 100%;
        aspect-ratio: 2 / 3;
        object-fit: cover;
        display: block;
    }
    .perforation {
        display: flex;
        justify-content: space-evenly;
        background: #0B0E14;
        padding: 4px 0;
    }
    .perforation span {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--bg);
        box-shadow: 0 0 0 2px rgba(232, 184, 75, 0.18) inset;
    }
    .movie-title {
        font-size: 0.82rem;
        font-weight: 600;
        text-align: center;
        padding: 0.6rem 0.5rem 0.8rem 0.5rem;
        color: var(--text);
        line-height: 1.25;
    }

    @keyframes riseIn {
        from { opacity: 0; transform: translateY(14px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="marquee-wrap">
        <div class="marquee-eyebrow">The Picture House Presents</div>
        <h1 class="marquee-title">MOVIE RECOMMENDER</h1>
        <div class="marquee-sub">Pick a title — we'll pull five reels you'll want to watch next.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎟️ About this reel")
    st.write(
        "Recommendations are matched by content — genre, cast, and theme tags — "
        "using cosine similarity over the catalog below."
    )
    st.metric("Titles in catalog", f"{len(data1):,}")

# ----------------------------------------------------------------------------
# Picker
# ----------------------------------------------------------------------------
st.markdown('<div class="picker-label">Search the catalog</div>', unsafe_allow_html=True)
movie_list = sorted(data1["original_title_x"].values)
selected_movie = st.selectbox(" ", movie_list, label_visibility="collapsed")

show = st.button("Show Recommendations")

# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
if show:
    names, posters, links = recommend(selected_movie)

    st.markdown(
        '<div class="now-showing"><span>Now Showing</span><hr></div>',
        unsafe_allow_html=True,
    )

    cards_html = '<div class="movie-grid">'
    for i, (name, poster, link) in enumerate(zip(names, posters, links)):
        href_open = f'<a href="{link}" target="_blank" style="text-decoration:none;color:inherit;">' if link else ""
        href_close = "</a>" if link else ""
        delay = i * 0.08
        cards_html += f"""
        <div class="movie-card" style="animation-delay:{delay}s;">
            {href_open}
            <img src="{poster}" alt="{name}" />
            <div class="perforation">
                <span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span>
            </div>
            <div class="movie-title">{name}</div>
            {href_close}
        </div>
        """
    cards_html += "</div>"

    st.markdown(cards_html, unsafe_allow_html=True)
