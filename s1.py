import pickle
import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page layout to wide for a better dashboard feel
st.set_page_config(page_title="Cinematch", layout="wide", page_icon="🎬")

# --- CUSTOM CSS FOR DESIGN ---
st.markdown("""
    <style>
    /* Background and overall app styling */
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #ffffff;
    }
    
    /* Header styling */
    .main-title {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: -webkit-linear-gradient(#ff416c, #ff4b2b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .sub-title {
        text-align: center;
        color: #a0aec0;
        font-size: 1.1rem;
        margin-bottom: 40px;
    }
    
    /* Custom Movie Card styling */
    .movie-card {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(255, 65, 108, 0.4);
        border: 1px solid rgba(255, 65, 108, 0.3);
    }
    
    /* Movie Title inside card */
    .movie-title {
        font-size: 1rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 10px;
        margin-bottom: 12px;
        min-height: 48px; /* Keeps titles aligned */
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    
    /* Watch Button styling */
    .watch-btn {
        display: inline-block;
        padding: 8px 16px;
        background: linear-gradient(90deg, #ff416c, #ff4b2b);
        color: white !important;
        text-decoration: none;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        transition: background 0.3s ease;
    }
    .watch-btn:hover {
        background: linear-gradient(90deg, #ff4b2b, #ff416c);
        box-shadow: 0 4px 12px rgba(255, 75, 43, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# --- DATA PROCESSING ---
@st.cache_data # Added caching so it doesn't reload on every click
def load_data():
    data1 = pd.read_csv('final_data.csv')
    cv = CountVectorizer(max_features=2000, stop_words='english')
    vector = cv.fit_transform(data1['tags_x']).toarray()
    similarity = cosine_similarity(vector)
    return data1, similarity

data1, similarity = load_data()

def recommend(movie):
    index = data1[data1['original_title_x'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    
    r_movie_name = []
    r_movie_posters = []
    r_movie_url = []
    
    for i in distances[1:6]:
        r_movie_name.append(data1.iloc[i[0]].original_title_x)
        # Fallback image if poster URL is missing
        poster = data1.iloc[i[0]].image_url_y
        r_movie_posters.append(poster if pd.notna(poster) else "https://via.placeholder.com/500x750?text=No+Poster+Available")
        r_movie_url.append(data1.iloc[i[0]].wiki_link_x)

    return r_movie_name, r_movie_posters, r_movie_url

# --- UI LAYOUT ---
st.markdown('<p class="main-title">🎬 CINEMATCH</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Your AI-powered personalized movie companion</p>', unsafe_allow_html=True)

movie_list = sorted(data1['original_title_x'].values)

# Centering the selector
col_space1, col_target, col_space2 = st.columns([1, 2, 1])
with col_target:
    selected_movie = st.selectbox(
        "Search or selection a movie from the collection:",
        movie_list
    )
    st.write("") # spacing
    # Center aligning the primary action button
    button_pressed = st.button('✨ Get Recommendations', use_container_width=True)

st.markdown("---")

if button_pressed:
    with st.spinner('Curating the best matches for you...'):
        recommended_movie_names, recommended_movie_posters, recommended_movie_link = recommend(selected_movie)
        
        # Displaying 5 beautiful responsive layout columns
        cols = st.columns(5)
        
        for idx, col in enumerate(cols):
            with col:
                # Combining components into an HTML container card
                st.markdown(f"""
                    <div class="movie-card">
                        <img src="{recommended_movie_posters[idx]}" style="width:100%; border-radius:8px; object-fit: cover; aspect-ratio: 2/3;" />
                        <div class="movie-title">{recommended_movie_names[idx]}</div>
                        <a class="watch-btn" href="{recommended_movie_link[idx]}" target="_blank">🌐 Info / Wiki</a>
                    </div>
                """, unsafe_allow_html=True)
