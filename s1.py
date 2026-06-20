import pickle
import streamlit as st
import requests
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2000,stop_words='english')

data1=pd.read_csv('final_data.csv')


vector =cv.fit_transform(data1['tags_x']).toarray()

from sklearn.metrics.pairwise import cosine_similarity

similarity =cosine_similarity(vector)
# def fetch_poster(movie)

def recommend(movie):
    index = data1[data1['original_title_x'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    r_movie_name=[]
    r_movie_posters=[]
    r_movie_url=[]
    for i in distances[1:6]:
        movie_id=data1.iloc[i[0]].imdb_id
        img_url=data1.iloc[i[0]].image_url_y

        r_movie_name.append(data1.iloc[i[0]].original_title_x)
        r_movie_posters.append(data1.iloc[i[0]].image_url_y)
        r_movie_url.append(data1.iloc[i[0]].wiki_link_x)

    return  r_movie_name,r_movie_posters,r_movie_url


    for i in distances[1:6]:
        print(data1.iloc[i[0]].original_title_x)


# print(recommend('Housefull 2'))
st.header("MOVIE RECOMMENDER SYSTEAM")

movie_list=sorted(data1['original_title_x'].values)


selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names,recommended_movie_posters,recommended_movie_link = recommend(selected_movie)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_movie_names[0])

        st.image(recommended_movie_posters[0])
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])

    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])

