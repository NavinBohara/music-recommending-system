import streamlit as st
import pandas as pd
import numpy as np
import joblib
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# ===== Spotify Setup from secrets.toml =====
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=st.secrets["SPOTIPY_CLIENT_ID"],
    client_secret=st.secrets["SPOTIPY_CLIENT_SECRET"]
))

@st.cache_data
def fetch_album_image(song_name, artist_name):
    query = f"{song_name} {artist_name}"
    try:
        result = sp.search(q=query, limit=1, type='track')
        return result['tracks']['items'][0]['album']['images'][0]['url']
    except:
        return None

# ===== Load Data and Models =====
df = pd.read_csv("hybrid_knn_dataset.csv")  # Make sure this file exists in your GitHub repo
scaler = joblib.load("scaler.joblib")

# Features used for similarity
features = [
    "danceability", "energy", "valence", "tempo",
    "acousticness", "instrumentalness", "speechiness", "liveness"
]

# ===== Streamlit UI =====
st.set_page_config(page_title="Music Recommender", layout="wide")
st.title("🎶 GeetYatra – Your Personalized Music Journey")

st.markdown("Select a song and get similar music recommendations with album covers. Powered by KNN + Spotify.")

# ==== Song Selection ====
song_list = df['track'].unique()
selected_song = st.selectbox("🔍 Select a song", sorted(song_list))

if st.button("🎵 Recommend Similar Songs"):
    try:
        input_song = df[df['track'] == selected_song].iloc[0]
        input_cluster = input_song['cluster']

        # Load KNN model for that cluster
        knn_model = joblib.load(f"models/knn_cluster_{int(input_cluster)}.joblib")
        cluster_df = df[df['cluster'] == input_cluster].reset_index(drop=True)

        # Scale features
        scaled_features = scaler.transform(cluster_df[features])
        index_in_cluster = cluster_df[cluster_df['track'] == selected_song].index[0]

        # Get recommendations using KNN
        distances, indices = knn_model.kneighbors([scaled_features[index_in_cluster]])

        st.subheader("🎧 Recommended Songs:")

        # Show in Streamlit columns (side-by-side)
        cols = st.columns(len(indices[0][1:]))

        for i, idx in enumerate(indices[0][1:]):  # skip first (itself)
            result = cluster_df.iloc[idx]
            track = result['track']
            artist = result['artist']
            language = result['language']
            image_url = fetch_album_image(track, artist)

            with cols[i]:
                if image_url:
                    st.image(image_url, width=140)
                else:
                    st.write("🎵 No Image Available")

                st.markdown(f"**{track}**  \n*{artist}*  \n🌐 {language}")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
