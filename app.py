import streamlit as st
import pandas as pd
import numpy as np
import joblib
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# ===== Spotify Setup =====
client_id = "48c27ca09dd848028b2014b25e883bb9"  # Replace with your actual Spotify client ID
client_secret = "42add4e9460d43c883a7da850130cb41"  # Replace with your actual Spotify client secret

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=client_id,
    client_secret=client_secret
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
df = pd.read_csv("hybrid_knn_dataset.csv")
scaler = joblib.load("scaler.joblib")

# Features used in the model
features = [
    "danceability", "energy", "valence", "tempo",
    "acousticness", "instrumentalness", "speechiness", "liveness"
]

# ===== Streamlit UI =====
st.set_page_config(page_title="Music Recommender", layout="wide")
st.title("🎶 GeetYatra – Your Personalized Music Journey")

st.write("Select a song and get similar recommendations with album art!")

song_list = df['track'].unique()
selected_song = st.selectbox("🔍 Select a song", sorted(song_list))

if st.button("Recommend Similar Songs"):
    input_song = df[df['track'] == selected_song].iloc[0]
    input_cluster = input_song['cluster']

    # Load KNN model for that cluster
    knn_model = joblib.load(f"models/knn_cluster_{int(input_cluster)}.joblib")
    cluster_df = df[df['cluster'] == input_cluster].reset_index(drop=True)

    # Get features for the entire cluster
    scaled_features = scaler.transform(cluster_df[features])

    # Get index of selected song within the cluster
    index_in_cluster = cluster_df[cluster_df['track'] == selected_song].index[0]

    # Find similar songs using KNN
    distances, indices = knn_model.kneighbors([scaled_features[index_in_cluster]])

    st.subheader("🎧 Recommended Songs (Side by Side):")

    # Create Streamlit columns for side-by-side layout
    cols = st.columns(len(indices[0][1:]))

    for i, idx in enumerate(indices[0][1:]):  # Skip the first (input song itself)
        result = cluster_df.iloc[idx]
        track = result['track']
        artist = result['artist']
        language = result['language']

        image_url = fetch_album_image(track, artist)

        with cols[i]:
            if image_url:
                st.image(image_url, width=140)
            else:
                st.write("🎵 No Image")

            st.markdown(f"**{track}**  \n*{artist}*  \n🌐 {language}")
