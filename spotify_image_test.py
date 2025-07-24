import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# === Replace with your actual credentials ===
client_id = "48c27ca09dd848028b2014b25e883bb9"
client_secret = "42add4e9460d43c883a7da850130cb41"

# === Authenticate ===
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=client_id,
    client_secret=client_secret
))

# === Function to fetch album image ===
def fetch_album_image(song_name, artist_name):
    query = f"{song_name} {artist_name}"
    result = sp.search(q=query, limit=1, type='track')

    try:
        image_url = result['tracks']['items'][0]['album']['images'][0]['url']
        return image_url
    except (IndexError, KeyError):
        return None

# === Test song ===
song = "Kesariya"
artist = "Arijit Singh"

image_url = fetch_album_image(song, artist)

if image_url:
    print(f"✅ Image URL: {image_url}")
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    plt.imshow(img)
    plt.axis('off')
    plt.title(f"{song} – {artist}")
    plt.show()
else:
    print("❌ No image found.")
