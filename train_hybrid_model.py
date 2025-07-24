import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import joblib
import os

# === 1. Load your dataset ===
df = pd.read_csv("final_music_dataset.csv")

# === 2. Select features to train on ===
feature_cols = [
    "danceability", "energy", "valence", "tempo",
    "acousticness", "instrumentalness", "speechiness", "liveness"
]

X = df[feature_cols]

# === 3. Standardize the features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 4. Apply KMeans clustering ===
n_clusters = 10  # You can tune this (start with 10)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)

# === 5. Save the global scaler and kmeans ===
joblib.dump(scaler, "scaler.joblib")
joblib.dump(kmeans, "kmeans_model.joblib")

# === 6. Train and save a KNN model per cluster ===
if not os.path.exists("models"):
    os.makedirs("models")

for cluster_num in range(n_clusters):
    cluster_df = df[df["cluster"] == cluster_num]
    X_cluster = scaler.transform(cluster_df[feature_cols])

    knn = NearestNeighbors(n_neighbors=6, metric="cosine")
    knn.fit(X_cluster)

    joblib.dump(knn, f"models/knn_cluster_{cluster_num}.joblib")
    print(f"✅ Saved KNN model for Cluster {cluster_num} (Songs: {len(cluster_df)})")

# === 7. Save the updated dataset with cluster info ===
df.to_csv("hybrid_knn_dataset.csv", index=False)

print("✅ All hybrid models trained and saved!")
