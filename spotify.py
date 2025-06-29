import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
try:
    df = pd.read_csv("SpotifyFeatures.csv")
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print("❌ ERROR: SpotifyFeatures.csv not found.")
    exit()

# Sample data for performance
df = df.sample(n=2000, random_state=42)

# Select audio features
features = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

if not all(feature in df.columns for feature in features):
    print("❌ ERROR: Some required audio features are missing.")
    exit()

X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans Clustering
print("\n🔍 Finding the best K for KMeans...")
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    print(f"Testing K = {k}")
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    score = silhouette_score(X_scaled, model.labels_)
    silhouette_scores.append(score)

best_k = K_range[silhouette_scores.index(max(silhouette_scores))]
print(f"\n✅ Best K found: {best_k}")

kmeans = KMeans(n_clusters=best_k, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

# Plot silhouette scores
plt.figure(figsize=(8, 4))
sns.lineplot(x=list(K_range), y=silhouette_scores, marker='o')
plt.title("K vs Silhouette Score")
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.tight_layout()
plt.savefig("k_vs_silhouette.png")

# DBSCAN Clustering
print("\n📦 Running DBSCAN...")
dbscan = DBSCAN(eps=1.5, min_samples=10)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

# Evaluation
print("\n📊 Evaluation Metrics:")
print(f"KMeans Silhouette Score     : {silhouette_score(X_scaled, df['KMeans_Cluster']):.4f}")
print(f"KMeans Davies-Bouldin Score : {davies_bouldin_score(X_scaled, df['KMeans_Cluster']):.4f}")
print(f"DBSCAN Davies-Bouldin Score : {davies_bouldin_score(X_scaled, df['DBSCAN_Cluster']):.4f}")

# Search Function for songs
def search_song(keyword):
    result = df[df['track_name'].str.contains(keyword, case=False, na=False)]
    return result[['track_name', 'artist_name']].drop_duplicates().reset_index(drop=True)

# Recommendation Function
def recommend_song(song_name, cluster_type='KMeans_Cluster', top_n=5):
    # Match song name case-insensitively
    matches = df[df['track_name'].str.lower() == song_name.lower()]

    if matches.empty:
        suggestions = df[df['track_name'].str.contains(song_name[:4], case=False, na=False)]
        if not suggestions.empty:
            print("\n🤔 Did you mean one of these?")
            print(suggestions['track_name'].unique()[:5])
        return f"❌ Song '{song_name}' not found in dataset."

    song_row = matches.iloc[0]
    cluster_id = song_row[cluster_type]

    candidates = df[
        (df[cluster_type] == cluster_id) &
        (df['track_name'].str.lower() != song_name.lower())
    ][['track_name', 'artist_name']].drop_duplicates()

    if candidates.empty:
        return "⚠ No similar songs found in this cluster."

    return candidates.sample(n=min(top_n, len(candidates))).reset_index(drop=True)

# Show matching songs
print("\n🎯 Search Results for 'shape':")
print(search_song("shape"))

# User Input
user_song = input("\n🎵 Enter a song name to get similar recommendations: ")
print("\n🎧 Recommended Songs:")
print(recommend_song(user_song))
