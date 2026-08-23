import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from kneed import KneeLocator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FEATURES = ['bare_proportion', 'determiner_proportion', 'quantifier_proportion',
            'numeral_proportion', 'genitive_proportion']

df = pd.read_csv('/home/claude/extraction/lemma_feature_vectors.csv')
X_raw = df[FEATURES].values

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# --- choose min_samples ---
# rule-of-thumb floor is dimensionality + 1 (Sander et al. 1998); with 5 features
# that floor is 6. We evaluate a small band around it and settle on min_samples = 6.
MIN_SAMPLES = 6

# --- k-distance graph to select eps ---
nbrs = NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(X)
distances, _ = nbrs.kneighbors(X)
k_dist = np.sort(distances[:, -1])  # distance to the k-th neighbor, ascending

kneedle = KneeLocator(range(len(k_dist)), k_dist, curve='convex', direction='increasing')
knee_idx = kneedle.knee
eps = k_dist[knee_idx]
print(f"min_samples = {MIN_SAMPLES}")
print(f"Knee located at index {knee_idx} of {len(k_dist)}, eps = {eps:.4f}")

plt.figure(figsize=(7, 5))
plt.plot(k_dist, linewidth=1.5)
plt.axvline(knee_idx, color='red', linestyle='--', linewidth=1, label=f'selected eps = {eps:.3f}')
plt.axhline(eps, color='red', linestyle='--', linewidth=1)
plt.xlabel(f'Points, sorted by distance to {MIN_SAMPLES}-th nearest neighbor')
plt.ylabel(f'Distance to {MIN_SAMPLES}-th nearest neighbor')
plt.title('k-distance graph for eps selection (standardized features)')
plt.legend()
plt.tight_layout()
plt.savefig('/home/claude/extraction/k_distance_plot.png', dpi=150)
print("saved k_distance_plot.png")

# --- run DBSCAN ---
db = DBSCAN(eps=eps, min_samples=MIN_SAMPLES)
labels = db.fit_predict(X)

df['cluster'] = labels
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = int((labels == -1).sum())
print(f"\nClusters found: {n_clusters}")
print(f"Noise points: {n_noise} of {len(df)} ({n_noise/len(df)*100:.1f}%)")
print()
print("Cluster sizes:")
print(df['cluster'].value_counts().sort_index())

df.to_csv('/home/claude/extraction/lemma_feature_vectors_clustered.csv', index=False)

# per-cluster mean profile
profile = df.groupby('cluster')[FEATURES + ['n_total']].mean().round(3)
profile['n_lemmas'] = df.groupby('cluster').size()
print()
print("Per-cluster mean feature profile:")
print(profile.to_string())
profile.to_csv('/home/claude/extraction/cluster_profiles.csv')
