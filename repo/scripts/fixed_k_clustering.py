"""
fixed_k_clustering.py

Fixed-k agglomerative (Ward) clustering on arcsine-transformed,
standardized lemma feature vectors, as a targeted robustness check
against the original DBSCAN solution (Section 3.5 of the thesis).
Also reproduces the raw-proportion clustering and the k-sweep used to
select k=8 (matching Allan 1980) and k=12 (matching Grimm & Wahlang
2021's coarsest PDE model).

Input:  lemma_feature_vectors.csv (output of build_features.py)
Output: printed cluster profiles, silhouette scores, robustness
        comparison (Adjusted Rand Index between raw and arcsine
        solutions), and saved figures (dendrogram, cluster heatmaps,
        PCA scatterplots) for k=8 and k=12.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_CSV = "lemma_feature_vectors.csv"
FEATURES = [
    "bare_proportion", "determiner_proportion", "quantifier_proportion",
    "numeral_proportion", "genitive_proportion",
]


def arcsine(p):
    return np.arcsin(np.sqrt(p))


def cluster_profile(df, features, label_col):
    profile = df.groupby(label_col)[features].mean().round(3)
    profile["n"] = df.groupby(label_col).size()
    return profile.sort_values("n", ascending=False)


def main():
    df = pd.read_csv(INPUT_CSV)
    X_raw = StandardScaler().fit_transform(df[FEATURES].values)
    X_arc = StandardScaler().fit_transform(arcsine(df[FEATURES].values))

    # --- k sweep to justify k selection (Section 3.5.2) ---
    print("k sweep (arcsine-transformed features):")
    for k in range(2, 25):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_arc)
        sil = silhouette_score(X_arc, labels)
        sizes = pd.Series(labels).value_counts()
        print(f"  k={k:2d}  silhouette={sil:.4f}  min_cluster={sizes.min()}  max_cluster={sizes.max()}")
    print()

    # --- raw vs. arcsine robustness comparison at k=8 and k=12 ---
    for k in [8, 12]:
        labels_raw = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_raw)
        labels_arc = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_arc)
        sil_raw = silhouette_score(X_raw, labels_raw)
        sil_arc = silhouette_score(X_arc, labels_arc)
        ari = adjusted_rand_score(labels_raw, labels_arc)
        print(f"k={k}: silhouette raw={sil_raw:.3f}  arcsine={sil_arc:.3f}  agreement (ARI)={ari:.3f}")

        df[f"cluster_raw_k{k}"] = labels_raw
        df[f"cluster_arc_k{k}"] = labels_arc

        print(f"\n--- k={k}, arcsine-transformed cluster profiles ---")
        print(cluster_profile(df, FEATURES, f"cluster_arc_k{k}").to_string())
        print()

    # --- dendrogram with k=8/k=12 cut lines ---
    Z = linkage(X_arc, method="ward")
    fig, ax = plt.subplots(figsize=(11, 5))
    dendrogram(Z, no_labels=True, color_threshold=0, ax=ax, above_threshold_color="#888888")
    for k, color, style in [(8, "#C44E52", "--"), (12, "#4C72B0", ":")]:
        for h in sorted(Z[:, 2]):
            if len(set(fcluster(Z, h, criterion="distance"))) <= k:
                ax.axhline(h, color=color, linestyle=style, linewidth=1.3, label=f"k={k} cut")
                break
    ax.set_title("Hierarchical clustering dendrogram (Ward linkage, arcsine-transformed features)")
    ax.set_xlabel(f"{len(df)} lemmas")
    ax.set_ylabel("Ward distance")
    ax.legend()
    plt.tight_layout()
    plt.savefig("dendrogram.png", dpi=150)
    plt.close()

    # --- PCA scatterplots + heatmaps for k=8 and k=12 ---
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_arc)
    print("PCA explained variance:", pca.explained_variance_ratio_.round(3))

    for k in [8, 12]:
        labels = df[f"cluster_arc_k{k}"].values
        cmap = plt.get_cmap("tab10" if k <= 10 else "tab20")

        fig, ax = plt.subplots(figsize=(7, 6.2))
        for c in range(k):
            m = labels == c
            ax.scatter(coords[m, 0], coords[m, 1], s=18, alpha=0.8,
                       color=cmap(c % cmap.N), label=f"cluster {c} (n={m.sum()})")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
        ax.set_title(f"Hierarchical clustering (Ward, arcsine-transformed), k={k}")
        ax.legend(fontsize=7, ncol=2, loc="best")
        plt.tight_layout()
        plt.savefig(f"cluster_scatter_k{k}_arcsine.png", dpi=150)
        plt.close()

        profile = cluster_profile(df, FEATURES, f"cluster_arc_k{k}")
        heat = profile[FEATURES].copy()
        heat.index = [f"Cluster {i} (n={int(profile.loc[i, 'n'])})" for i in profile.index]
        heat.columns = [c.replace("_proportion", "") for c in heat.columns]

        fig, ax = plt.subplots(figsize=(7, 0.55 * k + 1.5))
        sns.heatmap(heat, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1,
                    linewidths=0.5, linecolor="white", cbar_kws={"label": "mean proportion"}, ax=ax)
        ax.set_title(f"Mean feature profiles by cluster (k={k})")
        ax.set_ylabel("")
        plt.tight_layout()
        plt.savefig(f"cluster_heatmap_k{k}_arcsine.png", dpi=150)
        plt.close()

    df.to_csv("lemma_feature_vectors_fixed_k_clustered.csv", index=False)
    print("Saved: dendrogram.png, cluster_scatter_k{8,12}_arcsine.png, "
          "cluster_heatmap_k{8,12}_arcsine.png, lemma_feature_vectors_fixed_k_clustered.csv")


if __name__ == "__main__":
    main()
