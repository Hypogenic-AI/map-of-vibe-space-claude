"""
Step 4: Statistical significance testing for vibe space vs raw text space.

Performs bootstrap confidence intervals, permutation tests, and effect size calculations.
"""

import json
import numpy as np
import random
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

EMBEDDINGS_PATH = "results/data/embeddings.npz"
METADATA_PATH = "results/data/metadata.json"


def bootstrap_silhouette(embeddings, labels, n_bootstrap=1000, sample_frac=0.8):
    """Bootstrap confidence interval for silhouette score."""
    n = len(embeddings)
    sample_size = int(n * sample_frac)
    scores = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, sample_size, replace=True)
        try:
            s = silhouette_score(embeddings[idx], labels[idx], metric="cosine")
            scores.append(s)
        except:
            pass
    return np.mean(scores), np.percentile(scores, 2.5), np.percentile(scores, 97.5)


def permutation_test_ari(labels_a, labels_b, n_permutations=5000):
    """Permutation test: is the ARI between two clusterings significantly different from chance?"""
    observed_ari = adjusted_rand_score(labels_a, labels_b)
    count = 0
    for _ in range(n_permutations):
        perm_b = np.random.permutation(labels_b)
        perm_ari = adjusted_rand_score(labels_a, perm_b)
        if perm_ari >= observed_ari:
            count += 1
    p_value = (count + 1) / (n_permutations + 1)
    return observed_ari, p_value


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0


def main():
    print("=" * 60)
    print("Step 4: Statistical Significance Testing")
    print("=" * 60)

    emb = np.load(EMBEDDINGS_PATH)
    raw_emb = emb["raw_embeddings"]
    vibe_emb = emb["vibe_embeddings"]

    results = {}

    # Cluster both spaces
    k = 10
    km_raw = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels_raw = km_raw.fit_predict(raw_emb)
    km_vibe = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels_vibe = km_vibe.fit_predict(vibe_emb)

    # 1. Bootstrap CI for silhouette scores
    print("\n--- Bootstrap Silhouette Scores (1000 resamples) ---")
    mean_raw, ci_low_raw, ci_high_raw = bootstrap_silhouette(raw_emb, labels_raw)
    mean_vibe, ci_low_vibe, ci_high_vibe = bootstrap_silhouette(vibe_emb, labels_vibe)
    print(f"Raw:  {mean_raw:.4f} [{ci_low_raw:.4f}, {ci_high_raw:.4f}]")
    print(f"Vibe: {mean_vibe:.4f} [{ci_low_vibe:.4f}, {ci_high_vibe:.4f}]")

    results["silhouette_bootstrap"] = {
        "raw": {"mean": mean_raw, "ci_low": ci_low_raw, "ci_high": ci_high_raw},
        "vibe": {"mean": mean_vibe, "ci_low": ci_low_vibe, "ci_high": ci_high_vibe},
    }

    # 2. Permutation test for ARI
    print("\n--- Permutation Test for ARI (5000 permutations) ---")
    observed_ari, p_value = permutation_test_ari(labels_raw, labels_vibe)
    print(f"Observed ARI: {observed_ari:.4f}")
    print(f"p-value (vs chance): {p_value:.4f}")
    significance = "significant" if p_value < 0.05 else "not significant"
    print(f"Result: {significance} at α=0.05")

    results["ari_permutation"] = {
        "observed_ari": observed_ari,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }

    # 3. Neighbor overlap analysis with statistical test
    print("\n--- Neighbor Overlap Statistical Test ---")
    K = 5
    raw_sim = cosine_similarity(raw_emb)
    vibe_sim = cosine_similarity(vibe_emb)
    np.fill_diagonal(raw_sim, -1)
    np.fill_diagonal(vibe_sim, -1)
    n = len(raw_emb)

    raw_neighbors = np.argsort(-raw_sim, axis=1)[:, :K]
    vibe_neighbors = np.argsort(-vibe_sim, axis=1)[:, :K]

    overlaps = []
    for i in range(n):
        overlap = len(set(raw_neighbors[i]) & set(vibe_neighbors[i]))
        overlaps.append(overlap)

    # Expected overlap under random assignment
    expected_random = K * K / n
    t_stat, t_pval = stats.ttest_1samp(overlaps, expected_random)
    print(f"Mean overlap: {np.mean(overlaps):.3f} (out of {K})")
    print(f"Expected random overlap: {expected_random:.3f}")
    print(f"t-statistic: {t_stat:.3f}, p-value: {t_pval:.6f}")

    results["neighbor_overlap"] = {
        "mean_overlap": float(np.mean(overlaps)),
        "std_overlap": float(np.std(overlaps)),
        "expected_random": float(expected_random),
        "t_stat": float(t_stat),
        "p_value": float(t_pval),
    }

    # 4. Content similarity comparison (vibe neighbors vs raw neighbors)
    print("\n--- Content Diversity of Vibe vs Raw Neighbors ---")
    raw_neighbor_content_sims = []
    vibe_neighbor_content_sims = []
    for i in range(n):
        for j in vibe_neighbors[i]:
            raw_neighbor_content_sims.append(cosine_similarity(raw_emb[i:i+1], raw_emb[j:j+1])[0][0])
        for j in raw_neighbors[i]:
            vibe_neighbor_content_sims.append(cosine_similarity(raw_emb[i:i+1], raw_emb[j:j+1])[0][0])

    # Note: reversed naming - raw_neighbor_content_sims is the similarity of VIBE neighbors in raw space
    # vibe_neighbor_content_sims is the similarity of RAW neighbors in raw space
    t_stat2, t_pval2 = stats.mannwhitneyu(
        vibe_neighbor_content_sims, raw_neighbor_content_sims, alternative="greater"
    )
    d = cohens_d(np.array(vibe_neighbor_content_sims), np.array(raw_neighbor_content_sims))

    print(f"Raw-space neighbors (content sim):  {np.mean(vibe_neighbor_content_sims):.4f} ± {np.std(vibe_neighbor_content_sims):.4f}")
    print(f"Vibe-space neighbors (content sim): {np.mean(raw_neighbor_content_sims):.4f} ± {np.std(raw_neighbor_content_sims):.4f}")
    print(f"Mann-Whitney U p-value: {t_pval2:.6f}")
    print(f"Cohen's d: {d:.4f}")

    results["content_diversity"] = {
        "raw_neighbors_content_sim": float(np.mean(vibe_neighbor_content_sims)),
        "vibe_neighbors_content_sim": float(np.mean(raw_neighbor_content_sims)),
        "mann_whitney_p": float(t_pval2),
        "cohens_d": float(d),
    }

    # 5. Vibe embedding space dimensionality analysis
    print("\n--- Embedding Space Effective Dimensionality ---")
    from sklearn.decomposition import PCA
    pca_raw = PCA().fit(raw_emb)
    pca_vibe = PCA().fit(vibe_emb)

    # Number of components for 90% variance
    raw_90 = np.argmax(np.cumsum(pca_raw.explained_variance_ratio_) >= 0.90) + 1
    vibe_90 = np.argmax(np.cumsum(pca_vibe.explained_variance_ratio_) >= 0.90) + 1
    print(f"Components for 90% variance - Raw: {raw_90}, Vibe: {vibe_90}")

    results["dimensionality"] = {
        "raw_90pct_components": int(raw_90),
        "vibe_90pct_components": int(vibe_90),
    }

    # Save
    with open("results/statistical_tests.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved results to results/statistical_tests.json")

    return results


if __name__ == "__main__":
    main()
