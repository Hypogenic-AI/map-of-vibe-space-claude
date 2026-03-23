"""
Step 3: Analyze vibe space vs raw text space, compute metrics, generate visualizations.

Tests:
- H1: Vibe diversity (unique descriptions)
- H2: Vibe ≠ Topic (ARI between clusterings)
- H3: Cluster quality (silhouette scores)
- H4: Cross-domain discovery (nearest-neighbor analysis)
"""

import json
import os
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import textwrap

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

EMBEDDINGS_PATH = "results/data/embeddings.npz"
METADATA_PATH = "results/data/metadata.json"
FIGURES_DIR = "figures"
RESULTS_DIR = "results"

os.makedirs(FIGURES_DIR, exist_ok=True)


def load_data():
    emb = np.load(EMBEDDINGS_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    return emb, metadata


def analyze_vibe_diversity(metadata):
    """H1: Are vibe descriptions diverse and meaningful?"""
    print("\n" + "=" * 60)
    print("H1: Vibe Description Diversity Analysis")
    print("=" * 60)

    vibes = [m["vibe_description"] for m in metadata]
    n = len(vibes)

    # Unique descriptions
    unique = len(set(vibes))
    print(f"Total descriptions: {n}")
    print(f"Unique descriptions: {unique} ({100*unique/n:.1f}%)")

    # Average length
    lengths = [len(v) for v in vibes]
    print(f"Avg description length: {np.mean(lengths):.0f} chars (std: {np.std(lengths):.0f})")

    # Vocabulary richness
    all_words = " ".join(vibes).lower().split()
    vocab = set(all_words)
    print(f"Total words: {len(all_words)}, Unique words: {len(vocab)}")
    print(f"Type-token ratio: {len(vocab)/len(all_words):.3f}")

    # Most common opening phrases (after "it's giving")
    openings = []
    for v in vibes:
        lower = v.lower()
        if "it's giving" in lower:
            after = lower.split("it's giving")[1].strip()
            # Get first few words
            words = after.split()[:3]
            if words:
                openings.append(" ".join(words))
    opening_counts = Counter(openings).most_common(15)
    print(f"\nTop 15 vibe openings:")
    for phrase, count in opening_counts:
        print(f"  '{phrase}': {count}")

    # Check "it's giving" compliance
    compliance = sum(1 for v in vibes if v.lower().startswith("it's giving"))
    print(f"\n'It's giving' compliance: {compliance}/{n} ({100*compliance/n:.1f}%)")

    results = {
        "total": n,
        "unique": unique,
        "unique_pct": 100*unique/n,
        "avg_length": float(np.mean(lengths)),
        "type_token_ratio": len(vocab)/len(all_words),
        "its_giving_compliance": compliance,
        "top_openings": opening_counts,
    }
    return results


def cluster_and_compare(emb):
    """H2 & H3: Compare clustering in vibe space vs raw text space."""
    print("\n" + "=" * 60)
    print("H2/H3: Clustering Comparison (Vibe vs Raw)")
    print("=" * 60)

    raw_emb = emb["raw_embeddings"]
    vibe_emb = emb["vibe_embeddings"]

    results = {}

    for k in [5, 10, 15, 20]:
        # Cluster raw embeddings
        km_raw = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels_raw = km_raw.fit_predict(raw_emb)
        sil_raw = silhouette_score(raw_emb, labels_raw, metric="cosine", sample_size=min(500, len(labels_raw)))

        # Cluster vibe embeddings
        km_vibe = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels_vibe = km_vibe.fit_predict(vibe_emb)
        sil_vibe = silhouette_score(vibe_emb, labels_vibe, metric="cosine", sample_size=min(500, len(labels_vibe)))

        # ARI between the two clusterings
        ari = adjusted_rand_score(labels_raw, labels_vibe)

        print(f"k={k:2d}: Silhouette(raw)={sil_raw:.3f}, Silhouette(vibe)={sil_vibe:.3f}, ARI={ari:.3f}")
        results[k] = {"sil_raw": sil_raw, "sil_vibe": sil_vibe, "ari": ari}

    return results


def nearest_neighbor_analysis(emb, metadata):
    """H4: Do vibe-space neighbors come from different content domains?"""
    print("\n" + "=" * 60)
    print("H4: Cross-Domain Nearest Neighbor Analysis")
    print("=" * 60)

    raw_emb = emb["raw_embeddings"]
    vibe_emb = emb["vibe_embeddings"]
    K = 5  # number of neighbors

    # Cosine similarity matrices
    raw_sim = cosine_similarity(raw_emb)
    vibe_sim = cosine_similarity(vibe_emb)

    # For each page, find K nearest neighbors in both spaces
    n = len(raw_emb)
    np.fill_diagonal(raw_sim, -1)
    np.fill_diagonal(vibe_sim, -1)

    raw_neighbors = np.argsort(-raw_sim, axis=1)[:, :K]
    vibe_neighbors = np.argsort(-vibe_sim, axis=1)[:, :K]

    # Overlap between raw and vibe neighbors
    overlaps = []
    for i in range(n):
        overlap = len(set(raw_neighbors[i]) & set(vibe_neighbors[i]))
        overlaps.append(overlap / K)

    avg_overlap = np.mean(overlaps)
    print(f"Average neighbor overlap (raw vs vibe, K={K}): {avg_overlap:.3f}")
    print(f"  (0 = completely different neighbors, 1 = identical neighbors)")

    # Content diversity of vibe neighbors vs raw neighbors
    # Use text similarity as proxy for "same domain"
    raw_neighbor_sims = []
    vibe_neighbor_sims = []
    for i in range(n):
        # Raw text similarity between page i and its vibe neighbors
        for j in vibe_neighbors[i]:
            # Reset diagonal
            raw_sim_val = cosine_similarity(raw_emb[i:i+1], raw_emb[j:j+1])[0][0]
            vibe_neighbor_sims.append(raw_sim_val)
        for j in raw_neighbors[i]:
            raw_sim_val = cosine_similarity(raw_emb[i:i+1], raw_emb[j:j+1])[0][0]
            raw_neighbor_sims.append(raw_sim_val)

    print(f"\nContent similarity of neighbors (measured by raw text embedding cosine sim):")
    print(f"  Raw-space neighbors: {np.mean(raw_neighbor_sims):.3f} (±{np.std(raw_neighbor_sims):.3f})")
    print(f"  Vibe-space neighbors: {np.mean(vibe_neighbor_sims):.3f} (±{np.std(vibe_neighbor_sims):.3f})")
    print(f"  → Lower content similarity for vibe neighbors = cross-domain discovery")

    results = {
        "avg_neighbor_overlap": float(avg_overlap),
        "raw_neighbor_content_sim": float(np.mean(raw_neighbor_sims)),
        "vibe_neighbor_content_sim": float(np.mean(vibe_neighbor_sims)),
        "K": K,
    }
    return results


def plot_2d_maps(emb, metadata):
    """Create side-by-side 2D scatter plots of raw vs vibe space."""
    raw_2d = emb["raw_2d"]
    vibe_2d = emb["vibe_2d"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Cluster labels for coloring (use k=10)
    from sklearn.cluster import KMeans
    km_raw = KMeans(n_clusters=10, random_state=SEED, n_init=10)
    labels_raw = km_raw.fit_predict(emb["raw_embeddings"])
    km_vibe = KMeans(n_clusters=10, random_state=SEED, n_init=10)
    labels_vibe = km_vibe.fit_predict(emb["vibe_embeddings"])

    cmap = plt.cm.get_cmap("tab10")

    # Raw text space
    ax = axes[0]
    scatter = ax.scatter(raw_2d[:, 0], raw_2d[:, 1], c=labels_raw, cmap="tab10",
                        s=15, alpha=0.7, edgecolors="none")
    ax.set_title("Raw Text Embedding Space", fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])

    # Vibe space
    ax = axes[1]
    scatter = ax.scatter(vibe_2d[:, 0], vibe_2d[:, 1], c=labels_vibe, cmap="tab10",
                        s=15, alpha=0.7, edgecolors="none")
    ax.set_title("Vibe Space (\"It's Giving\" Descriptions)", fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/raw_vs_vibe_space.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {FIGURES_DIR}/raw_vs_vibe_space.png")


def plot_vibe_map_annotated(emb, metadata):
    """Create an annotated vibe map with sample labels."""
    vibe_2d = emb["vibe_2d"]

    # Cluster
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=10, random_state=SEED, n_init=10)
    labels = km.fit_predict(emb["vibe_embeddings"])

    fig, ax = plt.subplots(figsize=(14, 12))
    scatter = ax.scatter(vibe_2d[:, 0], vibe_2d[:, 1], c=labels, cmap="tab10",
                        s=20, alpha=0.6, edgecolors="none")

    # Annotate cluster centers with representative vibes
    for cluster_id in range(10):
        mask = labels == cluster_id
        if not np.any(mask):
            continue
        center = vibe_2d[mask].mean(axis=0)
        # Find the point closest to center
        cluster_indices = np.where(mask)[0]
        dists = np.linalg.norm(vibe_2d[cluster_indices] - center, axis=1)
        representative = cluster_indices[np.argmin(dists)]
        vibe_text = metadata[representative]["vibe_description"]

        # Truncate for display
        if len(vibe_text) > 80:
            vibe_text = vibe_text[:77] + "..."

        ax.annotate(vibe_text, center, fontsize=6.5,
                   ha="center", va="center",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="gray"),
                   wrap=True)

    ax.set_title("A Map of Vibe Space", fontsize=18, fontweight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/vibe_map_annotated.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES_DIR}/vibe_map_annotated.png")


def plot_cluster_comparison(cluster_results):
    """Plot silhouette scores and ARI across k values."""
    ks = sorted(cluster_results.keys())
    sil_raw = [cluster_results[k]["sil_raw"] for k in ks]
    sil_vibe = [cluster_results[k]["sil_vibe"] for k in ks]
    aris = [cluster_results[k]["ari"] for k in ks]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Silhouette comparison
    ax = axes[0]
    x = np.arange(len(ks))
    width = 0.35
    ax.bar(x - width/2, sil_raw, width, label="Raw Text", color="#2196F3", alpha=0.8)
    ax.bar(x + width/2, sil_vibe, width, label="Vibe Space", color="#FF5722", alpha=0.8)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Cluster Quality: Raw vs Vibe Space")
    ax.set_xticks(x)
    ax.set_xticklabels(ks)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # ARI
    ax = axes[1]
    ax.bar(x, aris, color="#4CAF50", alpha=0.8)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Adjusted Rand Index")
    ax.set_title("Agreement Between Raw & Vibe Clusterings\n(Lower = More Different)")
    ax.set_xticks(x)
    ax.set_xticklabels(ks)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/cluster_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES_DIR}/cluster_comparison.png")


def plot_neighbor_overlap_distribution(emb):
    """Histogram of neighbor overlap between raw and vibe space."""
    raw_sim = cosine_similarity(emb["raw_embeddings"])
    vibe_sim = cosine_similarity(emb["vibe_embeddings"])
    np.fill_diagonal(raw_sim, -1)
    np.fill_diagonal(vibe_sim, -1)

    K = 5
    n = len(raw_sim)
    raw_neighbors = np.argsort(-raw_sim, axis=1)[:, :K]
    vibe_neighbors = np.argsort(-vibe_sim, axis=1)[:, :K]

    overlaps = []
    for i in range(n):
        overlap = len(set(raw_neighbors[i]) & set(vibe_neighbors[i]))
        overlaps.append(overlap)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(overlaps, bins=range(K+2), align="left", color="#9C27B0", alpha=0.8, edgecolor="white")
    ax.set_xlabel(f"Number of Shared Neighbors (out of {K})")
    ax.set_ylabel("Count")
    ax.set_title("Neighbor Overlap: Raw Text vs Vibe Space\n(Low overlap = vibe captures different information)")
    ax.set_xticks(range(K+1))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/neighbor_overlap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES_DIR}/neighbor_overlap.png")


def show_interesting_vibe_neighbors(emb, metadata):
    """Find and display interesting cross-domain vibe neighbors."""
    print("\n" + "=" * 60)
    print("Interesting Vibe-Space Neighbors (Cross-Domain Discovery)")
    print("=" * 60)

    vibe_sim = cosine_similarity(emb["vibe_embeddings"])
    raw_sim = cosine_similarity(emb["raw_embeddings"])
    np.fill_diagonal(vibe_sim, -1)
    np.fill_diagonal(raw_sim, -1)

    # Find pairs that are close in vibe space but far in raw text space
    n = len(vibe_sim)
    interesting_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            vibe_s = vibe_sim[i, j]
            raw_s = raw_sim[i, j]
            if vibe_s > 0.7 and raw_s < 0.3:  # Close in vibe, far in topic
                interesting_pairs.append((i, j, vibe_s, raw_s))

    interesting_pairs.sort(key=lambda x: x[2] - x[3], reverse=True)
    print(f"Found {len(interesting_pairs)} pairs close in vibe but far in content")

    examples = []
    for i, j, vs, rs in interesting_pairs[:10]:
        example = {
            "page_a_text": metadata[i]["text_preview"][:150],
            "page_b_text": metadata[j]["text_preview"][:150],
            "page_a_vibe": metadata[i]["vibe_description"],
            "page_b_vibe": metadata[j]["vibe_description"],
            "vibe_similarity": float(vs),
            "raw_similarity": float(rs),
        }
        examples.append(example)
        print(f"\n--- Pair (vibe_sim={vs:.3f}, raw_sim={rs:.3f}) ---")
        print(f"  Page A: {metadata[i]['text_preview'][:100]}...")
        print(f"  Vibe A: {metadata[i]['vibe_description']}")
        print(f"  Page B: {metadata[j]['text_preview'][:100]}...")
        print(f"  Vibe B: {metadata[j]['vibe_description']}")

    return examples


def save_interactive_html(emb, metadata):
    """Create an interactive HTML visualization of vibe space."""
    import plotly.graph_objects as go

    vibe_2d = emb["vibe_2d"]
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=10, random_state=SEED, n_init=10)
    labels = km.fit_predict(emb["vibe_embeddings"])

    # Build hover text
    hover_texts = []
    for m in metadata:
        vibe = m["vibe_description"]
        preview = m["text_preview"][:120].replace("\n", " ")
        hover_texts.append(f"<b>Vibe:</b> {vibe}<br><b>Content:</b> {preview}...")

    colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
              '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabebe']

    fig = go.Figure()
    for cluster_id in range(10):
        mask = labels == cluster_id
        fig.add_trace(go.Scatter(
            x=vibe_2d[mask, 0],
            y=vibe_2d[mask, 1],
            mode='markers',
            marker=dict(size=6, color=colors[cluster_id], opacity=0.7),
            text=[hover_texts[i] for i in range(len(mask)) if mask[i]],
            hoverinfo='text',
            name=f'Vibe Cluster {cluster_id}',
        ))

    fig.update_layout(
        title=dict(text="A Map of Vibe Space", font=dict(size=24)),
        xaxis=dict(showticklabels=False, title=""),
        yaxis=dict(showticklabels=False, title=""),
        width=1200,
        height=800,
        hovermode="closest",
        template="plotly_white",
    )

    output_path = f"{FIGURES_DIR}/vibe_map_interactive.html"
    fig.write_html(output_path)
    print(f"Saved interactive map: {output_path}")


def main():
    emb, metadata = load_data()

    # H1: Vibe diversity
    diversity_results = analyze_vibe_diversity(metadata)

    # H2/H3: Clustering comparison
    cluster_results = cluster_and_compare(emb)

    # H4: Nearest neighbor analysis
    nn_results = nearest_neighbor_analysis(emb, metadata)

    # Visualizations
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    plot_2d_maps(emb, metadata)
    plot_vibe_map_annotated(emb, metadata)
    plot_cluster_comparison(cluster_results)
    plot_neighbor_overlap_distribution(emb)

    # Cross-domain examples
    cross_domain_examples = show_interesting_vibe_neighbors(emb, metadata)

    # Interactive HTML
    save_interactive_html(emb, metadata)

    # Save all results
    all_results = {
        "diversity": diversity_results,
        "clustering": {str(k): v for k, v in cluster_results.items()},
        "nearest_neighbors": nn_results,
        "cross_domain_examples": cross_domain_examples[:5],
    }
    with open(f"{RESULTS_DIR}/analysis_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved all results to {RESULTS_DIR}/analysis_results.json")


if __name__ == "__main__":
    main()
