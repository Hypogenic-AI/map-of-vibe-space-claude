"""
Step 2: Embed both raw texts and vibe descriptions, then project to 2D with UMAP.

Compares: raw text embeddings vs vibe description embeddings.
"""

import json
import os
import numpy as np
import random
from sentence_transformers import SentenceTransformer
import umap

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

INPUT_PATH = "results/data/vibe_descriptions.json"
OUTPUT_PATH = "results/data/embeddings.npz"
EMBEDDING_MODEL = "all-mpnet-base-v2"


def main():
    print("=" * 60)
    print("Step 2: Embed texts and vibe descriptions, project with UMAP")
    print("=" * 60)

    # Load vibe descriptions
    with open(INPUT_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} pages with vibe descriptions")

    raw_texts = [d["text"] for d in data]
    vibe_texts = [d["vibe_description"] for d in data]

    # Load embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")

    # Embed raw texts (truncated to first 512 tokens by model)
    print("Embedding raw page texts...")
    raw_embeddings = model.encode(raw_texts, show_progress_bar=True, batch_size=64)
    print(f"  Raw embeddings shape: {raw_embeddings.shape}")

    # Embed vibe descriptions
    print("Embedding vibe descriptions...")
    vibe_embeddings = model.encode(vibe_texts, show_progress_bar=True, batch_size=64)
    print(f"  Vibe embeddings shape: {vibe_embeddings.shape}")

    # UMAP projection to 2D
    print("\nUMAP projection (raw texts)...")
    umap_raw = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=15, min_dist=0.1, metric="cosine")
    raw_2d = umap_raw.fit_transform(raw_embeddings)

    print("UMAP projection (vibe descriptions)...")
    umap_vibe = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=15, min_dist=0.1, metric="cosine")
    vibe_2d = umap_vibe.fit_transform(vibe_embeddings)

    # Save everything
    np.savez(OUTPUT_PATH,
             raw_embeddings=raw_embeddings,
             vibe_embeddings=vibe_embeddings,
             raw_2d=raw_2d,
             vibe_2d=vibe_2d)
    print(f"\nSaved embeddings to {OUTPUT_PATH}")

    # Also save metadata alongside
    meta_path = "results/data/metadata.json"
    metadata = []
    for i, d in enumerate(data):
        metadata.append({
            "page_idx": d["page_idx"],
            "text_preview": d["text"][:200],
            "vibe_description": d["vibe_description"],
            "text_length": d["text_length"],
        })
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
