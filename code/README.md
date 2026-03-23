# Cloned Repositories

## Repo 1: VibeCheck
- **URL**: https://github.com/lisadunlap/VibeCheck
- **Purpose**: Reference implementation for discovering and quantifying "vibes" in LLM outputs
- **Location**: code/vibecheck/
- **Key files**:
  - `vibecheck/` — Core library with vibe discovery, validation, iteration
  - `data/friendly_and_cold_sample.csv` — Sample vibe-annotated data
  - `notebooks/` — Example usage notebooks
- **Notes**: Conceptual foundation for our project. Uses GPT-4o for vibe discovery and GPT-4o-mini + Llama-3-70b as judge panel. Published at ICLR 2025.

## Repo 2: Embedding Atlas (Apple)
- **URL**: https://github.com/apple/embedding-atlas
- **Purpose**: Scalable interactive embedding visualization tool
- **Location**: code/embedding-atlas/
- **Key files**:
  - `packages/embedding-atlas/` — Core visualization package
  - `packages/embedding-atlas-widget/` — Jupyter widget
  - `examples/` — Usage examples
- **Notes**: MIT license. Primary visualization tool for our vibe map. Install via `pip install embedding-atlas`. Supports drag-and-drop Parquet, in-browser UMAP, automatic clustering/labeling, density contours. Scales to millions of points via WebGPU.

## Key Dependencies (Install via pip, not cloned)

| Package | Purpose | Install |
|---------|---------|---------|
| trafilatura | Web content extraction | `pip install trafilatura` |
| sentence-transformers | Text embedding | `pip install sentence-transformers` |
| umap-learn | Dimensionality reduction | `pip install umap-learn` |
| embedding-atlas | Interactive visualization | `pip install embedding-atlas` |
