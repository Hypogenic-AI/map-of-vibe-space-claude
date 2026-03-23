# Resources Catalog

## Summary
This document catalogs all resources gathered for the "A Map of Vibe Space" research project, which investigates whether prompting an LLM with web pages and asking "What is this web page giving? Start your answer with 'it's giving'" can create a navigable "vibe space" via text embeddings.

## Papers
Total papers downloaded: 18

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| VibeCheck | Dunlap et al. | 2024 | papers/2410.12851_vibecheck.pdf | Formalizes "vibes" as measurable LLM output traits |
| Demystifying Embedding Spaces using LLMs | Tennenholtz et al. | 2023 | papers/2310.04475_demystifying_embeddings.pdf | LLMs interpreting embedding spaces |
| Embedding Atlas | Ren et al. (Apple) | 2025 | papers/2505.06386_embedding_atlas.pdf | Interactive embedding visualization tool |
| WizMap | Polo Club (Georgia Tech) | 2023 | papers/2306.09328_wizmap.pdf | Scalable multi-resolution embedding viz |
| UMAP | McInnes et al. | 2018 | papers/1802.03426_umap.pdf | Dimensionality reduction algorithm |
| Embedding Projector | Smilkov et al. (Google) | 2016 | papers/1611.05469_embedding_projector.pdf | Seminal embedding viz tool |
| Sentence-BERT | Reimers, Gurevych | 2019 | papers/1908.10084_sentence_bert.pdf | Standard sentence embedding method |
| Contrastive Text Embeddings | OpenAI | 2022 | papers/2201.10005_contrastive_embeddings.pdf | Contrastive pre-training for embeddings |
| Understanding HTML with LLMs | Gur et al. (Google) | 2022 | papers/2210.03945_html_llm.pdf | LLMs for web page understanding |
| Beyond a Single Extractor | Li et al. | 2026 | papers/2602.19548_html_to_text_extraction.pdf | HTML-to-text extraction strategies |
| Embedding Style Beyond Topics | Various | 2025 | papers/2501.00828_embedding_style.pdf | Style in embedding geometry |
| Visualizing Spatial Semantics | Liu et al. | 2024 | papers/2409.03949_spatial_semantics.pdf | Word clouds on 2D embeddings |
| Multi-Scale Semantic Structure | Haschka et al. | 2025 | papers/2512.23471_multiscale_semantic_structure.pdf | Hierarchical embedding structure |
| SLANG Comprehension | Various | 2024 | papers/2401.12585_slang.pdf | LLM understanding of internet slang |
| Informal Language Processing | Various | 2024 | papers/2404.02323_informal_language.pdf | Slang in LLMs |
| LLMs for Web Scraping | Various | 2024 | papers/2406.08246_llm_web_scraping.pdf | LLM-based web extraction |
| Semantic Navigation | Various | 2026 | papers/2602.05971_semantic_navigation.pdf | Concept navigation as embedding trajectories |
| Pairwise Judgment Embedding | Various | 2024 | papers/2408.04197_pairwise_judgment_embedding.pdf | Semantic embedding for web search |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 3

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Multi-Label Web Categorization | HuggingFace | 49,399 rows | URL + snippet + 11 categories | datasets/multi_label_web_categorization/ | Primary dataset for vibe pipeline |
| C4 Small Sample | HuggingFace | 10,000 rows | Clean web text | datasets/c4_small/ | Pre-extracted text for prototyping |
| URL Classifications (50K) | HuggingFace | 50,000 rows | URL + 223 fine-grained categories | datasets/url_classifications_50k/ | Fine-grained validation set |

See datasets/README.md for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: 2

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| VibeCheck | github.com/lisadunlap/VibeCheck | Vibe discovery framework | code/vibecheck/ | ICLR 2025, conceptual foundation |
| Embedding Atlas | github.com/apple/embedding-atlas | Interactive embedding viz | code/embedding-atlas/ | MIT license, primary viz tool |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
- Used paper-finder service with multiple query variations (vibe embeddings, text embedding visualization, LLM web understanding, semantic navigation)
- Web search for slang/aesthetic research, web content extraction, and interactive visualization tools
- Cross-referenced Papers with Code, GitHub, HuggingFace, and arXiv

### Selection Criteria
- Papers: Prioritized work directly relevant to vibes/aesthetics, embedding visualization, web content understanding, and slang in LLMs
- Datasets: Focused on web page datasets with URLs/content and category labels for ground-truth evaluation
- Code: Selected tools directly needed for the pipeline (content extraction, embedding, visualization)

### Challenges Encountered
- The concept of "vibe space" is genuinely novel — no prior work directly combines web content extraction + LLM vibe description + embedding visualization
- Some HuggingFace datasets use legacy loading scripts incompatible with current `datasets` library
- `wget` not available in environment; used `curl` and Python for downloads

### Gaps and Workarounds
- No pre-existing "vibe-annotated web page" dataset exists — must be generated as part of the experiment
- The "it's giving" prompt pattern is unexplored in literature — this is a research contribution, not a gap to fill

## Recommendations for Experiment Design

1. **Primary dataset(s)**: Multi-Label Web Categorization (49K URLs with 11 categories for ground truth) + C4 Small (10K web texts for no-scraping prototyping)
2. **Baseline methods**:
   - Direct text embedding of raw web content (no LLM vibe step)
   - Topic modeling (LDA) on web content
   - URL category labels as ground-truth clusters
3. **Evaluation metrics**:
   - Silhouette score for cluster quality
   - Adjusted Rand Index vs. ground-truth categories
   - Qualitative assessment of vibe coherence in clusters
   - Nearest-neighbor semantic consistency
4. **Code to adapt/reuse**:
   - `trafilatura` for web content extraction
   - `sentence-transformers` for embedding "it's giving" descriptions
   - `umap-learn` for 2D projection
   - `embedding-atlas` for interactive visualization
5. **Pipeline**:
   ```
   URLs → trafilatura (extract text) → LLM ("it's giving...") → sentence-transformers (embed) → UMAP (2D) → Embedding Atlas (visualize)
   ```
6. **LLM for vibe generation**: Use a capable model (GPT-4, Claude, or open-source like Llama-3) with the prompt: "What is this web page giving? Start your answer with 'it's giving'"
7. **Embedding models to compare**: `all-MiniLM-L6-v2` (fast) vs. `all-mpnet-base-v2` (accurate) vs. `nomic-embed-text-v1.5`
