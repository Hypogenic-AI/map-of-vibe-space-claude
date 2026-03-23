# Literature Review: A Map of Vibe Space

## Research Area Overview

This project sits at the intersection of several active research areas: (1) using LLMs to generate subjective/stylistic descriptions of content, (2) embedding text descriptions in a shared semantic space, and (3) visualizing those embeddings as interactive 2D maps. The core hypothesis is that prompting an LLM with "What is this web page giving? Start your answer with 'it's giving'" produces descriptions that capture the *vibe* or *aesthetic character* of web content, and that embedding these descriptions creates a navigable "vibe space."

## Key Papers

### 1. VibeCheck: Discover & Quantify Qualitative Differences in LLMs
- **Authors**: Dunlap, Mandal, Darrell, Steinhardt, Gonzalez (UC Berkeley)
- **Year**: 2024 (ICLR 2025)
- **arXiv**: 2410.12851
- **Key Contribution**: Formalizes "vibes" as measurable axes (tone, formatting, writing style) along which LLM outputs differ. Introduces a system to automatically discover and quantify vibes.
- **Methodology**: Iterative pipeline: (1) LLM examines output pairs to discover vibe axes (e.g., "formal → friendly"), (2) panel of LLM judges score outputs on each vibe, (3) vibes are filtered for being well-defined (inter-annotator agreement), differentiating (separability score), and user-aligned (preference prediction). Uses Cohen's Kappa, logistic regression.
- **Key Results**: Vibes predict model identity with 80% accuracy and human preference with 61% accuracy on Chatbot Arena. Discovered that Llama-3 is more humorous, uses more formatting, comments less on ethics than GPT-4/Claude.
- **Datasets Used**: Chatbot Arena (Llama-3-70b battles), HC3 (Human ChatGPT Comparison Corpus), CNN/DailyMail, MATH, COCO captions.
- **Code**: https://github.com/lisadunlap/VibeCheck
- **Relevance**: **Most directly relevant paper**. Establishes the concept of "vibes" as a formal construct for characterizing text. Our project extends this from comparing LLM outputs to describing web pages via a specific prompt pattern, then mapping the resulting vibe descriptions in embedding space.

### 2. Demystifying Embedding Spaces using Large Language Models
- **Authors**: Tennenholtz, Chow, Hsu, Jeong, Shani, et al. (Google Research)
- **Year**: 2023 (ICLR 2024)
- **arXiv**: 2310.04475
- **Key Contribution**: Proposes ELM (Embedding Language Model) — trains adapter layers to inject domain embeddings into an LLM, enabling natural language querying of embedding spaces.
- **Methodology**: Two-stage training: first train adapter W→Z with frozen LLM, then fine-tune all parameters. 25 training tasks for movie/user embeddings from MovieLens 25M. Evaluation via semantic consistency (re-embed output, compare to source) and behavioral consistency (use generated profiles for recommendation).
- **Key Results**: High semantic consistency (0.77-0.96 cosine similarity) and human evaluation scores (0.56-0.96) across 24 movie tasks. Can describe hypothetical entities at interpolated embedding points.
- **Relevance**: Demonstrates the inverse of our approach — while we use LLMs to *create* embeddings (via text descriptions), this paper uses LLMs to *interpret* existing embeddings. Could be combined: generate vibe descriptions → embed → use ELM to narrate regions of vibe space.

### 3. Embedding Atlas: Low-Friction, Interactive Embedding Visualization
- **Authors**: Ren, Hohman, Lin, Moritz (Apple)
- **Year**: 2025
- **arXiv**: 2505.06386
- **Key Contribution**: Scalable, interactive browser-based embedding visualization with automatic clustering, labeling, density contours, and cross-filtering.
- **Implementation**: WebGPU rendering, DuckDB for in-browser queries, UMAP for projection, SentenceTransformers for embedding. Supports drag-and-drop Parquet loading, Jupyter widget, and CLI.
- **Key Features**: Millions of points, in-browser UMAP and embedding computation, density contours, real-time nearest-neighbor search, metadata cross-filtering.
- **Code**: https://github.com/apple/embedding-atlas (MIT license)
- **Relevance**: **Primary visualization tool** for the project. Can directly render our vibe embeddings as an interactive map. The wine review demo (196K reviews clustered by description) is a close analogue to our web page vibe map.

### 4. WizMap: Scalable Interactive Visualization for Exploring Large ML Embeddings
- **Authors**: Polo Club of Data Science (Georgia Tech)
- **Year**: 2023
- **arXiv**: 2306.09328
- **Key Contribution**: Multi-resolution embedding visualization using map-like navigation in the browser via WebGL. Scales to millions of points.
- **Code**: https://github.com/poloclub/wizmap
- **Relevance**: Alternative visualization approach with multi-resolution summaries and time-evolution animation.

### 5. UMAP: Uniform Manifold Approximation and Projection
- **Authors**: McInnes, Healy, Melville
- **Year**: 2018
- **arXiv**: 1802.03426
- **Key Contribution**: The foundational dimensionality reduction method for projecting high-dimensional embeddings to 2D while preserving local and global structure.
- **Relevance**: **Core algorithm** for creating the 2D map from high-dimensional vibe embeddings. Standard choice over t-SNE for its better preservation of global structure and faster computation.

### 6. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
- **Authors**: Reimers, Gurevych
- **Year**: 2019
- **arXiv**: 1908.10084
- **Key Contribution**: Siamese/triplet BERT networks for producing semantically meaningful sentence embeddings suitable for cosine-similarity comparison.
- **Relevance**: **Core embedding backbone**. The "it's giving" descriptions need to be embedded using a sentence transformer model (e.g., all-MiniLM-L6-v2 or all-mpnet-base-v2) before UMAP projection.

### 7. Text and Code Embeddings by Contrastive Pre-Training
- **Authors**: OpenAI
- **Year**: 2022
- **arXiv**: 2201.10005
- **Key Contribution**: Shows contrastive pre-training at scale produces high-quality text embeddings for semantic search and classification.
- **Relevance**: Alternative embedding approach using OpenAI's models. Relevant as a comparison embedding backbone.

### 8. Understanding HTML with Large Language Models
- **Authors**: Gur, Nachum, et al. (Google)
- **Year**: 2022
- **arXiv**: 2210.03945
- **Key Contribution**: Demonstrates LLMs pretrained on natural language transfer well to HTML understanding tasks including semantic classification and description generation.
- **Relevance**: Validates that LLMs can meaningfully process web page content, which is the first step in our pipeline (feeding web pages to an LLM for vibe description).

### 9. Beyond a Single Extractor: HTML-to-Text Extraction for LLM Pretraining
- **Authors**: Li, Gardner, et al.
- **Year**: 2026
- **arXiv**: 2602.19548
- **Key Contribution**: Shows extraction strategy significantly impacts what content survives for LLM processing. Different extractors yield substantially different page coverage.
- **Relevance**: Important for our pipeline — the choice of HTML-to-text extraction tool (e.g., trafilatura, readability) will affect what "vibe" the LLM perceives from each page.

### 10. Embedding Style Beyond Topics
- **Authors**: Various
- **Year**: 2025
- **arXiv**: 2501.00828
- **Key Contribution**: Analyzes how writing style (beyond topic) affects the geometry of embedding spaces across multiple language models.
- **Relevance**: Directly relevant — our hypothesis is that "it's giving" descriptions capture style/vibe rather than topic, and this paper provides evidence that style information exists in embedding spaces.

### 11. Discovering Multi-Scale Semantic Structure in Text Corpora
- **Authors**: Haschka et al.
- **Year**: 2025
- **arXiv**: 2512.23471
- **Key Contribution**: Builds hierarchical semantic trees from LLM embedding spaces, revealing multi-scale cluster structure.
- **Relevance**: Provides methodology for understanding hierarchical vibe clusters (e.g., "chill" contains "lo-fi" and "cottagecore" sub-vibes).

### 12. Visualizing Spatial Semantics of Dimensionally Reduced Text Embeddings
- **Authors**: Liu, North, Faust
- **Year**: 2024
- **arXiv**: 2409.03949
- **Key Contribution**: Gradient-based methods to overlay spatial word clouds on 2D document projections, making regions of embedding space interpretable.
- **Relevance**: Could be used to label regions of the vibe map with characteristic words/descriptions.

### 13. SLANG: New Concept Comprehension of LLMs
- **Authors**: Various
- **Year**: 2024
- **arXiv**: 2401.12585
- **Key Contribution**: Addresses how LLMs handle rapidly evolving internet slang and memes.
- **Relevance**: "It's giving" is Gen Z slang — this paper is relevant to understanding whether LLMs can properly use such framing in their outputs.

### 14. Toward Informal Language Processing: Knowledge of Slang in LLMs
- **Authors**: Various
- **Year**: 2024
- **arXiv**: 2404.02323
- **Key Contribution**: Studies how pre-trained LLMs handle slang versus literal equivalents.
- **Relevance**: Calibrating the "it's giving" prompt — LLMs may assign lower probability to slang continuations, affecting output quality.

### 15. Leveraging LLMs for Web Scraping
- **Authors**: Various
- **Year**: 2024
- **arXiv**: 2406.08246
- **Key Contribution**: Using LLMs for structured web content extraction.
- **Relevance**: Alternative approach to web content extraction for our pipeline.

### 16. Characterizing Human Semantic Navigation as Trajectories in Embedding Space
- **Authors**: Various
- **Year**: 2026
- **arXiv**: 2602.05971
- **Key Contribution**: Models human concept navigation as trajectories through transformer embedding space.
- **Relevance**: Directly relevant to the idea of "browsing by vibe similarity" — provides a theoretical framework for how users might navigate through vibe space.

### 17. Embedding Projector: Interactive Visualization and Interpretation of Embeddings
- **Authors**: Smilkov, Thorat, et al. (Google Brain)
- **Year**: 2016
- **arXiv**: 1611.05469
- **Key Contribution**: The seminal interactive embedding visualization tool supporting t-SNE, UMAP, and PCA projections.
- **Relevance**: Historical context for embedding visualization tools.

### 18. Pairwise Judgment for Semantic Embedding in Web Search
- **Authors**: Various
- **Year**: 2024
- **arXiv**: 2408.04197
- **Key Contribution**: Semantic embedding models for web search using Siamese architectures.
- **Relevance**: Relevant to building similarity-based web page retrieval in vibe space.

## Common Methodologies

1. **LLM-as-descriptor**: Using LLMs to generate natural language descriptions of content (VibeCheck, ELM, our "it's giving" approach)
2. **Sentence embedding + UMAP**: Standard pipeline for text-to-2D visualization (Embedding Atlas, WizMap, Embedding Projector)
3. **LLM-as-judge**: Using LLMs to evaluate and compare outputs (VibeCheck's panel of judges)
4. **Contrastive learning for embeddings**: Training embeddings that capture semantic similarity (Sentence-BERT, OpenAI embeddings)

## Standard Baselines

- **Topic-based clustering**: Traditional LDA/TF-IDF topic modeling as a baseline for showing vibe ≠ topic
- **Direct content embedding**: Embedding the raw web page text (without LLM vibe description) as a comparison
- **URL-category classification**: Using existing website categories (DMOZ/Curlie) as ground truth clusters

## Evaluation Metrics

- **Cluster quality**: Silhouette score, adjusted Rand index (if ground-truth categories available)
- **Embedding coherence**: Nearest-neighbor semantic similarity within clusters
- **User study**: Qualitative assessment of whether the vibe map enables meaningful browsing
- **Vibe distinctiveness**: Whether different URLs/web pages get different vibe descriptions (vs. generic responses)

## Gaps and Opportunities

1. **No existing work on "vibe-based web browsing"**: The specific idea of creating a navigable vibe space from web content is novel
2. **"It's giving" as a prompt**: The use of Gen Z slang framing to elicit aesthetic/vibe descriptions is unexplored
3. **Web content → vibe → embedding pipeline**: No existing work chains HTML extraction → LLM vibe description → sentence embedding → 2D map
4. **Comparing vibe similarity vs. topical similarity**: An opportunity to show that vibe space captures something different from traditional topic clustering

## Recommendations for Experiment

- **Primary dataset**: Multi-label web categorization (49K URLs with categories) + C4 web text sample (10K pages)
- **Recommended baselines**: (1) Direct text embedding of web content, (2) Topic-based clustering (LDA), (3) URL category as ground truth
- **Recommended metrics**: Silhouette score, ARI vs. ground-truth categories, qualitative vibe coherence
- **Embedding models**: Start with `all-MiniLM-L6-v2` (fast) and `all-mpnet-base-v2` (more accurate)
- **Visualization**: Use Embedding Atlas for interactive exploration
- **Content extraction**: Use `trafilatura` for clean text from URLs
