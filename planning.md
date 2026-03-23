# Research Plan: A Map of Vibe Space

## Motivation & Novelty Assessment

### Why This Research Matters
Current web navigation relies on hyperlinks (structural), search engines (keyword/topic), and recommendation algorithms (behavioral). None of these capture the *aesthetic character* or *vibe* of web content. A user looking for "cozy, handcrafted, indie" content must manually browse and judge each page. This research proposes a fundamentally new navigation paradigm: browsing the web as a spatial map where proximity reflects similarity in vibe rather than topic or link structure.

### Gap in Existing Work
Per the literature review:
- **VibeCheck** (Dunlap et al., 2024) formalizes vibes for comparing LLM outputs, but doesn't apply this to web content or create navigable spaces.
- **Embedding Atlas** and **WizMap** provide visualization tools but focus on topic/semantic clustering, not aesthetic/vibe clustering.
- **No prior work** combines: web content extraction → LLM vibe description → sentence embedding → navigable 2D map.
- The specific "it's giving" prompt pattern (using Gen Z slang to elicit aesthetic descriptions) is entirely unexplored.

### Our Novel Contribution
1. A pipeline that transforms web pages into vibe descriptions using a specific LLM prompt pattern ("it's giving")
2. Empirical evidence that vibe-based embeddings capture different information than raw text embeddings (vibe ≠ topic)
3. A concrete prototype of "vibe space navigation" — an interactive map where distant-in-clicks pages cluster by shared aesthetic

### Experiment Justification
- **Experiment 1 (Vibe Generation)**: Tests whether LLMs produce meaningfully diverse vibe descriptions from web content, or collapse to generic outputs.
- **Experiment 2 (Vibe vs. Raw Embedding)**: Tests the core hypothesis — do vibe embeddings organize pages differently than raw text embeddings?
- **Experiment 3 (Cluster Quality)**: Quantifies whether vibe space reveals meaningful structure beyond topic categories.
- **Experiment 4 (Cross-Domain Discovery)**: Tests the key user value — can vibe space surface surprising connections between pages from different domains?

## Research Question
Does prompting an LLM with web pages and asking "What is this web page giving? Start your answer with 'it's giving'" produce descriptions that, when embedded in vector space, create a navigable "vibe map" capturing aesthetic similarity distinct from topical similarity?

## Hypothesis Decomposition
1. **H1 (Vibe Diversity)**: LLM-generated "it's giving" descriptions are meaningfully diverse across different web pages (not generic).
2. **H2 (Vibe ≠ Topic)**: Vibe embeddings cluster pages differently than raw text embeddings — pages with similar vibes but different topics cluster together.
3. **H3 (Cluster Quality)**: Vibe space exhibits meaningful cluster structure (measurable via silhouette score and qualitative inspection).
4. **H4 (Cross-Domain Discovery)**: Pages from different traditional categories (e.g., cooking blog and DIY craft site) can be neighbors in vibe space.

## Proposed Methodology

### Approach
Use the C4 Small dataset (10K pre-extracted web page texts) as our primary corpus. Sample 500 diverse pages for the LLM pipeline (balancing cost and statistical power). Generate vibe descriptions via GPT-4.1, embed them with sentence-transformers, project to 2D with UMAP, and compare against raw text embedding baselines.

### Pipeline
```
Web page text → GPT-4.1 ("What is this web page giving?") → "It's giving..." description
→ sentence-transformers (embed) → UMAP (2D projection) → Interactive visualization
```

### Experimental Steps
1. **Data sampling**: Select 500 diverse pages from C4 (stratified by text length and content diversity)
2. **Vibe generation**: Prompt GPT-4.1 with each page's text + the "it's giving" prompt
3. **Embedding**: Embed both raw texts and vibe descriptions using `all-mpnet-base-v2`
4. **Dimensionality reduction**: UMAP projection to 2D for both embedding sets
5. **Clustering**: HDBSCAN on both embedding spaces
6. **Comparison**: Measure cluster overlap (ARI), silhouette scores, nearest-neighbor analysis
7. **Visualization**: Interactive scatter plots with hover text showing vibe descriptions

### Baselines
1. **Raw text embedding**: Embed the original web page text directly (no LLM step)
2. **TF-IDF + topic modeling**: LDA topic clusters as a topical baseline
3. **Random baseline**: Random 2D positions (sanity check)

### Evaluation Metrics
- **Silhouette score**: Cluster cohesion/separation quality
- **Vibe diversity**: Unique n-grams, vocabulary richness across vibe descriptions
- **Adjusted Rand Index (ARI)**: Agreement between vibe clusters and raw text clusters (low ARI = vibe captures different info)
- **Cross-domain neighbor rate**: Fraction of nearest neighbors that come from different topic categories
- **Qualitative coherence**: Manual inspection of vibe clusters for aesthetic consistency

### Statistical Analysis Plan
- Paired t-tests / Wilcoxon signed-rank for metric comparisons
- Bootstrap confidence intervals (1000 resamples) for silhouette scores
- Permutation tests for ARI significance
- α = 0.05 with Bonferroni correction for multiple comparisons

## Expected Outcomes
- **Support for hypothesis**: Vibe embeddings produce coherent clusters with high silhouette scores AND low ARI vs. raw text clusters (meaning vibe space captures something different from topic space)
- **Refutation**: If vibe descriptions are generic/repetitive, or if vibe clusters closely mirror topic clusters (high ARI)

## Timeline and Milestones
1. Planning: 10 min ✓
2. Environment setup: 5 min
3. Data preparation + sampling: 10 min
4. Vibe generation via GPT-4.1: 20 min (API calls)
5. Embedding + UMAP: 10 min
6. Analysis + visualization: 30 min
7. Documentation: 20 min

## Potential Challenges
- **API rate limits**: Batch requests, implement retry logic
- **Generic vibe outputs**: If LLM produces repetitive descriptions → try prompt variations
- **Cost**: 500 pages × ~1K tokens each ≈ $2-5, well within budget
- **Short/uninformative pages in C4**: Filter by minimum text length

## Success Criteria
1. Vibe descriptions show meaningful diversity (>80% unique descriptions)
2. Vibe clusters have silhouette score > 0.1 (non-trivial structure)
3. ARI between vibe clusters and raw text clusters < 0.3 (capturing different info)
4. Qualitative inspection reveals aesthetically coherent vibe neighborhoods
