# A Map of Vibe Space

**Can we navigate the internet as a map of "vibe space" instead of links and topics?**

This project tests whether prompting an LLM with web pages and asking *"What is this web page giving?"* creates descriptions that, when embedded in vector space, organize the web by aesthetic similarity rather than topical similarity.

## Key Findings

- **100% unique vibe descriptions**: GPT-4.1 generates genuinely distinctive, culturally fluent characterizations for every web page ("it's giving cozy niche forum energy", "corporate LinkedIn energy", "soft girl eco-chic, Montessori-core vibes")
- **Vibe ≠ Topic**: Vibe-space clusters and content-space clusters agree only ~11% (ARI=0.113, p=0.0002) — vibe captures fundamentally different information
- **Cross-domain discovery works**: Vibe-space neighbors have 47% lower content similarity (Cohen's d=1.44), meaning pages from different domains cluster by shared aesthetic character
- **293 compelling cross-domain pairs found**: E.g., a Romanian collab blog and a British movie blog both described as "early-2010s personal blog energy"
- **Interactive vibe map**: Browse the results at `figures/vibe_map_interactive.html`

## Pipeline

```
Web page text → GPT-4.1 ("What is this web page giving?")
  → "It's giving..." description → Sentence embedding (all-mpnet-base-v2)
  → UMAP 2D projection → Interactive visualization
```

## Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv add datasets openai sentence-transformers umap-learn plotly matplotlib seaborn scikit-learn tqdm

# Set API key
export OPENAI_API_KEY=your_key_here

# Run pipeline
python src/01_sample_and_generate_vibes.py  # ~20 min, ~$3
python src/02_embed_and_project.py          # ~30 sec
python src/03_analyze_and_visualize.py      # ~1 min
python src/04_statistical_tests.py          # ~2 min
```

## File Structure

```
├── REPORT.md                          # Full research report with results
├── planning.md                        # Research plan and methodology
├── src/
│   ├── 01_sample_and_generate_vibes.py  # LLM vibe generation pipeline
│   ├── 02_embed_and_project.py          # Embedding + UMAP projection
│   ├── 03_analyze_and_visualize.py      # Analysis + visualizations
│   └── 04_statistical_tests.py          # Statistical significance tests
├── results/
│   ├── data/                          # Raw data and embeddings
│   ├── analysis_results.json          # All metrics
│   └── statistical_tests.json         # Statistical test results
├── figures/
│   ├── vibe_map_interactive.html      # Interactive browsable map
│   ├── vibe_map_annotated.png         # Annotated vibe space
│   ├── raw_vs_vibe_space.png          # Side-by-side comparison
│   ├── cluster_comparison.png         # Clustering metrics
│   └── neighbor_overlap.png           # Overlap analysis
├── datasets/                          # Pre-downloaded datasets
├── papers/                            # Reference papers
└── code/                              # Reference implementations
```

## Full Report

See [REPORT.md](REPORT.md) for complete methodology, results, statistical analysis, and discussion.
