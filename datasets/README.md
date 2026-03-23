# Downloaded Datasets

This directory contains datasets for the "A Map of Vibe Space" research project.
Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: Multi-Label Web Categorization

### Overview
- **Source**: https://huggingface.co/datasets/tshasan/multi-label-web-categorization
- **Size**: 49,399 rows, ~19.9 MB
- **Format**: HuggingFace Dataset (Arrow)
- **Task**: Web page categorization with URLs, titles, snippets
- **Categories**: 11 categories (News, Entertainment, Shop, Chat, Education, Government, Health, Technology, Work, Travel, Uncategorized)
- **License**: CC BY 4.0

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("tshasan/multi-label-web-categorization", split="train")
dataset.save_to_disk("datasets/multi_label_web_categorization")
```

### Loading

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/multi_label_web_categorization")
```

### Columns
- `url`: Web page URL
- `title`: Page title
- `snippet`: Text snippet
- `language`: Language code
- `category`: List of category labels
- `meta_description`: Meta description
- `warc_date`: Crawl date

### Notes
- Primary dataset for the vibe pipeline (URLs → extract content → LLM vibe description → embed)
- Categories serve as ground-truth clusters for evaluation
- Many non-English pages included; filter by `language == 'en'` for English-only

---

## Dataset 2: C4 Small Sample

### Overview
- **Source**: https://huggingface.co/datasets/brando/small-c4-dataset
- **Size**: 10,000 rows (train split)
- **Format**: HuggingFace Dataset (Arrow)
- **Task**: Clean web text for LLM processing
- **License**: ODC-BY

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("brando/small-c4-dataset", split="train")
dataset.save_to_disk("datasets/c4_small")
```

### Loading

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/c4_small")
```

### Columns
- `text`: Cleaned web page text content

### Notes
- Pre-extracted clean web text (no HTML parsing needed)
- Good for prototyping the "it's giving" pipeline without web scraping
- No URLs or categories — useful for unsupervised vibe clustering only

---

## Dataset 3: URL Classifications (50K sample)

### Overview
- **Source**: https://huggingface.co/datasets/snats/url-classifications
- **Size**: 50,000 rows (sample from 1.06M total)
- **Format**: HuggingFace Dataset (Arrow)
- **Task**: URL classification into 223 fine-grained categories
- **License**: Unknown

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("snats/url-classifications", split="train")
# Save sample
dataset.select(range(50000)).save_to_disk("datasets/url_classifications_50k")
```

### Loading

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/url_classifications_50k")
```

### Columns
- `url`: Web page URL
- `classification`: Category label (223 fine-grained categories like "tech", "arts", "health")

### Notes
- Very fine-grained categorization (223 classes) for validation
- Full dataset is 1.06M URLs; we saved 50K sample
- URLs need content extraction before LLM processing

---

## Additional Recommended Datasets (Not Downloaded)

### Curlie (DMOZ) Web Directory
- **Source**: https://curlie.org/docs/en/rdf.html
- **Size**: 2.9M entries, ~200 MB compressed
- **Why**: Human-curated web directory with descriptions and hierarchical categories
- **Download**: `curl -O https://curlie.org/directory-dl/curlie_data.tar.gz`

### Chatbot Arena Conversations (LMSYS)
- **Source**: https://huggingface.co/datasets/lmsys/chatbot_arena_conversations
- **Size**: 33K conversations
- **Why**: Paired LLM outputs with human preferences for vibe comparison

### StyleEmbeddingData (Anna Wegmann)
- **Source**: https://huggingface.co/datasets/AnnaWegmann/StyleEmbeddingData
- **Size**: 300K rows
- **Why**: Style-independent-of-content embeddings; companion Style-Embedding model

### LAION-Aesthetics 12M UMAP
- **Source**: https://huggingface.co/datasets/dclure/laion-aesthetics-12m-umap
- **Size**: 2.44 GB
- **Why**: Precedent for UMAP on aesthetic content (image-text pairs with aesthetic scores)
