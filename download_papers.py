#!/usr/bin/env python3
"""Download papers from arXiv."""
import subprocess
import sys
import os

papers = [
    ("2410.12851", "vibecheck"),
    ("1802.03426", "umap"),
    ("2505.06386", "embedding_atlas"),
    ("2306.09328", "wizmap"),
    ("1611.05469", "embedding_projector"),
    ("2210.03945", "html_llm"),
    ("2310.04475", "demystifying_embeddings"),
    ("1908.10084", "sentence_bert"),
    ("2201.10005", "contrastive_embeddings"),
    ("2409.03949", "spatial_semantics"),
    ("2501.00828", "embedding_style"),
    ("2401.12585", "slang"),
    ("2404.02323", "informal_language"),
    ("2406.08246", "llm_web_scraping"),
    ("2602.05971", "semantic_navigation"),
    ("2602.19548", "html_to_text_extraction"),
    ("2512.23471", "multiscale_semantic_structure"),
    ("2408.04197", "pairwise_judgment_embedding"),
]

os.makedirs("papers", exist_ok=True)
for arxiv_id, name in papers:
    outfile = f"papers/{arxiv_id}_{name}.pdf"
    if os.path.exists(outfile) and os.path.getsize(outfile) > 1000:
        print(f"SKIP {outfile} (already exists)")
        continue
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    result = subprocess.run(
        ["curl", "-sL", "-o", outfile, url],
        capture_output=True, text=True, timeout=30
    )
    size = os.path.getsize(outfile) if os.path.exists(outfile) else 0
    if size > 1000:
        print(f"OK   {outfile} ({size} bytes)")
    else:
        print(f"FAIL {outfile} ({size} bytes)")
        if os.path.exists(outfile):
            os.remove(outfile)
