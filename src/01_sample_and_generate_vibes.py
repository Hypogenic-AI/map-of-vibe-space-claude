"""
Step 1: Sample web pages from C4 and generate "it's giving" vibe descriptions via GPT-4.1.

Pipeline: C4 text → filter by quality → sample 500 → GPT-4.1 vibe prompt → save results
"""

import os
import json
import random
import time
import numpy as np
from datasets import Dataset
from openai import OpenAI
from tqdm import tqdm

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Config
SAMPLE_SIZE = 500
MIN_TEXT_LENGTH = 200  # Filter out very short pages
MAX_TEXT_LENGTH = 5000  # Truncate very long pages for API efficiency
MAX_PROMPT_TEXT = 2000  # Max chars of page text to include in prompt
OUTPUT_PATH = "results/data/vibe_descriptions.json"

def load_and_filter_c4():
    """Load C4 dataset and filter for quality."""
    ds = Dataset.from_file("datasets/c4_small/data-00000-of-00001.arrow")
    texts = ds["text"]

    # Filter by length
    filtered = [(i, t) for i, t in enumerate(texts) if len(t) >= MIN_TEXT_LENGTH]
    print(f"C4 total: {len(texts)}, after length filter (>={MIN_TEXT_LENGTH} chars): {len(filtered)}")

    # Sample
    if len(filtered) > SAMPLE_SIZE:
        sampled = random.sample(filtered, SAMPLE_SIZE)
    else:
        sampled = filtered

    print(f"Sampled: {len(sampled)} pages")
    return sampled


def generate_vibe(client, text, page_idx):
    """Generate a vibe description for a single web page."""
    # Truncate text if needed
    truncated = text[:MAX_PROMPT_TEXT]
    if len(text) > MAX_PROMPT_TEXT:
        truncated += "..."

    prompt = f"""Here is the content of a web page:

---
{truncated}
---

What is this web page giving? Start your answer with "it's giving"."""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a cultural commentator who describes the vibe, aesthetic, and energy of things using contemporary internet language. Be specific, evocative, and concise (1-3 sentences)."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  Error on page {page_idx}: {e}")
        return None


def main():
    print("=" * 60)
    print("Step 1: Sample C4 pages and generate vibe descriptions")
    print("=" * 60)

    # Load and sample
    sampled = load_and_filter_c4()

    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Generate vibes
    results = []
    failed = 0

    print(f"\nGenerating vibe descriptions for {len(sampled)} pages...")
    for idx, (orig_idx, text) in enumerate(tqdm(sampled)):
        vibe = generate_vibe(client, text, idx)

        if vibe is None:
            failed += 1
            continue

        results.append({
            "page_idx": orig_idx,
            "text": text[:MAX_TEXT_LENGTH],  # Store truncated for embedding later
            "vibe_description": vibe,
            "text_length": len(text),
        })

        # Brief pause to avoid rate limits (GPT-4.1 is fast)
        if idx % 50 == 0 and idx > 0:
            print(f"  Progress: {idx}/{len(sampled)}, failed: {failed}")

    print(f"\nDone! Generated {len(results)} vibe descriptions, {failed} failures.")

    # Save results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {OUTPUT_PATH}")

    # Print samples
    print("\n--- Sample Vibe Descriptions ---")
    for r in random.sample(results, min(5, len(results))):
        print(f"\nPage text (first 100 chars): {r['text'][:100]}...")
        print(f"Vibe: {r['vibe_description']}")


if __name__ == "__main__":
    main()
