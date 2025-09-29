# data_tools/curate_benign.py
"""
CURATE_BENIGN — IMPLEMENTATION SPEC (READ FIRST)

Goal
-----
Produce a small CLI that curates *benign* prompts from public HF datasets and
writes them to JSONL for evaluation. The script should filter out obviously
unsafe/toxic terms and (for RTP) exclude prompts with non-trivially high
toxicity scores.

Datasets (initial)
------------------
1) databricks/databricks-dolly-15k  (split: 'train')
   - Field to use: 'instruction' (string)
   - Label: "benign"
   - Source tag: "dolly-15k"

2) allenai/real-toxicity-prompts  (split: 'train')
   - Field to use: 'prompt' -> 'text' (string)
   - Optional field: 'toxicity' (float or None)
   - Keep only clearly non-toxic prompts:
       * if `toxicity` is float and > 0.2 → EXCLUDE
       * if `toxicity` is None/missing → treat as unknown (DEFAULT: keep)
   - Label: "benign"
   - Source tag: "real-toxicity-prompts"

Filtering Rules
---------------
- Maintain a conservative exclude regex for obviously unsafe topics:
    weapon, kill, poison, explosive, assault, harm, suicide, self-harm,
    drugs, terror, hate, extrem*, violent
  (case-insensitive; see EXCLUDE_RE below)

- OPTIONAL niceties (documented, not required):
  * Trim whitespace; skip empty strings
  * Minimum length threshold (e.g., ≥ 8 characters) if it helps avoid junk

Output
------
- Newline-delimited JSON (UTF-8), each line:
    {"text": <str>, "label": "benign", "source": <source_tag>}
- Write at most N records for each dataset (--n-dolly / --n-rtp)
- Shuffle before capping (seeded) to avoid topic clusters at the top

CLI (must support exactly)
--------------------------
--n-dolly        int, max samples from Dolly (default: 1500)
--n-rtp          int, max samples from RTP (default: 500)
--out-dolly      path, default: data/eval/benign_1500.jsonl
--out-rtp        path, default: data/eval/benign_rtp_extra_500.jsonl
--run            {'both','dolly','rtp'}, default 'both'
--seed           int, default 9172
--tox-thresh     float, default 0.2 (RTP only: exclude if toxicity > thresh)

Behavior & Logging
------------------
- Dolly:
  * Load split 'train'; collect ex['instruction'] if present.
  * Apply EXCLUDE_RE filter; skip empties.
  * Shuffle with seed; cap to --n-dolly; write JSONL with source 'dolly-15k'.
  * Print: "[done] dolly → {out} ({written} lines)"

- RTP:
  * Load split 'train'; get text via ex.get('prompt', {}).get('text', '').
  * If toxicity is a float and > --tox-thresh, skip.
  * Apply EXCLUDE_RE; skip empties.
  * Shuffle with seed; cap to --n-rtp; write JSONL with source 'real-toxicity-prompts'.
  * Print: "[done] rtp → {out} ({written} lines)"

- Run mode:
  * If --run=dolly, only run Dolly path (ignore RTP).
  * If --run=rtp, only run RTP path.
  * If --run=both, run both (order doesn’t matter).

Acceptance Tests
----------------
1) Dolly only (small sample):
   $ python scripts/curate_benign.py --run dolly --n-dolly 100 --out-dolly /tmp/dolly.jsonl
   - File exists with ~100 lines; each line has label "benign" and source "dolly-15k".
   - No lines match EXCLUDE_RE terms (case-insensitive).

2) RTP only (tox filter):
   $ python scripts/curate_benign.py --run rtp --n-rtp 200 --out-rtp /tmp/rtp.jsonl --tox-thresh 0.2
   - File exists with ~200 lines; label "benign", source "real-toxicity-prompts".
   - No lines match EXCLUDE_RE; no example with toxicity > 0.2 included.

3) Both (default):
   $ python scripts/curate_benign.py
   - Produces the default two files; prints two "[done] ..." summaries.

Implementation Hints
--------------------
- Use: from datasets import load_dataset
- Keep memory low: gather candidate strings into a Python list, shuffle, slice,
  write line-by-line (the datasets are small enough).
- Always open outputs with encoding='utf-8'.
- Prefer robust accessors:
    t = (ex.get("instruction") or "").strip()
    t = (ex.get("prompt") or {}).get("text", "").strip()
- Toxicity handling:
    tox = ex.get("toxicity", None)
    if isinstance(tox, float) and tox > thresh: continue
"""

from datasets import load_dataset
import re
import json
import random
import argparse
from typing import Optional

SEED = 9172

# Keep this exclude list stable; collaborators should not weaken it without review.
EXCLUDE_RE = re.compile(
    r"\b(weapon|kill|poison|explosive|assault|harm|suicide|self-harm|drugs|terror|hate|extrem|violent)\b",
    re.I
)


def dump_dolly(n: int, out: str, seed: int = SEED) -> None:
    """
    TODO: Implement Dolly curation as specified in the module docstring.

    Steps:
      - ds = load_dataset("databricks/databricks-dolly-15k", split="train")
      - For each example:
          * t = (ex.get("instruction") or "").strip()
          * if not t: continue
          * if EXCLUDE_RE.search(t): continue
          * collect t
      - random.seed(seed); random.shuffle(rows); rows = rows[:n]
      - Write JSONL to `out`: {"text": t, "label": "benign", "source": "dolly-15k"}
      - Print summary: f"[done] dolly → {out} ({len(rows)} lines)"
    """
    raise NotImplementedError


def dump_rtp(n: int, out: str, tox_thresh: float = 0.2, seed: int = SEED) -> None:
    """
    TODO: Implement Real Toxicity Prompts curation as specified.

    Steps:
      - ds = load_dataset("allenai/real-toxicity-prompts", split="train")
      - For each example:
          * t = (ex.get("prompt") or {}).get("text", "").strip()
          * tox = ex.get("toxicity", None)
          * if not t: continue
          * if isinstance(tox, float) and tox > tox_thresh: continue
          * if EXCLUDE_RE.search(t): continue
          * collect t
      - random.seed(seed); random.shuffle(rows); rows = rows[:n]
      - Write JSONL to `out`: {"text": t, "label": "benign", "source": "real-toxicity-prompts"}
      - Print summary: f"[done] rtp → {out} ({len(rows)} lines)"
    """
    raise NotImplementedError


def main() -> None:
    """
    TODO: Wire up the CLI and run the selected path(s).

    CLI:
      --n-dolly    (int, default 1500)
      --n-rtp      (int, default 500)
      --out-dolly  (str, default data/eval/benign_1500.jsonl)
      --out-rtp    (str, default data/eval/benign_rtp_extra_500.jsonl)
      --run        (choices: both|dolly|rtp, default both)
      --seed       (int, default 9172)
      --tox-thresh (float, default 0.2)

    Behavior:
      - Parse args.
      - If args.run in ('both','dolly') and args.n-dolly > 0: call dump_dolly(args.n_dolly, args.out_dolly, args.seed)
      - If args.run in ('both','rtp')   and args.n-rtp   > 0: call dump_rtp(args.n_rtp, args.out_rtp, args.tox_thresh, args.seed)
      - Exit 0.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
