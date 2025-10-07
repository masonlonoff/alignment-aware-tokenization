# data_tools/curate_benign.py
"""
CURATE_BENIGN — IMPLEMENTATION SPEC (READ FIRST)

Goal
-----
Produce a small Command Line Tool that curates *benign* prompts from two public HuggingFace datasets
(databricks/databricks-dolly-15k & allenai/real-toxicity-prompts) and
writes them to JSONL for evaluation. At the end we will have "safe" prompts.
The script should filter out obviously unsafe/toxic terms for both datasets and (for RTP) exclude prompts with non-trivially high
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

- OPTIONAL niceties (documented, not required, but included here):
  * Trim whitespace; skip empty strings
  * Minimum length threshold (≥ 8 characters) to avoid junk

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
   $ python scripts/curate_benign.py --run dolly --n-dolly 100 --out-dolly data/eval/dolly.jsonl
   - File exists with ~100 lines; each line has label "benign" and source "dolly-15k".
   - No lines match EXCLUDE_RE terms (case-insensitive).

2) RTP only (tox filter):
   $ python scripts/curate_benign.py --run rtp --n-rtp 200 --out-rtp data/eval/rtp.jsonl --tox-thresh 0.2
   - File exists with ~200 lines; label "benign", source "real-toxicity-prompts".
   - No lines match EXCLUDE_RE; no example with toxicity > 0.2 included.

3) Both (default):
   $ python data_tools/curate_benign.py
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
from pathlib import Path

SEED = 9172
MIN_LEN = 8  # optional nicety (enforced here)

# Keep this exclude list stable; collaborators should not weaken it without review.
EXCLUDE_RE = re.compile(
    r"\b(weapon|kill|poison|explosive|assault|harm|suicide|self-harm|drugs|terror|hate|extrem|violent)\b",
    re.I
)

def _ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def _write_jsonl(rows, out_path: str, source: str) -> int:
    """Write list[str] to JSONL with fields (text,label,source). Returns #lines."""
    _ensure_parent_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for t in rows:
            f.write(json.dumps({"text": t, "label": "benign", "source": source}, ensure_ascii=False) + "\n")
    return len(rows)

def dump_dolly(n: int, out: str, seed: int = SEED) -> None:
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    rows = []
    for ex in ds:
        t = (ex.get("instruction") or "").strip()
        if not t:
            continue
        if len(t) < MIN_LEN:  # nicety: skip junky/ultra-short prompts
            continue
        if EXCLUDE_RE.search(t):
            continue
        rows.append(t)

    random.seed(seed)
    random.shuffle(rows)
    rows = rows[: max(0, int(n))]

    written = _write_jsonl(rows, out, source="dolly-15k")
    print(f"[done] dolly → {out} ({written} lines)")

def dump_rtp(n: int, out: str, tox_thresh: float = 0.2, seed: int = SEED) -> None:
    ds = load_dataset("allenai/real-toxicity-prompts", split="train")
    rows = []
    for ex in ds:
        prompt_obj = ex.get("prompt") or {}
        t = (prompt_obj.get("text") or "").strip()
        if not t:
            continue
        if len(t) < MIN_LEN:  # nicety
            continue

        tox = ex.get("toxicity", None)
        # Treat ints as numbers too; exclude only if numeric and above threshold
        if isinstance(tox, (int, float)) and float(tox) > float(tox_thresh):
            continue

        if EXCLUDE_RE.search(t):
            continue

        rows.append(t)

    random.seed(seed)
    random.shuffle(rows)
    rows = rows[: max(0, int(n))]

    written = _write_jsonl(rows, out, source="real-toxicity-prompts")
    print(f"[done] rtp → {out} ({written} lines)")

def main() -> None:
    parser = argparse.ArgumentParser(description="Curate benign prompts from Dolly and RTP datasets.")
    parser.add_argument("--n-dolly", type=int, default=1500, help="Max samples from Dolly (default: 1500)")
    parser.add_argument("--n-rtp", type=int, default=500, help="Max samples from RTP (default: 500)")
    parser.add_argument("--out-dolly", type=str, default="data/eval/benign_1500.jsonl", help="Output path for Dolly prompts")
    parser.add_argument("--out-rtp", type=str, default="data/eval/benign_rtp_extra_500.jsonl", help="Output path for RTP prompts")
    parser.add_argument("--run", choices=["both", "dolly", "rtp"], default="both", help="Which dataset(s) to curate")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--tox-thresh", type=float, default=0.2, help="Toxicity threshold for RTP")
    args = parser.parse_args()

    if args.run in ("both", "dolly") and args.n_dolly > 0:
        dump_dolly(args.n_dolly, args.out_dolly, args.seed)
    if args.run in ("both", "rtp") and args.n_rtp > 0:
        dump_rtp(args.n_rtp, args.out_rtp, args.tox_thresh, args.seed)

if __name__ == "__main__":
    main()
    