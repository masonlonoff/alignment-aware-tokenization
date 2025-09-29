# data_tools/curate_anchors.py
"""
CURATE_ANCHORS — IMPLEMENTATION SPEC (READ FIRST)

Goal
-----
Build a CLI tool that curates ~≤500 short “hazard anchor” strings from one or more
Hugging Face datasets (e.g., JailbreakBench Behaviors, AdvBench). The anchors are
single-text fields (no step-by-step instructions), deduplicated, and written as a
newline-delimited JSONL file for downstream use.

Deliverable
-----------
A script that:
  1) Loads one or more source datasets (with/without configs; with/without auth).
  2) Discovers valid splits when unknown (e.g., JBB has 'harmful' and 'benign').
  3) Selects the proper text field per example (auto-detect if --text-key missing).
  4) Applies a conservative anti–how-to filter (exclude procedural/step lists).
  5) Deduplicates normalized strings and caps the output to --cap items.
  6) Emits JSONL: {"text": <str>, "label": "hazard", "source": <tag>, "meta": {}}
  7) Prints diagnostics (counts, chosen field names, splits used).

CLI (must support exactly)
--------------------------
--hf-token          Optional; HF token for gated datasets (e.g., AdvBench).
--out               Output path (default: data/anchors/anchors_500.jsonl).
--cap               Max number of anchors to write (default: 500).
--no-step-filter    Disable the step filter (debug or to widen recall).
--text-key          Optional explicit field name to read (else auto-detect).

Datasets (initial set)
----------------------
1) JailbreakBench/JBB-Behaviors
   - Requires config 'behaviors'
   - Splits are NOT 'train'; discover actual splits (usually ['harmful','benign']).
   - Public (no token required).

2) walledai/AdvBench
   - Often gated; requires login or token. If loading fails, skip with a warning.
   - Splits/configs may vary; discover if not known.

Behavior & Rules
----------------
1) Split discovery:
   - If no split is specified (we won’t expose a split flag), use:
       get_dataset_split_names(ds_name, config_name=?)
     and iterate all returned splits. On error, default to ['train'].

2) Loading:
   - Use load_dataset(ds_name, [config], split=sp).
   - If --hf-token is provided, pass it (prefer `token=...`, fallback to `use_auth_token=...`).
   - If a dataset fails to load (auth/gated/etc.), print a warning and continue.

3) Selecting text:
   - If --text-key was given, use ex.get(text_key).
   - Else auto-detect from the first example of each split using the ordered candidates:
       ["text","behavior","Behavior","prompt","instruction","question","content","inputs","input","request"]
     Then fallback to the first non-empty string field ≥ 10 characters.
   - Skip examples whose selected text is empty or shorter than a small threshold (e.g., 8 chars).

4) Step/procedural filter:
   - By default, filter out “how-to”/step-by-step phrasing using a conservative regex that
     matches numbers/steps explicitly (e.g., r"(?i)\\b(step\\s*\\d+|^\\s*\\d+\\.|follow these steps)\\b").
   - The flag --no-step-filter disables this filter entirely.

5) Normalize + dedup:
   - Define norm(t): lowercase, collapse internal whitespace: " ".join(t.lower().split()).
   - De-duplicate using a hash (e.g., md5(norm(t))) before writing.

6) Output:
   - For each accepted, unique anchor t:
       {"text": t, "label": "hazard", "source": "<DATASET+CONFIG or TAG>", "meta": {}}
   - Stop after writing --cap records.

7) Diagnostics (print):
   - Per dataset: splits used, total examples seen, kept, and a map of chosen text field → count.
   - Final: “anchors: {N} (wrote → {out})”.

Edge Cases & Robustness
-----------------------
- If a dataset is gated (e.g., AdvBench) and token is missing/invalid:
    - Warn: “[diag] AdvBench: skipped or empty (gated or load failed).”
    - Continue with the others.

- If the step filter prunes too aggressively:
    - Users can re-run with --no-step-filter to debug and compare diagnostics.

- If zero anchors are produced:
    - Ensure diagnostics help explain whether the issue is text field selection, filtering, or access.

Non-Goals
---------
- No need for multiprocessing or progress bars.
- No requirement to stream; in-memory loading is acceptable (datasets are small).
- No requirement to support arbitrary datasets beyond the two above, but code should be generic.

Acceptance Tests (run locally)
------------------------------
1) JBB only:
   $ python data_tools/curate_anchors.py
   - Prints diagnostic with splits (e.g., ['harmful','benign']), kept > 0, and writes ≤ 500 anchors.

2) Disable step filter:
   $ python data_tools/curate_anchors.py --no-step-filter --out /tmp/anchors_nostep.jsonl
   - Kept should be ≥ test #1; file is valid JSONL.

3) AdvBench gated (no token):
   $ python data_tools/curate_anchors.py --out /tmp/anchors_jbb_only.jsonl
   - Warns about AdvBench; still writes anchors from JBB.

4) AdvBench with token:
   $ HUGGINGFACE_HUB_TOKEN=<token> python data_tools/curate_anchors.py --hf-token <token> --out /tmp/anchors_full.jsonl
   - Includes anchors from both JBB and AdvBench (if accessible).

5) Text key override:
   - Add --text-key behavior and verify it’s used (by forcing a non-default dataset or mocking).

Implementation Hints
--------------------
- Use:
    from datasets import load_dataset, get_dataset_split_names
- Helper functions you’ll likely need:
    pick_text_field(example) -> (key, value)
    list_splits(name, config) -> [splits]
    norm(s) -> str
- Maintain small memory footprint: iterate and write line-by-line.
- Keep logs succinct and actionable.

"""

import os
import sys
import re
import json
import hashlib
import random
import argparse
from typing import Optional, Tuple, List, Dict
from datasets import load_dataset, get_dataset_split_names

SEED = 9172

# Default candidates for text auto-detection (keep order)
TEXT_CANDIDATES = [
    "text", "behavior", "Behavior", "prompt", "instruction", "question",
    "content", "inputs", "input", "request"
]

# Conservative step/recipe pattern (numbers/explicit steps only)
STEP_RE = re.compile(r"(?i)\b(step\s*\d+|^\s*\d+\.\s|follow these steps)\b")


def norm(s: str) -> str:
    """Lowercase + collapse whitespace."""
    return " ".join(s.lower().split())


def pick_text_field(ex: dict, explicit_key: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    TODO: Implement selection of a text field for a single example.

    Behavior:
      - If explicit_key is provided, return (explicit_key, ex.get(explicit_key, '').strip()).
      - Else iterate TEXT_CANDIDATES and return the first non-empty string.
      - Else return the first non-empty string field of length ≥ 10.
      - Return (None, None) if nothing suitable found.
    """
    raise NotImplementedError


def list_splits(ds_name: str, config: Optional[str]) -> List[str]:
    """
    TODO: Implement split discovery.

    Use get_dataset_split_names(ds_name, config_name=config), returning a list of split names.
    On error, return ['train'] as a fallback.
    """
    raise NotImplementedError


def take(
    ds_name: str,
    nmax: Optional[int] = None,
    config: Optional[str] = None,
    split_mode: str = "auto",
    hf_token: Optional[str] = None,
    trust_remote_code: bool = False,
    min_len: int = 8,
    apply_step_filter: bool = True,
    explicit_text_key: Optional[str] = None,
    diag_prefix: str = "",
) -> Tuple[List[str], Dict]:
    """
    TODO: Implement per the module docstring.

    Steps:
      - Determine splits: if split_mode in (None, 'auto') -> list_splits(...), else [split_mode].
      - For each split:
          * Load dataset split with load_dataset(ds_name, [config], split=sp, token/use_auth_token if provided).
          * For the first example, pick a text key (explicit_text_key or auto-detect).
          * Iterate examples:
              - Extract text with chosen key; skip if empty or shorter than min_len.
              - If apply_step_filter and STEP_RE matches, skip.
              - Collect into rows; stop early at nmax if provided.
          * Track diagnostics: totals, kept, by_key counts, splits list.
      - Return (rows, diag_dict)
    """
    raise NotImplementedError


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-token", default=os.getenv("HUGGINGFACE_HUB_TOKEN", ""))
    ap.add_argument("--out", default="data/anchors/anchors_500.jsonl")
    ap.add_argument("--cap", type=int, default=500)
    ap.add_argument("--no-step-filter", action="store_true",
                    help="Disable step-by-step filter if it over-prunes.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 1) JBB Behaviors (gives harmful/benign splits). Config is required.
    jbb_rows, jbb_diag = take(
        "JailbreakBench/JBB-Behaviors",
        nmax=600,  # take a bit more before dedup
        config="behaviors",
        split="auto",
        hf_token=args.hf_token,
        apply_step_filter=not args.no_step_filter,
        diag_prefix="[JBB] "
    )
    print(
        f"[diag] JBB: splits={jbb_diag['splits']} seen={jbb_diag['seen']} kept={jbb_diag['kept']} by_key={jbb_diag['by_key']}")

    # 2) AdvBench (gated); if not accessible, we’ll just skip it.
    adv_rows, adv_diag = take(
        "walledai/AdvBench",
        nmax=600,
        split="auto",
        hf_token=args.hf_token or None,  # requires login/token
        apply_step_filter=not args.no_step_filter,
        diag_prefix="[ADV] "
    )
    if adv_diag["seen"] == 0 and adv_diag["kept"] == 0:
        print("[diag] AdvBench: skipped or empty (gated or load failed).")

    # Combine, dedup, cap
    seen = set()
    out = []
    for t in (jbb_rows + adv_rows):
        k = hashlib.md5(norm(t).encode()).hexdigest()
        if k in seen:
            continue
        seen.add(k)
        out.append({"text": t, "label": "hazard", "source": "JBB+AdvBench", "meta": {}})
        if len(out) >= args.cap:
            break

    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    print(f"anchors: {len(out)} (wrote → {args.out})")


if __name__ == "__main__":
    main()
