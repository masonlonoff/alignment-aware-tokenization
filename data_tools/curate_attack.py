# data_tools/curate_attack.py
"""
CURATE_ATTACK — IMPLEMENTATION SPEC (READ FIRST)

Goal
-----
Build a small CLI that harvests “attack” prompts from one or more Hugging Face
datasets (e.g., AdvBench, JailbreakV-28K) and writes them to JSONL for eval.

Deliverable
-----------
A script that can:
  1) Load a dataset (with/without config; possibly gated) and discover a valid split
     when the user hasn’t provided one.
  2) Pick an appropriate text field from each example (auto-detect if needed).
  3) Shuffle (when supported), cap the number of examples, and write JSONL lines:
       {"text": <str>, "label": "attack", "source": "<dataset[:config]/split>"}
  4) Allow running either/both sources via a --run flag: {both|adv|jbv}.
  5) Handle gated datasets gracefully (AdvBench), warning and continuing.

CLI (must support exactly)
--------------------------
--hf-token            Optional token for gated datasets (e.g., AdvBench).
--trust-remote-code   Pass through to datasets.load_dataset (some repos require).
--n-adv               Max examples from AdvBench (default: 1000).
--n-jbv               Max examples from JailbreakV-28K (default: 500).
--out-adv             Output file for AdvBench (default: data/eval/attack_1000.jsonl).
--out-jbv             Output file for JailbreakV-28K (default: data/eval/attack_extra_500.jsonl).
--run                 Which source(s) to run: 'both' (default), 'adv', or 'jbv'.
--adv-ds              Dataset ID for AdvBench (default: walledai/AdvBench).
--adv-config          Optional config for AdvBench (default: None).
--adv-split           Optional split for AdvBench (default: None → discover).
--jbv-ds              Dataset ID for JBV (default: JailbreakV-28K/JailBreakV-28k).
--jbv-config          Config for JBV (default: JailBreakV_28K)  # REQUIRED by that repo
--jbv-split           Optional split for JBV (default: None → discover).

Behavior & Rules
----------------
1) Split discovery:
   - If split is not provided, call get_dataset_split_names(ds, config_name=config, token=token)
     and choose in priority order: "train" > "training" > "default" > "all" > first available.
   - If the call fails, default to "train".

2) Loading:
   - Use load_dataset(ds, [config], split=split, trust_remote_code=flag).
   - If --hf-token is provided, pass it. Prefer `token=...`, fallback to `use_auth_token=...`
     for older datasets. If loading fails (gated/403/404/whatever), warn and return 0 lines.

3) Text extraction:
   - Try fields in order: ["text","prompt","instruction","query","request","Behavior","behavior"].
   - If none found, fall back to the first non-empty string field with length ≥ 10.
   - Skip examples with empty text after stripping.

4) Shuffle & cap:
   - Attempt d.shuffle(seed=9172). If not supported, ignore error and proceed.
   - Write up to N examples per selected dataset (--n-adv / --n-jbv).

5) Output:
   - Each line is JSON: {"text": <str>, "label": "attack", "source": "<ds[:config]/split>"}.
   - Encoding: UTF-8; newline-terminated.

6) Run mode:
   - Use --run to control which dataset(s) actually execute.
   - If --n-adv == 0 or --run != 'adv'/'both', skip AdvBench.
   - If --n-jbv == 0 or --run != 'jbv'/'both', skip JBV.

7) Logging:
   - For each dataset run, print: "[done] {ds} → {out_path} ({written} lines)".
   - If AdvBench fails due to gating/repo issues, print:
       "[diag] AdvBench: skipped or empty (gated or load failed)."

Edge Cases
----------
- AdvBench is gated: require login/token; if not available, skip with a warning.
- JailbreakV-28K requires a config (e.g., JailBreakV_28K). If user changes it
  and it’s invalid, surface the underlying HF error.
- Different schemas: rely on auto text-field detection as described.

Non-Goals
---------
- No need to stream or to support compression.
- No multiprocessing or progress bars.

Acceptance Tests
----------------
1) JBV only:
   $ python scripts/curate_attack.py --run jbv --n-jbv 200 --out-jbv /tmp/jbv.jsonl
   - Produces ~200 lines, prints "[done] JailbreakV-28K/..." message.

2) AdvBench only (with token):
   $ huggingface-cli login
   $ python scripts/curate_attack.py --run adv --n-adv 100 --out-adv /tmp/adv.jsonl
   - Produces ~100 lines, prints "[done] walledai/AdvBench ..." message.

3) AdvBench without token:
   $ python scripts/curate_attack.py --run adv --n-adv 100 --out-adv /tmp/adv.jsonl
   - Prints a warning and writes 0 lines; JBV is not run.

4) Both (default):
   $ python data_tools/curate_attack.py
   - Runs both; honors --n-adv and --n-jbv defaults and out paths.

5) Text-field fallback:
   - Temporarily point --adv-ds to a JSON dataset with a "prompt" field only;
     ensure it is picked and non-empty lines are produced.

Implementation Hints
--------------------
- Use:
    from datasets import load_dataset, get_dataset_split_names
- Keep memory use low: write each line as you iterate.
- Make sure to catch and handle exceptions when loading gated datasets.
"""

import json
import random
import argparse
import sys
from typing import Optional, List, Tuple, Dict

from datasets import load_dataset, get_dataset_split_names

SEED = 9172
random.seed(SEED)

TEXT_FIELDS = ["text", "prompt", "instruction", "query", "request", "Behavior", "behavior"]


def pick_text(ex: dict) -> Optional[str]:
    """
    TODO: Return a usable text string from `ex`.

    Behavior:
      - Iterate TEXT_FIELDS; return the first non-empty, stripped string.
      - Else return the first non-empty string field with length ≥ 10.
      - Return None if nothing is suitable.
    """
    raise NotImplementedError


def discover_split(ds: str, config: Optional[str], token: Optional[str]) -> str:
    """
    TODO: Discover a reasonable split to use for a dataset.

    Steps:
      - Try: splits = get_dataset_split_names(ds, config_name=config, token=token)
      - Choose in priority order: 'train' > 'training' > 'default' > 'all' > first.
      - On exception, return 'train'.
    """
    raise NotImplementedError


def load_ds(ds: str, config: Optional[str], split: str, token: Optional[str], trust_remote_code: bool):
    """
    TODO: Load a dataset split with optional token and trust_remote_code.

    Steps:
      - Build kwargs = { 'split': split, 'trust_remote_code': trust_remote_code }.
      - If token is provided:
          * First try load_dataset(..., token=token, **kwargs).
          * If TypeError, fall back to load_dataset(..., use_auth_token=token, **kwargs).
      - If config is provided, pass it as the second argument to load_dataset.
      - Return the loaded dataset (or let exceptions bubble to caller for warning/skip).
    """
    raise NotImplementedError


def dump_generic(
    ds: str,
    out_path: str,
    n: int,
    *,
    config: Optional[str] = None,
    split: Optional[str] = None,
    token: Optional[str] = None,
    trust_remote_code: bool = False,
    label: str = "attack",
    source_tag: Optional[str] = None,
) -> int:
    """
    TODO: Implement the main harvesting logic for a single dataset.

    Required behavior:
      - Determine `split` (use discover_split if None).
      - Call load_ds to get the dataset; on exception, print a warning with ds/config/split and return 0.
      - Attempt to shuffle with d.shuffle(seed=SEED) (wrap in try/except).
      - Iterate examples, use pick_text(ex) to extract a string; skip empties.
      - Write up to `n` lines to out_path as JSONL with fields: text, label, source.
        * source = source_tag if provided else f"{ds}:{config}/{split}" or f"{ds}:{split}".
      - Print a completion line: "[done] {ds} → {out_path} ({written} lines)"
      - Return the number of lines written (int).
    """
    raise NotImplementedError


def main():
    """
    TODO: Wire up CLI and run selected jobs.

    Steps:
      - Parse arguments matching the CLI in the module docstring.
      - Define a helper `should(run_tag, n)` that returns True when:
          * --run is 'both' or equals run_tag, and n > 0.
      - If should('adv', n_adv): run dump_generic for AdvBench (args.adv_ds, args.out_adv, args.n_adv, ...).
          * If the result is 0, print: "[diag] AdvBench: skipped or empty (gated or load failed)."
      - If should('jbv', n_jbv): run dump_generic for JailbreakV-28K (args.jbv_ds, args.out_jbv, args.n_jbv, ...).
      - Exit 0.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
