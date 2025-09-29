# data_tools/curate_u.py
"""
Goal
----
Stream a large text dataset from Hugging Face Datasets and write out a capped
unlabeled JSONL file of approximately N megabytes, without downloading the
entire source dataset. The output is used as the "U" split for alignment-aware
tokenization experiments.

Deliverables
------------
1) A CLI tool that writes JSONL lines of the form:
   {"text": <str>, "label": "unlabeled", "source": "<dataset[:config]/split>"}
2) Byte-budget capping: stop writing when the on-disk bytes would exceed
   --target-mb (approximate is OK).
3) Robust dataset loading across:
   - C4 (allenai/c4) which REQUIRES a config (e.g., "en") and has splits "train"/"validation".
   - The Pile (EleutherAI/pile) which REQUIRES a config (e.g., "all") and
     often needs `trust_remote_code=True`, and may stream from flaky mirrors.
   - Pile Uncopyrighted (monology/pile-uncopyrighted) with no config.
4) Optional buffered shuffle if the dataset supports iterable shuffling.
5) Helpful logging + graceful fallbacks.

Functional Requirements
-----------------------
A) Loading:
   - If --dataset == "allenai/c4":
       * require or auto-detect a valid config (default "en").
       * split is usually "train" or "validation".
   - If --dataset == "EleutherAI/pile":
       * require or auto-pick "all" if no config provided.
       * accept --trust-remote-code to satisfy the loader.
       * handle mirror 404s as a soft failure (see D).
   - If --dataset == "monology/pile-uncopyrighted":
       * no config required; split "train".
   - Otherwise:
       * Try `load_dataset(ds_name, [config], split=split, streaming=True, ...)`.
       * If config names exist and none is provided, pick the first or a sensible default.

B) Output logic:
   - Iterate streaming examples, read `text_key` (default "text").
   - Skip empty/whitespace-only text.
   - Append JSONL lines with the specified schema.
   - Keep a running byte count using UTF-8 encoded length. Stop when adding
     the next line would exceed --target-mb (in MiB).

C) Shuffle (optional):
   - Attempt `ds.shuffle(seed=..., buffer_size=10_000)` inside a try/except.
   - If unsupported, proceed without shuffling.

D) Error handling & fallbacks:
   - Pile mirror 404: catch `FileNotFoundError` from the iterator and:
       * Log a warning that output may be partial, then break cleanly.
       * (Optional stretch) add a `--fallback-dataset` to auto-switch to
         "monology/pile-uncopyrighted" or "allenai/c4 --config en".
   - Config errors: if a provided config is invalid, print the list of
     available configs (via `get_dataset_config_names`) and exit(2).
   - Auth/gated datasets: provide a clear message suggesting
     `huggingface-cli login` or setting HUGGINGFACE_HUB_TOKEN.

E) Telemetry:
   - Print a final summary: lines written and approximate MB.
   - Print the detected/used source tag "dataset[:config]/split".
   - (Optional) print acceptance rate (written/seen) if cheap to compute.

Non-Functional Requirements
---------------------------
- Python 3.10+.
- No external deps beyond `datasets`.
- Deterministic behavior given the same seed (when shuffling supported).
- Windows-friendly file I/O (use encoding='utf-8').

Edge Cases To Handle
--------------------
- Datasets with different text columns; allow --text-key override.
- Missing text_key: skip examples rather than crash.
- Very small target budgets (< 5 MB).
- Interrupted runs should still leave a valid partial JSONL.
- Invalid combos (e.g., EleutherAI/pile with config "en"): fail fast with a
  helpful error that lists valid configs.

Testing Checklist
-----------------
1) C4 happy path:
   $ python scripts/curate_u_stream_cap.py --dataset allenai/c4 --config en --split train --target-mb 50
   - Produces ~50 MB JSONL, prints source "allenai/c4:en/train".

2) Pile happy path:
   $ python scripts/curate_u_stream_cap.py --dataset EleutherAI/pile --config all --split train --target-mb 50 --trust-remote-code
   - Produces ~50 MB JSONL or stops early with a clear warning if a mirror 404 occurs.

3) Pile-uncopyrighted:
   $ python scripts/curate_u_stream_cap.py --dataset monology/pile-uncopyrighted --split train --target-mb 50
   - Works without trust_remote_code.

4) Tiny budget:
   $ python scripts/curate_u_stream_cap.py --dataset allenai/c4 --config en --split validation --target-mb 5
   - Writes a handful of lines and exits cleanly.

5) Wrong config:
   $ python scripts/curate_u_stream_cap.py --dataset EleutherAI/pile --config en
   - Exits with available config list including 'all'.

Implementation Plan (Pseudo-code)
---------------------------------
def dump_stream_limited(ds_name, config, split, out_path, text_key="text",
                        target_mb=500, max_items=None, seed=9172,
                        trust_remote_code=False):
    """
    1) Resolve dataset config:
       - cfgs = get_dataset_config_names(ds_name)
       - If cfgs exists:
           * If config is None: choose sensible default
             (C4 -> "en"; Pile -> "all"; else cfgs[0]).
           * If config not in cfgs: raise helpful ValueError.

    2) Fix common C4 misuse:
       - If ds_name == "allenai/c4" and config is None and split in {"en", ...}:
           config, split = split, "train"

    3) Build dataset:
       - kwargs = {"split": split, "streaming": True}
       - If trust_remote_code: kwargs["trust_remote_code"] = True
       - ds = load_dataset(ds_name, config, **kwargs) if config else load_dataset(ds_name, **kwargs)

    4) Try buffered shuffle:
       - try: ds = ds.shuffle(seed=seed, buffer_size=10_000)
         except: pass

    5) Iterate and write:
       - target_bytes = target_mb * 1024 * 1024
       - written = n = 0
       - open(out_path, "w", encoding="utf-8") as f:
           for ex in ds:
               try:
                   t = (ex.get(text_key) or "").strip()
               except AttributeError:
                   continue
               if not t: continue
               rec = json.dumps({"text": t, "label": "unlabeled",
                                 "source": f"{ds_name}:{config}/{split}" if config else f"{ds_name}:{split}"}) + "\n"
               b = rec.encode("utf-8")
               if written + len(b) > target_bytes: break
               f.write(rec)
               written += len(b); n += 1
               if max_items and n >= max_items: break

    6) Print summary and return.
    """

CLI Arguments
-------------
--dataset (str)                 : HF dataset path, e.g. allenai/c4, EleutherAI/pile
--config (str, optional)        : dataset config (C4: en; Pile: all). If empty, auto-pick.
--split (str)                   : split to use (default: train)
--out (path)                    : output JSONL path
--text-key (str)                : text field name (default: text)
--target-mb (int)               : byte budget in MiB (default: 500)
--max-items (int, optional)     : hard cap on number of lines
--trust-remote-code (flag)      : required for some datasets (e.g., EleutherAI/pile)

Examples
--------
# 500 MB of English C4 (train)
python data_tools/curate_u_stream_cap.py --dataset allenai/c4 --config en --split train --target-mb 500

# 500 MB of Pile (may need login + trust_remote_code)
python data_tools/curate_u_stream_cap.py --dataset EleutherAI/pile --config all --split train --target-mb 500 --trust-remote-code

# 500 MB of Pile Uncopyrighted (no custom code)
python data_tools/curate_u_stream_cap.py --dataset monology/pile-uncopyrighted --split train --target-mb 500
"""

import json
import argparse
from datasets import load_dataset, get_dataset_config_names

# Known configs for auto-correction / hints. Keep this up to date as needed.
KNOWN_CONFIGS = {
    "allenai/c4": {
        "configs": {
            "en": ["train", "validation"],
            "en.noblocklist": ["train", "validation"],
            "en.noclean": ["train", "validation"],
            "realnewslike": ["train", "validation"],
        }
    },
    "EleutherAI/pile": {"configs": {"all": ["train"]}},
    "monology/pile-uncopyrighted": {"configs": {"default": ["train"]}},
}


def dump_stream_limited(
    ds_name: str,
    config: str | None,
    split: str | None,
    out_path: str,
    text_key: str = "text",
    target_mb: int = 500,
    max_items: int | None = None,
    seed: int = 9172,
    trust_remote_code: bool = False,
):
    """
    TODO: Implement per the module docstring.

    Required behaviors to pass review:
    - Correct config handling (auto-pick for C4/Pile; validate against get_dataset_config_names).
    - Streaming iteration with optional buffered shuffle (try/except).
    - Byte-budget cap using UTF-8 byte lengths.
    - Resilient error handling for mirror 404s and gated datasets.
    - Clear logging (source tag, lines written, MB written).
    """
    raise NotImplementedError("Implement dump_stream_limited per the module docstring.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="e.g., 'allenai/c4', 'EleutherAI/pile', 'monology/pile-uncopyrighted'")
    ap.add_argument("--config", default="", help="dataset config if required (e.g., 'en' for c4, 'all' for pile). Leave empty for auto.")
    ap.add_argument("--split", default="train", help="split name, e.g. 'train', 'validation'")
    ap.add_argument("--out", default="data/unlabeled/u_train.jsonl")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--target-mb", type=int, default=500)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--trust-remote-code", action="store_true", help="Required for datasets like EleutherAI/pile")
    args = ap.parse_args()

    # NOTE: Intentional stub call to show signature; will raise until implemented.
    cfg = None if args.config.strip() == "" else args.config.strip()
    dump_stream_limited(
        ds_name=args.dataset,
        config=cfg,
        split=args.split,
        out_path=args.out,
        text_key=args.text_key,
        target_mb=args.target_mb,
        max_items=args.max_items,
        trust_remote_code=args.trust_remote_code,
    )
