"""
Goal
----
Stream a large text dataset from Hugging Face Datasets and write out a capped
unlabeled JSONL file of approximately N megabytes, without downloading the
entire source dataset. The output is used as the "U" split for alignment-aware
tokenization experiments.
"""

import json
import argparse
import sys
from pathlib import Path
from datasets import load_dataset, get_dataset_config_names

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
    "segyges/OpenWebText2": {"configs": None},
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
    print(f"[INFO] Starting dataset curation: {ds_name}")
    print(f"[INFO] Target size: {target_mb} MB")
    
    config, split = _resolve_config(ds_name, config, split)
    
    print(f"[INFO] Loading dataset: {ds_name}" + (f" (config: {config})" if config else ""))
    try:
        ds = _load_streaming_dataset(ds_name, config, split, trust_remote_code)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}", file=sys.stderr)
        if "trust_remote_code" in str(e).lower():
            print("[HINT] Try adding --trust-remote-code flag", file=sys.stderr)
        if "gated" in str(e).lower() or "authentication" in str(e).lower():
            print("[HINT] This dataset may require authentication. Try:", file=sys.stderr)
            print("       huggingface-cli login", file=sys.stderr)
            print("       or set HUGGINGFACE_HUB_TOKEN environment variable", file=sys.stderr)
        sys.exit(1)
    
    ds = _try_shuffle(ds, seed)
    
    source_tag = f"{ds_name}:{config}/{split}" if config else f"{ds_name}:{split}"
    print(f"[INFO] Source tag: {source_tag}")
    print(f"[INFO] Writing to: {out_path}")
    
    stats = _stream_and_write(
        ds, out_path, text_key, target_mb, max_items, source_tag
    )
    
    _print_summary(stats, target_mb)


def _resolve_config(ds_name: str, config: str | None, split: str | None) -> tuple[str | None, str]:
    if ds_name == "allenai/c4" and config is None and split in {"en", "en.noblocklist", "en.noclean", "realnewslike"}:
        print(f"[INFO] Auto-correcting: treating '{split}' as config, using split='train'")
        return split, "train"

    if ds_name == "monology/pile-uncopyrighted":
        return None, (split or "train")

    try:
        available_configs = get_dataset_config_names(ds_name)
    except Exception:
        available_configs = None

    if config is None:
        if ds_name == "allenai/c4":
            config = "en"
            print(f"[INFO] Auto-selected config: {config}")
        elif ds_name == "EleutherAI/pile":
            config = "all"
            print(f"[INFO] Auto-selected config: {config}")
        elif ds_name == "segyges/OpenWebText2":
            print(f"[INFO] No config needed for {ds_name}")
        elif available_configs and len(available_configs) > 0:
            config = available_configs[0]
            print(f"[INFO] Auto-selected config: {config}")

    if config and available_configs and config not in available_configs:
        print(f"[ERROR] Invalid config '{config}' for dataset '{ds_name}'", file=sys.stderr)
        print(f"[ERROR] Available configs: {', '.join(available_configs)}", file=sys.stderr)
        sys.exit(2)

    if ds_name == "EleutherAI/pile" and config != "all":
        print(f"[ERROR] EleutherAI/pile requires '--config all' (got '{config}')", file=sys.stderr)
        sys.exit(2)

    return config, (split or "train")


def _load_streaming_dataset(ds_name: str, config: str | None, split: str, trust_remote_code: bool):
    kwargs = {"split": split, "streaming": True}
    
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    
    if config:
        return load_dataset(ds_name, config, **kwargs)
    else:
        return load_dataset(ds_name, **kwargs)


def _try_shuffle(ds, seed: int):
    try:
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
        print(f"[INFO] Shuffling enabled with buffer_size=10,000 and seed={seed}")
    except Exception as e:
        print(f"[WARN] Shuffling not supported: {e}")
        print("[WARN] Proceeding without shuffle")
    
    return ds


def _stream_and_write(ds, out_path: str, text_key: str, target_mb: int, 
                      max_items: int | None, source_tag: str) -> dict:
    target_bytes = target_mb * 1024 * 1024
    written_bytes = 0
    written_lines = 0
    seen_examples = 0
    skipped_empty = 0
    skipped_missing_key = 0
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            try:
                for ex in ds:
                    seen_examples += 1
                    
                    try:
                        text = ex.get(text_key)
                        if text is None:
                            skipped_missing_key += 1
                            continue
                        text = str(text).strip()
                    except (AttributeError, KeyError):
                        skipped_missing_key += 1
                        continue
                    
                    if not text:
                        skipped_empty += 1
                        continue
                    
                    record = {
                        "text": text,
                        "label": "unlabeled",
                        "source": source_tag
                    }
                    line = json.dumps(record, ensure_ascii=False) + "\n"
                    line_bytes = line.encode("utf-8")
                    
                    if written_bytes + len(line_bytes) > target_bytes:
                        print(f"[INFO] Reached byte budget ({target_mb} MB), stopping")
                        break
                    
                    f.write(line)
                    written_bytes += len(line_bytes)
                    written_lines += 1
                    
                    if written_lines % 1000 == 0:
                        mb_written = written_bytes / (1024 * 1024)
                        print(f"[PROGRESS] {written_lines} lines, {mb_written:.2f} MB", end="\r")
                    
                    if max_items and written_lines >= max_items:
                        print(f"\n[INFO] Reached max_items limit ({max_items}), stopping")
                        break
                        
            except FileNotFoundError as e:
                print(f"\n[WARN] Mirror error encountered: {e}")
                print("[WARN] Output file may be partial due to upstream availability issues")
                print("[WARN] Consider using --dataset monology/pile-uncopyrighted as fallback")
            except Exception as e:
                print(f"\n[ERROR] Error during streaming: {e}", file=sys.stderr)
                raise
                
    except Exception as e:
        print(f"\n[ERROR] Failed to write output: {e}", file=sys.stderr)
        raise
    
    return {
        "written_lines": written_lines,
        "written_bytes": written_bytes,
        "seen_examples": seen_examples,
        "skipped_empty": skipped_empty,
        "skipped_missing_key": skipped_missing_key,
    }


def _print_summary(stats: dict, target_mb: int):
    mb_written = stats["written_bytes"] / (1024 * 1024)
    acceptance_rate = (stats["written_lines"] / stats["seen_examples"] * 100) if stats["seen_examples"] > 0 else 0
    
    print("\n" + "=" * 60)
    print("CURATION SUMMARY")
    print("=" * 60)
    print(f"Lines written:       {stats['written_lines']:,}")
    print(f"Bytes written:       {stats['written_bytes']:,} ({mb_written:.2f} MB)")
    print(f"Target:              {target_mb} MB")
    print(f"Examples processed:  {stats['seen_examples']:,}")
    print(f"Acceptance rate:     {acceptance_rate:.1f}%")
    print(f"Skipped (empty):     {stats['skipped_empty']:,}")
    print(f"Skipped (no key):    {stats['skipped_missing_key']:,}")
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Stream and curate large text datasets with byte-budget control",
        epilog="""
Examples:
  # 3 GB of OpenWebText2 for U_train
  python data_tools/curate_u.py --dataset segyges/OpenWebText2 --split train --target-mb 3000
  
  # 200 MB of OpenWebText2 for U_dev
  python data_tools/curate_u.py --dataset segyges/OpenWebText2 --split train --target-mb 200 --out data/unlabeled/u_dev.jsonl
  
  # Alternative: C4 English
  python data_tools/curate_u.py --dataset allenai/c4 --config en --split train --target-mb 3000
  
  # Alternative: The Pile (requires trust_remote_code)
  python data_tools/curate_u.py --dataset EleutherAI/pile --config all --split train --target-mb 3000 --trust-remote-code
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    ap.add_argument("--dataset", required=True, 
                    help="HuggingFace dataset path (e.g., 'segyges/OpenWebText2', 'allenai/c4', 'EleutherAI/pile')")
    ap.add_argument("--config", default="", 
                    help="Dataset config if required (e.g., 'en' for C4, 'all' for Pile). Leave empty for auto-detection.")
    ap.add_argument("--split", default="train", 
                    help="Split name (default: train)")
    ap.add_argument("--out", default="data/unlabeled/u_train.jsonl",
                    help="Output JSONL file path (default: data/unlabeled/u_train.jsonl)")
    ap.add_argument("--text-key", default="text",
                    help="Field name containing text data (default: text)")
    ap.add_argument("--target-mb", type=int, default=500,
                    help="Target file size in megabytes (default: 500)")
    ap.add_argument("--max-items", type=int, default=None,
                    help="Optional hard cap on number of records (for testing)")
    ap.add_argument("--trust-remote-code", action="store_true", 
                    help="Required for datasets like EleutherAI/pile that execute custom code")
    
    args = ap.parse_args()

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