# data_tools/curate_u.py
import json
import argparse
import sys
from pathlib import Path
from datasets import load_dataset

C4_DATASET = "allenai/c4"
C4_CONFIGS = {"en", "en.noblocklist", "en.noclean", "realnewslike"}
C4_SPLITS = {"train", "validation"}


def dump_stream_limited(
    config: str | None,
    split: str | None,
    out_path: str,
    text_key: str = "text",
    target_mb: int = 500,
    max_items: int | None = None,
    seed: int = 9172,
):
    print(f"[INFO] Starting dataset curation: {C4_DATASET}")
    print(f"[INFO] Target size: {target_mb} MB")

    config, split = _resolve_c4_config_split(config, split)

    print(f"[INFO] Loading dataset: {C4_DATASET} (config: {config}, split: {split})")
    try:
        ds = load_dataset(C4_DATASET, config, split=split, streaming=True)
    except Exception as e:
        print(f"[ERROR] Failed to load C4: {e}", file=sys.stderr)
        sys.exit(1)

    ds = _try_shuffle(ds, seed)

    source_tag = f"{C4_DATASET}:{config}/{split}"
    print(f"[INFO] Source tag: {source_tag}")
    print(f"[INFO] Writing to: {out_path}")

    stats = _stream_and_write(
        ds=ds,
        out_path=out_path,
        text_key=text_key,
        target_mb=target_mb,
        max_items=max_items,
        source_tag=source_tag,
    )

    mb_written = stats["written_bytes"] / (1024 * 1024)
    print(f"[done] wrote {stats['written_lines']} lines, ~{mb_written:.1f} MB to {out_path}")


def _resolve_c4_config_split(config: str | None, split: str | None) -> tuple[str, str]:
    """
    Rules:
      - If split is actually one of the C4 configs, treat it as config and use 'train' split.
      - Default config='en', split='train'.
      - Validate config/split against known sets.
    """
    if split in C4_CONFIGS and (config is None or config == ""):
        print(f"[INFO] Auto-correcting: treating '{split}' as config, using split='train'")
        config, split = split, "train"

    if not config:
        config = "en"
        print(f"[INFO] Auto-selected config: {config}")

    if not split:
        split = "train"

    if config not in C4_CONFIGS:
        print(f"[ERROR] Invalid C4 config '{config}'. Valid: {sorted(C4_CONFIGS)}", file=sys.stderr)
        sys.exit(2)

    if split not in C4_SPLITS:
        print(f"[ERROR] Invalid split '{split}'. Valid: {sorted(C4_SPLITS)}", file=sys.stderr)
        sys.exit(2)

    return config, split


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

                record = {"text": text, "label": "unlabeled", "source": source_tag}
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

        except Exception as e:
            print(f"\n[ERROR] Error during streaming: {e}", file=sys.stderr)
            raise

    return {
        "written_lines": written_lines,
        "written_bytes": written_bytes,
        "seen_examples": seen_examples,
        "skipped_empty": skipped_empty,
        "skipped_missing_key": skipped_missing_key,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Stream and curate C4 (allenai/c4) with byte-budget control",
        epilog="""
Examples:
  # 3 GB of C4 English for U_train
  python data_tools/curate_u.py --config en --split train --target-mb 3072

  # 200 MB of C4 English for U_dev
  python data_tools/curate_u.py --config en --split validation --target-mb 200 --out data/unlabeled/u_dev.jsonl

  # Convenience: pass config via --split (auto-corrects to split=train)
  python data_tools/curate_u.py --split en --target-mb 500
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    ap.add_argument("--config", default="en",
                    help="C4 config: one of {en, en.noblocklist, en.noclean, realnewslike}. Default: en")
    ap.add_argument("--split", default="train",
                    help="Split name: train | validation (default: train)")
    ap.add_argument("--out", default="data/unlabeled/u_train.jsonl",
                    help="Output JSONL file path (default: data/unlabeled/u_train.jsonl)")
    ap.add_argument("--text-key", default="text",
                    help="Field name containing text data (default: text)")
    ap.add_argument("--target-mb", type=int, default=500,
                    help="Target file size in megabytes (default: 500)")
    ap.add_argument("--max-items", type=int, default=None,
                    help="Optional hard cap on number of records (for testing)")
    ap.add_argument("--seed", type=int, default=9172,
                    help="Shuffle seed (default: 9172)")

    args = ap.parse_args()

    cfg = args.config.strip() or None

    dump_stream_limited(
        config=cfg,
        split=args.split,
        out_path=args.out,
        text_key=args.text_key,
        target_mb=args.target_mb,
        max_items=args.max_items,
        seed=args.seed,
    )
