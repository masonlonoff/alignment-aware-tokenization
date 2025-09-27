import re
import json
import argparse
from datasets import load_dataset

BENIGN_PATTERNS = [
    r"\bphoto shoot\b", r"\bbombastic\b", r"\bshooting star\b", r"\bphotobomb\b",
    r"\bbullet points\b", r"\bexplosive growth\b"
]


def main(args):
    ds = load_dataset(args.dataset, split="train", streaming=True)
    out = open(args.out, "w")
    for ex in ds:
        text = ex.get("text", "").strip()
        if any(re.search(p, text, flags=re.I) for p in BENIGN_PATTERNS):
            out.write(json.dumps({"text": text, "label": "neutral"}) + "\n")
    out.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="segyges/OpenWebText2")
    p.add_argument("--out", default="data/neutral_lookalikes.jsonl")
    main(p.parse_args())
