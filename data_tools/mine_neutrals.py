# data_tools/mine_neutrals.py
import re
import json
import hashlib
import random
from collections import defaultdict

SEED = 9172
random.seed(SEED)


def stems_from_anchors(anchors_path, min_len=3):
    stems = set()
    for ln in open(anchors_path, "r", encoding="utf-8"):
        t = json.loads(ln)["text"].lower()
        for w in re.split(r"\s+|[^\w]", t):
            w = "".join(ch for ch in w if ch.isalnum())
            if len(w) >= min_len:
                stems.add(w)
    return stems


def is_benign_like(text):
    # light heuristics: avoid imperative "how to", "make a", numbers-as-steps
    return not re.search(r"\b(how to|make a|build a|kill|harm|poison|explode|weapon)\b", text, re.I)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--u_path", default="data/unlabeled/u_train.jsonl")
    ap.add_argument("--anchors", default="data/anchors/anchors_500.jsonl")
    ap.add_argument("--out", default="data/neutrals/neutrals_1000.jsonl")
    ap.add_argument("--per_stem", type=int, default=2)
    ap.add_argument("--cap", type=int, default=1000)
    args = ap.parse_args()

    stems = stems_from_anchors(args.anchors)
    picked = defaultdict(int)
    seen = set()
    out = []
    for ln in open(args.u_path, "r", encoding="utf-8"):
        j = json.loads(ln)
        t = (j["text"] or "")
        low = t.lower()
        if not is_benign_like(low):
            continue
        hits = [s for s in stems if s in low]
        if not hits:
            continue
        # keep only clearly benign phrasing around the stem
        if re.search(r"\b(photo\s+shoot|shooting star|bombastic|photobomb|bullet points|explosive growth)\b", low):
            k = hashlib.md5(low.encode()).hexdigest()
            if k in seen:
                continue
            # assign to first hit whose quota not full
            placed = False
            for s in hits:
                if picked[s] < args.per_stem:
                    picked[s] += 1
                    placed = True
                    break
            if not placed:
                continue
            seen.add(k)
            out.append({"text": t, "label": "neutral", "source": "U-mined", "meta": {"stems": hits}})
            if len(out) >= args.cap:
                break

    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r)+"\n")
    print("neutrals:", len(out))


if __name__ == "__main__":
    main()
