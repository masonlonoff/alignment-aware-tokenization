# scripts/mine_neutrals_from_u.py
import re
import json
import argparse
import hashlib
import random
import unicodedata  # NEW
from collections import defaultdict

SEED = 9172
random.seed(SEED)

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "to", "of", "in", "on", "at", "by", "with",
    "from", "as", "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those", "it", "its",
    "i", "you", "he", "she", "we", "they", "them", "us", "our", "your", "their", "my", "me", "mine", "yours", "theirs"
}

DENY_RE = re.compile(
    r"\b(how to|make a|build (?:a|an)|step[- ]?\d+|kill|harm|poison|explode|weapon|recipe for|tutorial)\b",
    re.I
)

RISK_RE = re.compile(
    r"\b(weapon|gun|knife|bomb|explosive|kill|murder|assault|attack\b|violent|suicide|self-?harm|"
    r"terror|extremis|isis|kkk|nazi|poison|toxin|overdose|cocaine|heroin|meth|lsd|mdma|ecstasy|"
    r"hack|ddos|crack|piracy|steal|phishing|ponzi)\b",
    re.I
)

BENIGN_CONTEXT = [
    r"photo\s+shoot", r"shoot(?:ing)?\s+(?:scene|film|movie|schedule)", r"shooting\s+star",
    r"bombastic", r"photo\s*bomb", r"seed\s+bomb",
    r"attack\s+surface", r"attack\s+rate", r"attack\s+vector",
    r"toxic(?:ity)?\s+(?:policy|classifier|comment[s]?)",
    r"bullet\s+points", r"explosive\s+growth",
    r"band|song|album|movie|film|metaphor|idiom|slang",
    r"test\s+server|dev\s+server|cloud", r"sandbox|practice|dummy|harmless|mock|example",
]
BENIGN_CTX_RE = re.compile("|".join(BENIGN_CONTEXT), re.I)

# Extra benign modifiers that may appear just outside the 60-char window  # NEW
BENIGN_MODS_WIDE = re.compile(
    r"\b(photo|film|movie|scene|metaphor|idiom|bombastic|band|song|album|"
    r"server|cloud|sandbox|practice|dummy|harmless|example)\b", re.I
)

# Fast, strict normalization for dedup


def strict_norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def normalize_text(t: str) -> str:
    t = unicodedata.normalize("NFKC", (t or "").strip())
    t = re.sub(r"\s+", " ", t)
    return t


def stems_from_anchors(anchors_path, min_len=3, max_len=24):
    stems = set()
    with open(anchors_path, "r", encoding="utf-8") as f:
        for ln in f:
            try:
                t = json.loads(ln)["text"]
            except Exception:
                continue
            for w in re.split(r"\s+|[^\w]", (t or "").lower()):
                w = "".join(ch for ch in w if ch.isalnum())
                if not w:
                    continue
                if not (min_len <= len(w) <= max_len):
                    continue
                if w in STOPWORDS:
                    continue
                if re.fullmatch(r"[a-z]{3}", w) and w in {"and", "for", "the", "not", "all", "any"}:
                    continue
                stems.add(w)
    return stems


def compile_stem_regex(stems):
    escaped = []
    for s in stems:
        if s.isalpha():
            escaped.append(rf"\b{re.escape(s)}\b")
        else:
            escaped.append(re.escape(s))
    escaped.sort(key=len, reverse=True)
    pattern = "|".join(escaped)
    return re.compile(pattern, re.I) if pattern else None


def is_candidate_neutral(text: str) -> bool:
    if not text or len(text) < 15:
        return False
    if len(text) > 1200:  # NEW: tighter cap so neutrals aren’t giant pages
        return False
    if "http://" in text or "https://" in text or "www." in text:
        return False
    if re.match(r"^\s*(answer|assistant|response)\s*[:\-]", text, re.I):
        return False
    if "```" in text or re.search(r"\bclass\s+\w+|def\s+\w+\(", text):
        return False
    if DENY_RE.search(text):
        return False
    if RISK_RE.search(text):
        return False
    noisy = sum(1 for ch in text if ord(ch) < 9 or ord(ch) > 126)
    if noisy > 0.05 * len(text):
        return False
    return True


def has_benign_context_near(match_span, text, window=60) -> bool:
    start, end = match_span
    a = max(0, start - window)
    b = min(len(text), end + window)
    ctx = text[a:b]
    if BENIGN_CTX_RE.search(ctx):
        return True
    # Allow slightly looser benign modifiers if the strict window missed it
    if BENIGN_MODS_WIDE.search(text[max(0, start-120):min(len(text), end+120)]):
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--u_path", default="data/unlabeled/u_train.jsonl")
    ap.add_argument("--anchors", default="data/anchors/anchors_500.jsonl")
    ap.add_argument("--out", default="data/neutrals/neutrals_1000.jsonl")
    ap.add_argument("--per_stem", type=int, default=2)
    ap.add_argument("--cap", type=int, default=1000)
    ap.add_argument("--min_len", type=int, default=3)
    args = ap.parse_args()

    stems = stems_from_anchors(args.anchors, min_len=args.min_len)
    stem_re = compile_stem_regex(stems)
    if stem_re is None:
        raise SystemExit("No usable stems extracted from anchors.")

    # Avoid exact overlaps with anchor texts (normalized)
    anchor_norm = set()
    with open(args.anchors, "r", encoding="utf-8") as f:
        for ln in f:
            try:
                t = json.loads(ln)["text"]
                anchor_norm.add(strict_norm(t))
            except Exception:
                pass

    picked = defaultdict(int)
    seen_hash = set()
    out = []

    with open(args.u_path, "r", encoding="utf-8") as f:
        for ln in f:
            if len(out) >= args.cap:
                break
            try:
                j = json.loads(ln)
            except Exception:
                continue
            t = normalize_text(j.get("text", ""))
            low = t.lower()
            if not is_candidate_neutral(low):
                continue

            # Exact overlap with any anchor text?
            if strict_norm(low) in anchor_norm:
                continue

            # Fast stem scan
            m = stem_re.search(low)
            if not m:
                continue

            # Require benign context near the matched stem
            if not has_benign_context_near(m.span(), low, window=60):
                ok = False
                for mm in stem_re.finditer(low):
                    if has_benign_context_near(mm.span(), low, window=60):
                        m = mm
                        ok = True
                        break
                if not ok:
                    continue

            # Identify the matched stem quickly (O(1))
            match_txt = low[m.start():m.end()]
            candidate_stems = [match_txt] if match_txt in stems else []
            if not candidate_stems:
                window = low[max(0, m.start()-20):min(len(low), m.end()+20)]
                toks = re.findall(r"[A-Za-z]{3,}", window)
                candidate_stems = [tok for tok in toks if tok in stems]

            # Per-stem quotas
            placed = False
            for s in candidate_stems:
                if picked[s] < args.per_stem:
                    picked[s] += 1
                    placed = True
                    break
            if not placed:
                continue

            # Global dedup (strict normalization)
            k = hashlib.md5(strict_norm(low).encode("utf-8")).hexdigest()  # NEW
            if k in seen_hash:
                continue
            seen_hash.add(k)

            out.append({
                "text": t,
                "label": "neutral",
                "source": "U-mined",
                "meta": {"stems": candidate_stems[:3]}
            })

    random.shuffle(out)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Stats
    distinct_stems = sum(1 for _, c in picked.items() if c > 0)
    top10 = sorted(picked.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"[done] neutrals: {len(out)}  (cap={args.cap}, per_stem={args.per_stem})")
    print(f"[done] distinct stems covered: {distinct_stems}/{len(stems)}")
    print(f"[done] top stems: {top10}")


if __name__ == "__main__":
    main()
