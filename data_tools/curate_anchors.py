# data_tools/curate_anchors.py
import os
import sys
import re
import json
import hashlib
import random
from typing import Optional, Tuple, Callable
from collections import defaultdict, Counter
from datasets import load_dataset, get_dataset_split_names

SEED = 9172
random.seed(SEED)

# -------------------------
# Patterns & helpers
# -------------------------
STEP_RE = re.compile(
    r"(?i)\b("
    r"step\s*\d+|^\s*\d+\.\s|^\s*-\s*\d+\s|first\s*,\s*second\s*,|"
    r"follow these steps|in order to build"
    r")\b"
)

INTENT_RE = re.compile(
    r"(\b(how|what|why|when|where|who)\b.+\?$)|"        # WH- questions
    r"(\b(can|could|would|should|may|might)\b.+\?$)|"   # modal questions
    r"(\b(write|explain|describe|summarize|compare|list|"
    r"generate|draft|outline|recommend|advise|teach|"
    r"give|show|help|design|plan|create|make|build|bypass|evade|spoof)\b)",  # imperatives
    re.I,
)

HAZARD_RE = re.compile(
    r"(?i)\b("
    # physical harm
    r"weapon|gun|knife|bomb|explosive|kill|murder|assault|arson|molotov|grenade|shoot|stab|"
    # bio/chem
    r"poison|toxin|ricin|fentanyl|meth|cocaine|heroin|"
    # cyber/crime
    r"ddos|malware|ransomware|backdoor|exploit|phishing|hack|crack|keylogger|botnet|spoof|bypass|jailbreak|"
    # extremism/terror
    r"terror|extremis|isis|nazi|kkk|bioweapon|"
    # evasion/illicit procurement
    r"undetectable|untraceable|serial\s*number\s*remov|silencer|ghost\s*gun|launder|forg(?:e|ery)|counterfeit"
    r")\b"
)

# Exclude benign-sense uses from hazard anchors (these belong in neutrals)
BENIGN_CTX_RE = re.compile(
    r"(photo\s+shoot|shoot(?:ing)?\s+(?:scene|film|movie|schedule)|shooting\s+star|"
    r"bombastic|photo\s*bomb|seed\s+bomb|attack\s+(?:surface|rate|vector)|"
    r"toxic(?:ity)?\s+(?:policy|classifier|comment[s]?)|"
    r"bullet\s+points|explosive\s+growth|test\s+server|dev\s+server|cloud|sandbox|"
    r"practice|dummy|harmless|mock|example|metaphor|idiom|slang)",
    re.I
)

# Hazard subtype buckets
HAZARD_GROUPS = [
    ("physical",  re.compile(
        r"\b(weapon|gun|knife|bomb|explosive|kill|murder|assault|arson|molotov|grenade|shoot|stab)\b", re.I)),
    ("bio_chem",  re.compile(r"\b(poison|toxin|ricin|fentanyl|meth|cocaine|heroin)\b", re.I)),
    ("cyber",     re.compile(
        r"\b(ddos|malware|ransomware|backdoor|exploit|phishing|hack|crack|keylogger|botnet|spoof|bypass|jailbreak)\b", re.I)),
    ("extremism", re.compile(r"\b(terror|extremis|isis|nazi|kkk|bioweapon)\b", re.I)),
    ("evasion",   re.compile(r"\b(undetectable|untraceable|serial\s*number\s*remov|silencer|ghost\s*gun|launder|forg(?:e|ery)|counterfeit)\b", re.I)),
]


def hazard_bucket(text_low: str) -> Optional[str]:
    for name, rx in HAZARD_GROUPS:
        if rx.search(text_low):
            return name
    return None


TEXT_FIELDS = ["behavior", "prompt", "instruction",
               "question", "input", "request", "user", "content", "text"]


def norm_strict(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def choose_text(ex: dict) -> Tuple[Optional[str], Optional[str]]:
    for k in TEXT_FIELDS:
        v = ex.get(k)
        if isinstance(v, str) and len(v.strip()) >= 8:
            return v.strip(), k
    for k, v in ex.items():
        if isinstance(v, str) and len(v.strip()) >= 12 and not re.search(r"(answer|response|output)", k, re.I):
            return v.strip(), k
    return None, None


def discover_split(ds_name, config=None, hf_token=None, pref=("harmful", "benign", "train", "training", "default", "all")):
    try:
        splits = get_dataset_split_names(ds_name, config_name=config, token=hf_token)
        for p in pref:
            if p in splits:
                return p
        return splits[0]
    except Exception:
        return "train"


def load_any(ds_name, config=None, split=None, hf_token=None, trust_remote_code=False):
    if split is None:
        split = "train"
    kw = dict(split=split, trust_remote_code=trust_remote_code)
    if hf_token:
        try:
            kw["token"] = hf_token
        except TypeError:
            kw["use_auth_token"] = hf_token
    return load_dataset(ds_name, config, **kw) if config else load_dataset(ds_name, **kw)

# -------------------------
# Keep predicates
# -------------------------


def base_hygiene(t: str, min_len=12, max_len=200) -> bool:
    if not t or len(t) < min_len or len(t) > max_len:
        return False
    noisy = sum(1 for ch in t if ord(ch) < 9 or ord(ch) > 126)
    if noisy > 0.05 * len(t):
        return False
    if re.match(r"^\s*(answer|assistant|response)\s*[:\-]", t, re.I):
        return False
    return True


def keep_hazard_anchor(text: str) -> bool:
    if not base_hygiene(text):
        return False
    low = text.lower().strip()
    if not INTENT_RE.search(low):
        return False
    if not HAZARD_RE.search(low):
        return False
    if STEP_RE.search(low):
        return False
    if BENIGN_CTX_RE.search(low):
        return False
    return True


def keep_neutral_control(text: str) -> bool:
    if not base_hygiene(text):
        return False
    low = text.lower().strip()
    if not INTENT_RE.search(low):
        return False
    if HAZARD_RE.search(low):
        return False
    if STEP_RE.search(low):
        return False
    # Benign-sense uses are fine for neutral controls
    return True

# -------------------------
# Harvest helpers
# -------------------------


def harvest(
    ds_name: str,
    *,
    config: Optional[str] = None,
    split: Optional[str] = None,
    cap: int = 300,
    hf_token: Optional[str] = None,
    trust_remote_code: bool = False,
    diag_prefix: str = "",
    keep_fn: Callable[[str], bool] = keep_hazard_anchor,
    source_tag: Optional[str] = None,
):
    if split is None:
        split = discover_split(ds_name, config, hf_token)
    try:
        ds = load_any(ds_name, config=config, split=split,
                      hf_token=hf_token, trust_remote_code=trust_remote_code)
    except Exception as e:
        print(f"[warn]{diag_prefix} load-failed {ds_name}/{config or ''}/{split}: {e}", file=sys.stderr)
        return [], {"seen": 0, "kept": 0, "by_key": {}, "split": split}

    try:
        ds = ds.shuffle(seed=SEED)
    except Exception:
        pass

    rows = []
    chosen_keys = defaultdict(int)
    seen = 0
    for ex in ds:
        seen += 1
        t, k = choose_text(ex)
        if not t:
            continue
        if keep_fn(t):
            rows.append(t.strip())
            chosen_keys[k] += 1
            if len(rows) >= cap:
                break
    return rows, {"seen": seen, "kept": len(rows), "by_key": dict(chosen_keys), "split": split}

# -------------------------
# Main (proposal-aligned H)
# -------------------------


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-token", default=os.getenv("HUGGINGFACE_HUB_TOKEN", ""))
    ap.add_argument("--out", default="data/anchors/anchors_500.jsonl")
    ap.add_argument("--cap", type=int, default=500, help="Total rows in H (hazard + neutral)")
    ap.add_argument("--hazard-ratio", type=float, default=0.75,
                    help="Proportion of hazard rows within H")
    ap.add_argument("--per_source_quota", type=int, default=350,
                    help="Max rows pulled per source before dedup/mix")
    ap.add_argument("--no-bucket-balance", action="store_true", help="Disable per-subtype quotas")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--use-dolly-neutral", action="store_true",
                    help="Top up neutral controls from databricks/databricks-dolly-15k if needed")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    hazard_target = int(round(args.cap * args.hazard_ratio))
    neutral_target = args.cap - hazard_target

    # ---- Hazard sources (JBB harmful + AdvBench) ----
    jbb_h_split = "harmful"  # explicit split for clarity
    jbb_h_rows, jbb_h_diag = harvest(
        "JailbreakBench/JBB-Behaviors", config="behaviors", split=jbb_h_split,
        cap=min(args.per_source_quota, hazard_target),
        hf_token=args.hf_token, trust_remote_code=args.trust_remote_code,
        diag_prefix="[JBB/H] ", keep_fn=keep_hazard_anchor, source_tag="JBB-Behaviors:harmful"
    )
    print(
        f"[diag] JBB-H: split={jbb_h_diag['split']} seen={jbb_h_diag['seen']} kept={jbb_h_diag['kept']} keys={jbb_h_diag['by_key']}")

    adv_rows, adv_diag = harvest(
        "walledai/AdvBench", config=None, split=None,
        cap=min(args.per_source_quota, max(0, hazard_target - len(jbb_h_rows))),
        hf_token=args.hf_token, trust_remote_code=args.trust_remote_code,
        diag_prefix="[ADV] ", keep_fn=keep_hazard_anchor, source_tag="AdvBench"
    )
    if adv_diag["seen"] == 0 and adv_diag["kept"] == 0:
        print("[diag] AdvBench likely gated/unavailable; continuing with JBB harmful only.")

    hazard_pool = [("JBB+AdvBench", t) for t in (jbb_h_rows + adv_rows)]

    # ---- Neutral-control sources (JBB benign + optional Dolly) ----
    jbb_b_split = "benign"
    jbb_b_rows, jbb_b_diag = harvest(
        "JailbreakBench/JBB-Behaviors", config="behaviors", split=jbb_b_split,
        cap=min(args.per_source_quota, neutral_target),
        hf_token=args.hf_token, trust_remote_code=args.trust_remote_code,
        diag_prefix="[JBB/B] ", keep_fn=keep_neutral_control, source_tag="JBB-Behaviors:benign"
    )
    print(
        f"[diag] JBB-B: split={jbb_b_diag['split']} seen={jbb_b_diag['seen']} kept={jbb_b_diag['kept']} keys={jbb_b_diag['by_key']}")

    neutral_pool = [("JBB-benign", t) for t in jbb_b_rows]

    if args.use_dolly_neutral:
        dolly_rows, dolly_diag = harvest(
            "databricks/databricks-dolly-15k", config=None, split="train",
            cap=min(args.per_source_quota, max(0, neutral_target - len(neutral_pool))),
            hf_token=args.hf_token, trust_remote_code=args.trust_remote_code,
            diag_prefix="[DOLLY] ", keep_fn=keep_neutral_control, source_tag="dolly-15k"
        )
        print(
            f"[diag] Dolly: split={dolly_diag['split']} seen={dolly_diag['seen']} kept={dolly_diag['kept']} keys={dolly_diag['by_key']}")
        neutral_pool += [("dolly-15k", t) for t in dolly_rows]

    # ---- Dedup + subtype quotas + assembly ----
    bucket_quota = defaultdict(lambda: 10**9)
    if not args.no_bucket_balance:
        bucket_quota.update({"physical": 180, "cyber": 180, "bio_chem": 80,
                            "extremism": 80, "evasion": 80})
    bucket_used = defaultdict(int)

    out, seen_hash = [], set()

    # hazards first
    random.shuffle(hazard_pool)
    for src, t in hazard_pool:
        if sum(1 for r in out if r["label"] == "hazard") >= hazard_target:
            break
        k = hashlib.md5(norm_strict(t).encode("utf-8")).hexdigest()
        if k in seen_hash:
            continue
        if not keep_hazard_anchor(t):
            continue
        low = t.lower()
        bname = hazard_bucket(low) or None
        if bname and bucket_used[bname] >= bucket_quota[bname]:
            continue
        seen_hash.add(k)
        out.append({"text": t, "label": "hazard", "source": src, "meta": {"subtype": bname}})
        if bname:
            bucket_used[bname] += 1

    # then neutrals
    random.shuffle(neutral_pool)
    for src, t in neutral_pool:
        if sum(1 for r in out if r["label"] == "neutral") >= neutral_target:
            break
        k = hashlib.md5(norm_strict(t).encode("utf-8")).hexdigest()
        if k in seen_hash:
            continue
        if not keep_neutral_control(t):
            continue
        seen_hash.add(k)
        out.append({"text": t, "label": "neutral", "source": src, "meta": {}})

    # Backfill if short (preserves filters; just increases supply)
    def backfill(pool, label, target, keep_fn):
        added = 0
        if sum(1 for r in out if r["label"] == label) >= target:
            return 0
        for src, t in pool:
            if sum(1 for r in out if r["label"] == label) >= target:
                break
            k = hashlib.md5(norm_strict(t).encode("utf-8")).hexdigest()
            if k in seen_hash or not keep_fn(t):
                continue
            if label == "hazard":
                low = t.lower()
                bname = hazard_bucket(low) or None
                if bname and bucket_used[bname] >= bucket_quota[bname]:
                    continue
                out.append({"text": t, "label": "hazard", "source": src, "meta": {"subtype": bname}})
                if bname:
                    bucket_used[bname] += 1
            else:
                out.append({"text": t, "label": "neutral", "source": src, "meta": {}})
            seen_hash.add(k)
            added += 1
        return added

    backfill(hazard_pool, "hazard", hazard_target, keep_hazard_anchor)
    backfill(neutral_pool, "neutral", neutral_target, keep_neutral_control)

    random.shuffle(out)

    # Write
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out[:args.cap]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Diagnostics
    counts = Counter([r["label"] for r in out[:args.cap]])
    subtypes = Counter([r["meta"].get("subtype") for r in out[:args.cap] if r["label"] == "hazard"])
    print(
        f"[mix] labels: {dict(counts)}  (target hazard={int(hazard_target)}, neutral={int(neutral_target)})")
    print(f"[mix] hazard subtypes: {dict(subtypes)}")
    print(f"[done] anchors (H): {len(out[:args.cap])} → {args.out}")


if __name__ == "__main__":
    main()
