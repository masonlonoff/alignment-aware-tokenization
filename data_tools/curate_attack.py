# data_tools/curate_attack.py
import os
import json
import random
import argparse
from typing import Optional, Dict, Any

from datasets import load_dataset, get_dataset_split_names

SEED = 9172
random.seed(SEED)

# Prefer Behaviors keys first, but keep general fallbacks
TEXT_FIELDS = ["text", "prompt", "instruction", "query", "request", "Behavior", "behavior"]


def pick_text(ex: Dict[str, Any]) -> Optional[str]:
    # 1) Prefer known fields
    for k in TEXT_FIELDS:
        v = ex.get(k)
        if isinstance(v, str):
            s = v.strip()
            if s:
                return s

    # 2) Fallback: any other non-empty string with length ≥ 10
    for v in ex.values():
        if isinstance(v, str):
            s = v.strip()
            if len(s) >= 10:
                return s

    return None


def discover_split(ds: str, config: Optional[str], token: Optional[str]) -> str:
    try:
        splits = get_dataset_split_names(ds, config_name=config, token=token)
        if not splits:
            return "train"

        lower_to_orig = {s.lower(): s for s in splits if isinstance(s, str)}

        for cand in ("train", "training", "default", "all"):
            if cand in lower_to_orig:
                return lower_to_orig[cand]

        return splits[0]
    except Exception:
        return "train"


def load_ds(ds: str, config: Optional[str], split: str, token: Optional[str], trust_remote_code: bool):
    kwargs = {"split": split, "trust_remote_code": trust_remote_code}

    # Positional args per HF signature: load_dataset(path, [name], **kwargs)
    args = [ds]
    if config is not None:
        args.append(config)

    if token:
        try:
            # Newer datasets use `token=...`
            return load_dataset(*args, token=token, **kwargs)
        except TypeError:
            # Older datasets use `use_auth_token=...`
            return load_dataset(*args, use_auth_token=token, **kwargs)
    else:
        return load_dataset(*args, **kwargs)


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
    # Resolve split
    if split is None:
        try:
            split = discover_split(ds, config=config, token=token)
        except Exception as e:
            # Fallback per rules if discovery fails
            print(f"[warn] split discovery failed for {ds} (config={config}): {e}. Using 'train'.")
            split = "train"

    # Load dataset
    try:
        d = load_ds(
            ds,
            config=config,
            split=split,
            token=token,
            trust_remote_code=trust_remote_code,
        )
    except Exception as e:
        # Generic warning (caller/CLI can print the special AdvBench diag if desired)
        print(f"[warn] load failed for {ds} (config={config}, split={split}): {e}")
        return 0

    # Shuffle if supported
    try:
        try:
            _ = SEED  # type: ignore[name-defined]
        except NameError:
            d = d.shuffle(seed=SEED)  # some datasets may not support shuffle
    except Exception:
        pass  # proceed unshuffled

    # Ensure output directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Build source string
    if source_tag:
        source_str = source_tag
    else:
        cfg_part = f":{config}" if config else ""
        source_str = f"{ds}{cfg_part}/{split}"

    written = 0
    # Write JSONL, newline-terminated, UTF-8
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            # Iterate dataset examples
            for ex in d:
                if written >= n:
                    break
                try:
                    text = pick_text(ex)
                except Exception as e:
                    # If pick_text crashes on a weird row, skip it
                    # (Keep it quiet; continue harvesting.)
                    continue

                if not text:
                    continue
                text = text.strip()
                if not text:
                    continue

                rec = {"text": text, "label": label, "source": source_str}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
    except Exception as e:
        print(f"[warn] writing to {out_path} failed: {e}")
        return 0

    print(f"[done] {ds} → {out_path} ({written} lines)")
    return written


def main():
    """
    Wire up CLI and run selected jobs.
    """
    p = argparse.ArgumentParser(description="Harvest attack prompts to JSONL.")
    # Auth / load flags
    p.add_argument("--hf-token", dest="hf_token", default=None)
    p.add_argument("--trust-remote-code", action="store_true", dest="trust_remote_code")

    # Limits & outputs
    p.add_argument("--n-adv", type=int, default=1000)
    p.add_argument("--n-jbv", type=int, default=500)
    p.add_argument("--out-adv", default="data/eval/attack_1000.jsonl")
    p.add_argument("--out-jbv", default="data/eval/attack_extra_500.jsonl")

    # Run mode
    p.add_argument("--run", choices=["both", "adv", "jbv"], default="both")

    # AdvBench dataset knobs
    p.add_argument("--adv-ds", default="walledai/AdvBench")
    p.add_argument("--adv-config", default=None)
    p.add_argument("--adv-split", default=None)

    # JBV dataset knobs
    p.add_argument("--jbv-ds",    default="JailbreakV-28K/JailBreakV-28k")
    p.add_argument("--jbv-config", default="JailBreakV_28K")   # Behaviors usually has no config
    p.add_argument("--jbv-split",  default=None)   # let discover_split pick (train)

    args = p.parse_args()

    def should(run_tag: str, n: int) -> bool:
        return (args.run in ("both", run_tag)) and (n > 0)

    # AdvBench
    if should("adv", args.n_adv):
        written = dump_generic(
            ds=args.adv_ds,
            out_path=args.out_adv,
            n=args.n_adv,
            config=args.adv_config,
            split=args.adv_split,
            token=args.hf_token,
            trust_remote_code=args.trust_remote_code,
            label="attack",
            source_tag=None,
        )
        if written == 0:
            print("[diag] AdvBench: skipped or empty (gated or load failed).")

    # JBV
    if should("jbv", args.n_jbv):
        _ = dump_generic(
            ds=args.jbv_ds,
            out_path=args.out_jbv,
            n=args.n_jbv,
            config=args.jbv_config,
            split=args.jbv_split,
            token=args.hf_token,
            trust_remote_code=args.trust_remote_code,
            label="attack",
            source_tag=None,
        )

    # Exit 0 explicitly (nice for shell usage)
    return 0


if __name__ == "__main__":
    main()