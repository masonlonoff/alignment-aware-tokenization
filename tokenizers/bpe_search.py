import argparse
import json
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from eval.eval_perplexity import perplexity
from eval.eval_drift import drift_score
from tokenizers.risky_merges import find_risky_merges


def edit_merges(base_tok):
    # Placeholder: load merges, remove a few risky ones (you'll implement properly)
    merges = getattr(base_tok, "merges", None)
    return base_tok  # TODO: write edited tokenizer JSON and reload


def score(tok, model, u_dev, v, neutrals, alpha=0.7, beta=0.1, ppl0=1.0, drift0=1.0, tpc0=1.0):
    ppl, tpc = perplexity(model, tok, u_dev), 1.0  # TODO: compute tokens/char
    drift = drift_score(model, tok, neutrals, v)
    return ppl/ppl0 + alpha*drift/drift0 + beta*tpc/tpc0, ppl, drift, tpc


def main(args):
    base_tok = AutoTokenizer.from_pretrained(args.base_tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.base_tokenizer)
    # Load dev and neutrals, concept vector v…
    # Rounds of propose→score→accept
    best_tok = base_tok
    for r in range(args.rounds):
        cand_tok = edit_merges(best_tok)
        # compute J; keep best
    best_tok.save_pretrained(args.out)
    print("Saved searched BPE tokenizer to", args.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base_tokenizer", required=True)
    p.add_argument("--anchors", required=True)
    p.add_argument("--neutrals", required=True)
    p.add_argument("--u_dev_size", type=int, default=20000)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--out", required=True)
    main(p.parse_args())
