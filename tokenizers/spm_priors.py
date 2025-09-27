import argparse
import json
import sentencepiece as spm
import os
import tempfile


def prior_score(token, hazard_stems, benign_words):
    # Very simple proxy: overlap with stems and substring count in benign set
    s = sum(token in w for w in benign_words)
    o = 1 if token in hazard_stems else 0
    return 0.3*o + 0.5*(s > 2)


def main(args):
    corpus = args.corpus  # path to txt
    model_prefix = args.out.replace(".model", "")
    spm.SentencePieceTrainer.Train(
        input=corpus, model_prefix=model_prefix, vocab_size=args.vocab_size,
        user_defined_symbols=",".join(args.pin.split(",")) if args.pin else "")
    print("Saved SPM with hazard pinning to", args.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", default="tokenizers/spm_hazard.model")
    p.add_argument("--vocab_size", type=int, default=50000)
    p.add_argument("--pin", type=str, default="")
    main(p.parse_args())
