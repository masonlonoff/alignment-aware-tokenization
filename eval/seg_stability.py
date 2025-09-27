import argparse
import random
from transformers import AutoTokenizer


def boundary_flip_rate(tok, texts):
    def seg(s): return tok.tokenize(s)
    flips, cnt = 0, 0
    for t in texts:
        t2 = t + "!" if random.random() < 0.5 else t.replace("a", "a ")
        a, b = seg(t), seg(t2)
        flips += int(a != b)
        cnt += 1
    return flips / max(cnt, 1)


def main(args):
    tok = AutoTokenizer.from_pretrained(
        args.base_tokenizer if args.base_tokenizer.endswith("/") else args.base_tokenizer)
    texts = ["We booked a photo shoot.", "The keynote had a bombastic tone."]*100
    print("Boundary flip rate:", boundary_flip_rate(tok, texts))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base_tokenizer", required=True)
    main(p.parse_args())
