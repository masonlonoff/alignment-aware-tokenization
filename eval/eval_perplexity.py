import argparse
import yaml
import math
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def perplexity(model, tok, u_dev_texts):
    nll, tok_count = 0.0, 0
    for t in u_dev_texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            out = model(**ids, labels=ids["input_ids"])
        nll += out.loss.item() * ids["input_ids"].size(1)
        tok_count += ids["input_ids"].size(1)
    return math.exp(nll / max(tok_count, 1))


def main(args):
    cfg = yaml.safe_load(open(args.config))
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"]).to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    ds = load_dataset(cfg["data"]["unlabeled_stream"], split="train").shuffle(
        seed=42).select(range(cfg["eval"]["u_dev_sample"]))
    ppl = perplexity(model, tok, [x["text"] for x in ds])
    print(f"Perplexity: {ppl:.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    main(p.parse_args())
