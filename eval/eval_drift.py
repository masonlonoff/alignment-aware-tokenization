import argparse
import json
import yaml
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


def drift_score(model, tok, texts, v, layer=10):
    v = torch.tensor(v, dtype=torch.float32, device=model.device)
    scores = []
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=256).to(model.device)
            h = model(**ids, output_hidden_states=True).hidden_states[layer][0].mean(0)
            scores.append(torch.dot(h, v.squeeze()).item())
    import numpy as np
    return float(np.mean(scores))


def main(args):
    cfg = yaml.safe_load(open(args.config))
    v = np.load(args.probe, allow_pickle=True)
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModel.from_pretrained(cfg["model_name"]).to("cuda").eval()
    N = [json.loads(l)["text"] for l in open(cfg["data"]["neutrals"])]
    print("Neutral drift mean:", drift_score(model, tok, N, v, cfg["drift"]["layer"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--probe", required=True)
    main(p.parse_args())
