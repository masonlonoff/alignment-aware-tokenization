import argparse, json, yaml, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
import torch, os

def pooled_hidden(model, tok, texts, layer):
    with torch.no_grad():
        outs = []
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            h = model(**ids, output_hidden_states=True).hidden_states[layer][0]  # [T, H]
            outs.append(h.mean(dim=0).cpu().numpy())
        return np.stack(outs)

def main(args):
    cfg = yaml.safe_load(open(args.config))
    model = AutoModel.from_pretrained(cfg["model_name"], torch_dtype=torch.bfloat16).eval().to("cuda")
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    H = [json.loads(l)["text"] for l in open(cfg["data"]["anchors"])]
    N = [json.loads(l)["text"] for l in open(cfg["data"]["neutrals"])]
    X = np.vstack([pooled_hidden(model, tok, H, cfg["drift"]["layer"]),
                   pooled_hidden(model, tok, N, cfg["drift"]["layer"])])
    y = np.array([1]*len(H) + [0]*len(N))
    clf = LogisticRegression(max_iter=200).fit(X, y)
    v = clf.coef_.astype(np.float32)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    np.save(args.save, v / np.linalg.norm(v))
    print(f"Saved concept vector to {args.save}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--save", required=True)
    main(p.parse_args())
