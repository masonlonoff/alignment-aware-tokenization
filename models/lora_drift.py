import argparse
import json
import yaml
import numpy as np
import torch
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from torch.utils.data import islice


def stream_text(ds_name, max_items=20000):
    ds = load_dataset(ds_name, split="train", streaming=True)
    for i, ex in enumerate(ds):
        if i >= max_items:
            break
        yield ex["text"]


def drift_penalty(model, tok, v, texts, layer, margin):
    v = torch.tensor(v, dtype=model.dtype, device=model.device)
    with torch.no_grad():
        scores = []
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=256).to(model.device)
            h = model.transformer(**ids, output_hidden_states=True).hidden_states[layer][0].mean(0)
            s = torch.dot(h, v.squeeze())
            scores.append(torch.maximum(
                s - margin, torch.tensor(0, dtype=s.dtype, device=s.device))**2)
        return torch.stack(scores).mean()


def main(args):
    cfg = yaml.safe_load(open(args.config))
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=torch.bfloat16).to("cuda")
    lcfg = LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"],
                      target_modules=cfg["lora"]["target_modules"])
    model = get_peft_model(model, lcfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    sch = get_cosine_schedule_with_warmup(opt, cfg["train"]["warmup_steps"], cfg["train"]["steps"])

    neutrals = [json.loads(l)["text"] for l in open(cfg["data"]["neutrals"])]
    v = np.load("probes/v_layer.pt.npy") if os.path.exists(
        "probes/v_layer.pt.npy") else np.load("probes/v_layer.pt", allow_pickle=True)
    stream = stream_text(cfg["data"]["unlabeled_stream"], max_items=cfg["eval"]["u_dev_sample"])
    model.train()
    for step, text in enumerate(stream):
        if step >= cfg["train"]["steps"]:
            break
        ids = tok(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        out = model(**ids, labels=ids["input_ids"])
        lm_loss = out.loss
        drift = drift_penalty(model, tok, v, neutrals[:8],
                              cfg["drift"]["layer"], cfg["drift"]["margin"])
        loss = lm_loss + cfg["drift"]["lambda"] * drift
        loss.backward()
        opt.step()
        sch.step()
        model.zero_grad()
        if step % 50 == 0:
            print(f"step {step} | lm {lm_loss.item():.3f} | drift {drift.item():.3f}")
    os.makedirs(args.save, exist_ok=True)
    model.save_pretrained(args.save)
    tok.save_pretrained(args.save)
    print("Saved LoRA adapter to", args.save)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--save", required=True)
    main(p.parse_args())
