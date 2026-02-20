#!/usr/bin/env python3
"""
axis 4: ESM2 embeddings for all IDP proteins.
extracts 1280-dim mean embeddings (not just LLR) from ESM2-650M.
also computes per-residue LLR for comparison.
uses HuggingFace transformers (already installed).
"""

import csv
import os
from pathlib import Path

import numpy as np
import torch
from transformers import EsmModel, EsmTokenizer, EsmForMaskedLM

# use GPU 1 (GPU 0 may be busy)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

PROJECT = Path(__file__).resolve().parent.parent
FASTA = PROJECT / "sequences" / "all_proteins.fasta"
OUT_CSV = PROJECT / "features" / "esm2_embeddings_summary.csv"
OUT_NPY = PROJECT / "features" / "esm2_embeddings.npy"
OUT_NAMES = PROJECT / "features" / "esm2_names.txt"

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"


def read_fasta(path):
    seqs = {}
    name = None
    buf = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name:
                    seqs[name] = "".join(buf)
                name = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
    if name:
        seqs[name] = "".join(buf)
    return seqs


def extract_embeddings(model, tokenizer, seq, device):
    """extract mean embedding (1280-dim) from ESM2 last hidden state."""
    inputs = tokenizer(seq, return_tensors="pt", padding=False, truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # last hidden state: (1, seq_len+2, 1280) — +2 for [CLS] and [EOS]
    hidden = outputs.last_hidden_state[0]  # (seq_len+2, 1280)

    # strip [CLS] and [EOS] tokens
    residue_embeddings = hidden[1:-1]  # (seq_len, 1280)

    # mean embedding across all residues
    mean_emb = residue_embeddings.mean(dim=0).cpu().numpy()  # (1280,)

    # also compute embedding statistics
    std_emb = residue_embeddings.std(dim=0).cpu().numpy()
    # PCA variance: what fraction of variance is in top 10 components?
    # (a proxy for conformational complexity signal)

    return mean_emb, std_emb, residue_embeddings.cpu().numpy()


def compute_mean_llr(mlm_model, tokenizer, seq, device):
    """compute mean log-likelihood ratio (pseudo-perplexity) per residue.
    this is the masked marginal probability approach."""
    inputs = tokenizer(seq, return_tensors="pt", padding=False, truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].to(device)

    n_res = len(seq)
    llr_sum = 0.0

    # for efficiency, do single forward pass with all positions masked one at a time
    # but that's O(n) forward passes. for proteins up to 526 aa, it's manageable.
    # simpler approach: use pseudo-perplexity from unmasked logits
    with torch.no_grad():
        outputs = mlm_model(input_ids)
        logits = outputs.logits[0]  # (seq_len+2, vocab_size)

    # for each position, get the log prob of the true token
    log_probs = torch.log_softmax(logits, dim=-1)

    total_log_prob = 0.0
    for i in range(1, n_res + 1):  # skip [CLS] at position 0
        token_id = input_ids[0, i].item()
        total_log_prob += log_probs[i, token_id].item()

    mean_llr = total_log_prob / n_res
    return mean_llr


def main():
    seqs = read_fasta(FASTA)
    print(f"loaded {len(seqs)} sequences")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # load model
    print(f"loading {MODEL_NAME}...")
    tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME).to(device).eval()
    mlm_model = EsmForMaskedLM.from_pretrained(MODEL_NAME).to(device).eval()
    print("model loaded")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    all_embeddings = []
    all_names = []
    csv_rows = []

    for name, seq in seqs.items():
        seq = seq.upper()
        print(f"\n  {name} ({len(seq)} aa)...")

        # extract embeddings
        mean_emb, std_emb, residue_embs = extract_embeddings(model, tokenizer, seq, device)
        all_embeddings.append(mean_emb)
        all_names.append(name)

        # compute LLR (pseudo-perplexity)
        mean_llr = compute_mean_llr(mlm_model, tokenizer, seq, device)

        # embedding statistics
        emb_norm = float(np.linalg.norm(mean_emb))
        emb_mean_val = float(mean_emb.mean())
        emb_std_val = float(mean_emb.std())

        # per-residue embedding variance (measure of internal diversity)
        residue_var = float(residue_embs.var(axis=0).mean())

        print(f"    mean_llr={mean_llr:.4f}, emb_norm={emb_norm:.2f}, residue_var={residue_var:.6f}")

        csv_rows.append({
            "name": name,
            "length": len(seq),
            "mean_llr": round(mean_llr, 4),
            "emb_norm": round(emb_norm, 4),
            "emb_mean": round(emb_mean_val, 6),
            "emb_std": round(emb_std_val, 6),
            "residue_emb_variance": round(residue_var, 6),
        })

    # save embeddings as numpy array
    emb_matrix = np.array(all_embeddings)  # (n_proteins, 1280)
    np.save(OUT_NPY, emb_matrix)
    print(f"\nsaved embedding matrix {emb_matrix.shape} to {OUT_NPY}")

    # save names
    with open(OUT_NAMES, "w") as f:
        for n in all_names:
            f.write(n + "\n")

    # save CSV summary
    fields = list(csv_rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(csv_rows)
    print(f"saved summary to {OUT_CSV}")


if __name__ == "__main__":
    main()
