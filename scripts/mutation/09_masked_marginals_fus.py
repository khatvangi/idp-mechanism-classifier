#!/usr/bin/env python3
"""
09_masked_marginals_fus.py — masked marginal ESM2 inference on FUS

GO/NO-GO GATE:
  AUROC > 0.70 → blind spot is approximation artifact, STOP and reassess
  AUROC < 0.55 → blind spot is real, proceed with paper

method:
  for each of 526 FUS positions, mask that position and run ESM2 forward pass.
  compute true P(aa | context without position) from the masked logits.
  masked_LLR = log P(ref_aa | masked_context) - log P(alt_aa | masked_context)
  compare with unmasked LLR from script 06.

runtime: ~5 min on Titan RTX (526 forward passes)
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, EsmForMaskedLM
from sklearn.metrics import roc_auc_score
from pathlib import Path
import time

BASE = Path("/storage/kiran-stuff/idp-mechanism-classifier")
DATA = BASE / "data"

# ── load ESM2 ────────────────────────────────────────────────────────────────
print("loading ESM2-650M...")
model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = EsmForMaskedLM.from_pretrained(model_name)
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"  device: {device}")

# ── load FUS sequence ─────────────────────────────────────────────────────────
sequences = {}
current_gene = None
current_seq = []
with open(DATA / "sequences/all_proteins.fasta") as f:
    for line in f:
        if line.startswith(">"):
            if current_gene:
                sequences[current_gene] = "".join(current_seq)
            # header format: >GENE|UNIPROT — extract gene name
            current_gene = line.strip().lstrip(">").split("|")[0]
            current_seq = []
        else:
            current_seq.append(line.strip())
    if current_gene:
        sequences[current_gene] = "".join(current_seq)

fus_seq = sequences["FUS"]
seq_len = len(fus_seq)
print(f"FUS sequence: {seq_len} aa")

# ── tokenize ──────────────────────────────────────────────────────────────────
inputs = tokenizer(fus_seq, return_tensors="pt", add_special_tokens=True)
input_ids = inputs["input_ids"].to(device)  # [1, seq_len+2] with CLS and EOS

# amino acid token mapping
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
aa_token_ids = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in AA_ORDER}
aa_to_idx = {aa: i for i, aa in enumerate(AA_ORDER)}
mask_token_id = tokenizer.mask_token_id

# ── masked marginal inference ─────────────────────────────────────────────────
# for each position: mask it, run forward, extract log P(aa) at that position
print(f"\nrunning masked marginal inference ({seq_len} positions)...")
masked_logprobs = np.zeros((seq_len, 20))  # [position, aa_index]
t0 = time.time()

with torch.no_grad():
    for i in range(seq_len):
        # mask position i (token index i+1 because of CLS token at index 0)
        masked_input = input_ids.clone()
        masked_input[0, i + 1] = mask_token_id

        # forward pass
        outputs = model(masked_input)
        logits = outputs.logits[0, i + 1, :]  # logits at masked position

        # log-softmax → log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # extract log P for each of 20 standard AAs
        for j, aa in enumerate(AA_ORDER):
            masked_logprobs[i, j] = log_probs[aa_token_ids[aa]].item()

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (seq_len - i - 1) / rate
            print(f"  {i+1}/{seq_len} ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

elapsed = time.time() - t0
print(f"masked inference complete in {elapsed:.1f}s ({elapsed/seq_len:.2f}s per position)")

# ── unmasked logits for comparison ────────────────────────────────────────────
print("\ncomputing unmasked logits...")
with torch.no_grad():
    outputs = model(input_ids)
    # logits at sequence positions (skip CLS at index 0)
    all_logits = outputs.logits[0, 1:seq_len + 1, :]  # [seq_len, vocab_size]
    unmasked_log_probs = F.log_softmax(all_logits, dim=-1)

    unmasked_logprobs = np.zeros((seq_len, 20))
    for j, aa in enumerate(AA_ORDER):
        unmasked_logprobs[:, j] = unmasked_log_probs[:, aa_token_ids[aa]].cpu().numpy()

# ── load FUS variants ─────────────────────────────────────────────────────────
# esm2_features.csv has the target column and original ESM2 LLR
df = pd.read_csv(DATA / "variants/esm2_features.csv")
fus = df[df["gene"] == "FUS"].copy()
print(f"\nFUS variants: {len(fus)} total, {(fus['target'] == 1).sum()} pathogenic, "
      f"{(fus['target'] == 0).sum()} benign/VUS")

# ── compute LLRs for each variant ────────────────────────────────────────────
masked_llrs = []
unmasked_llrs = []

for _, row in fus.iterrows():
    pos = int(row["position"]) - 1  # convert 1-indexed to 0-indexed
    ref_aa = row["ref_aa"]
    alt_aa = row["alt_aa"]

    if ref_aa not in aa_to_idx or alt_aa not in aa_to_idx:
        masked_llrs.append(np.nan)
        unmasked_llrs.append(np.nan)
        continue

    ref_idx = aa_to_idx[ref_aa]
    alt_idx = aa_to_idx[alt_aa]

    # masked LLR: log P(ref | masked context) - log P(alt | masked context)
    m_llr = masked_logprobs[pos, ref_idx] - masked_logprobs[pos, alt_idx]
    masked_llrs.append(m_llr)

    # unmasked LLR (recomputed fresh, should match script 06)
    u_llr = unmasked_logprobs[pos, ref_idx] - unmasked_logprobs[pos, alt_idx]
    unmasked_llrs.append(u_llr)

fus["masked_llr"] = masked_llrs
fus["unmasked_llr_recomputed"] = unmasked_llrs

# drop any NaN rows
valid = fus.dropna(subset=["masked_llr", "unmasked_llr_recomputed"])
print(f"valid variants for comparison: {len(valid)} (dropped {len(fus) - len(valid)} with unknown AAs)")

# ── THE GO/NO-GO GATE ────────────────────────────────────────────────────────
y = valid["target"].values
n_path = int(y.sum())
n_ben = int((1 - y).sum())

print(f"\n{'='*65}")
print(f"  FUS MASKED MARGINALS — GO/NO-GO GATE")
print(f"{'='*65}")
print(f"  n={len(y)}, pathogenic={n_path}, benign/VUS={n_ben}")
print()

auroc_masked = roc_auc_score(y, valid["masked_llr"])
auroc_unmasked_new = roc_auc_score(y, valid["unmasked_llr_recomputed"])
auroc_original = roc_auc_score(y, valid["esm2_llr"])

print(f"  masked marginal LLR AUROC:     {auroc_masked:.4f}")
print(f"  unmasked LLR (recomputed):     {auroc_unmasked_new:.4f}")
print(f"  unmasked LLR (from script 06): {auroc_original:.4f}")
print()

if auroc_masked > 0.70:
    print("  >>> FINDING IS ARTIFACT")
    print("  >>> masked marginals fix the blind spot")
    print("  >>> STOP: reassess the entire paper premise")
elif auroc_masked < 0.55:
    print("  >>> BLIND SPOT IS REAL")
    print("  >>> masked marginals do NOT rescue FUS prediction")
    print("  >>> PROCEED with paper")
else:
    print("  >>> AMBIGUOUS")
    print("  >>> masked marginals partially improve but don't fix")
    print("  >>> further analysis needed")

print(f"{'='*65}")

# ── per-region analysis ───────────────────────────────────────────────────────
# NLS: positions 501-526 (1-indexed), PY-NLS for transportin-1 binding
nls_mask = (valid["position"] >= 501) & (valid["position"] <= 526)
n_nls = nls_mask.sum()
n_nls_path = (valid.loc[nls_mask, "target"] == 1).sum()

print(f"\n--- per-region analysis ---")
print(f"NLS variants: {n_nls} (pathogenic: {n_nls_path})")
print(f"non-NLS variants: {(~nls_mask).sum()} (pathogenic: {(valid.loc[~nls_mask, 'target'] == 1).sum()})")

if n_nls_path > 0 and (n_nls - n_nls_path) > 0:
    auroc_nls_masked = roc_auc_score(
        valid.loc[nls_mask, "target"], valid.loc[nls_mask, "masked_llr"])
    auroc_nls_unmasked = roc_auc_score(
        valid.loc[nls_mask, "target"], valid.loc[nls_mask, "unmasked_llr_recomputed"])
    print(f"  within-NLS masked LLR AUROC:   {auroc_nls_masked:.4f}")
    print(f"  within-NLS unmasked LLR AUROC: {auroc_nls_unmasked:.4f}")

# ── LLR statistics by pathogenicity ──────────────────────────────────────────
print(f"\n--- LLR statistics ---")
for label, name in [(1, "pathogenic"), (0, "benign/VUS")]:
    subset = valid[valid["target"] == label]
    print(f"\n  {name} (n={len(subset)}):")
    print(f"    masked LLR:   mean={subset['masked_llr'].mean():.2f}, "
          f"median={subset['masked_llr'].median():.2f}, "
          f"std={subset['masked_llr'].std():.2f}")
    print(f"    unmasked LLR: mean={subset['unmasked_llr_recomputed'].mean():.2f}, "
          f"median={subset['unmasked_llr_recomputed'].median():.2f}, "
          f"std={subset['unmasked_llr_recomputed'].std():.2f}")

# ── correlation ───────────────────────────────────────────────────────────────
corr = np.corrcoef(valid["masked_llr"], valid["unmasked_llr_recomputed"])[0, 1]
print(f"\ncorrelation(masked, unmasked) LLR: {corr:.3f}")

# ── delta: how much does masking change LLR? ─────────────────────────────────
valid_copy = valid.copy()
valid_copy["llr_delta"] = valid_copy["masked_llr"] - valid_copy["unmasked_llr_recomputed"]
print(f"\nLLR delta (masked - unmasked):")
print(f"  mean: {valid_copy['llr_delta'].mean():.3f}")
print(f"  std:  {valid_copy['llr_delta'].std():.3f}")
print(f"  pathogenic mean delta: {valid_copy.loc[valid_copy['target']==1, 'llr_delta'].mean():.3f}")
print(f"  benign/VUS mean delta: {valid_copy.loc[valid_copy['target']==0, 'llr_delta'].mean():.3f}")

# ── save ──────────────────────────────────────────────────────────────────────
fus.to_csv(DATA / "variants/fus_masked_marginals.csv", index=False)
print(f"\nsaved to {DATA / 'variants/fus_masked_marginals.csv'}")

# also save the full masked logprob matrix (for potential reuse)
np.save(DATA / "variants/fus_masked_logprobs.npy", masked_logprobs)
print(f"saved masked logprob matrix ({masked_logprobs.shape}) to fus_masked_logprobs.npy")
