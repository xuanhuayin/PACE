from pace.paths import PROJECT_ROOT
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = PROJECT_ROOT


SNAP = os.environ.get("LLADA_SNAP", "GSAI-ML/LLaDA-8B-Instruct")
MASK_ID = 126336


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num[i, : remainder[i]] += 1
    return num


@torch.no_grad()
def generate_capture(model, prompt, gen_length=256, steps=256, block_length=16, topk=2048):
    """LLaDA generate (temperature 0, low-confidence remasking, cfg 0) + commit-time logits."""
    B, Lp = prompt.shape
    x = torch.full((B, Lp + gen_length), MASK_ID, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt
    commit_lp = [None] * gen_length
    num_blocks = gen_length // block_length
    steps_b = steps // num_blocks
    for nb in range(num_blocks):
        lo, hi = (Lp + nb * block_length, Lp + (nb + 1) * block_length)
        block_mask = x[:, lo:hi] == MASK_ID
        ntt = get_num_transfer_tokens(block_mask, steps_b)
        for i in range(steps_b):
            mask_index = x == MASK_ID
            logits = model(x).logits
            x0 = torch.argmax(logits, dim=-1)
            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            x0_p[:, hi:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            conf = torch.where(mask_index, x0_p, -np.inf)
            transfer = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(B):
                _, sel = torch.topk(conf[j], k=int(ntt[j, i]))
                transfer[j, sel] = True
            lp = F.log_softmax(logits[0, :, :126349].float(), dim=-1)
            for pos in torch.nonzero(transfer[0]).flatten().tolist():
                row = lp[pos]
                tv, ti = torch.topk(row, topk)
                commit_lp[pos - Lp] = (
                    ti.cpu().numpy().astype(np.int32),
                    tv.cpu().numpy().astype(np.float16),
                    float(row[126081]),
                    float(row.max()),
                )
            x[transfer] = x0[transfer]
    return (x[0, Lp:].cpu().tolist(), commit_lp)
