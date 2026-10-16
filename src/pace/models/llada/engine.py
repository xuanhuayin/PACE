from pace.paths import PROJECT_ROOT
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = PROJECT_ROOT


from pace.models.llada.native import get_num_transfer_tokens, MASK_ID, SNAP

EOS = 126081


class LLaDAEngine:
    eos_id = EOS
    pad_id = EOS
    canvas_total = 1024

    def __init__(self):
        from transformers import AutoTokenizer, AutoModel

        self.tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)
        self.model = (
            AutoModel.from_pretrained(SNAP, trust_remote_code=True, torch_dtype=torch.bfloat16)
            .cuda()
            .eval()
        )
        self.torch = torch

    def prompt_ids(self, prompt):
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
        )
        return self.tok(text)["input_ids"]

    @torch.no_grad()
    def _generate(self, x, Lp, gen_start, gen_length, steps, block_length, seed):
        """LLaDA loop over masked positions in [gen_start, gen_start+gen_length); fixed tokens stay."""
        torch.manual_seed(seed)
        num_blocks = max(1, gen_length // block_length)
        steps_b = max(1, steps // num_blocks)
        for nb in range(num_blocks):
            lo, hi = (
                gen_start + nb * block_length,
                min(x.shape[1], gen_start + (nb + 1) * block_length),
            )
            block_mask = x[:, lo:hi] == MASK_ID
            if not block_mask.any():
                continue
            ntt = get_num_transfer_tokens(block_mask, steps_b)
            for i in range(steps_b):
                mask_index = x == MASK_ID
                if not mask_index[:, lo:hi].any():
                    break
                logits = self.model(x).logits
                x0 = torch.argmax(logits, dim=-1)
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                x0_p[:, hi:] = -np.inf
                x0 = torch.where(mask_index, x0, x)
                conf = torch.where(mask_index, x0_p, -np.inf)
                transfer = torch.zeros_like(x0, dtype=torch.bool)
                k = int(min(ntt[0, i], int(mask_index[0, lo:hi].sum())))
                if k <= 0:
                    continue
                _, sel = torch.topk(conf[0], k=k)
                transfer[0, sel] = True
                x[transfer] = x0[transfer]
        return x

    def generate_suffix(
        self, prompt, answer_prefix, seed, *, total=None, gen_length=256, block_length=16
    ):
        """Fixed answer prefix, then generate `gen_length` masked positions (official sampler settings)."""
        t0 = time.perf_counter()
        pids = self.prompt_ids(prompt)
        Lp = len(pids)
        pre = list(answer_prefix)
        x = torch.full((1, Lp + len(pre) + gen_length), MASK_ID, dtype=torch.long, device="cuda")
        x[0, :Lp] = torch.tensor(pids, device="cuda")
        if pre:
            x[0, Lp : Lp + len(pre)] = torch.tensor(pre, device="cuda")
        x = self._generate(x, Lp, Lp + len(pre), gen_length, gen_length, block_length, seed)
        ids = x[0, Lp:].tolist()
        torch.cuda.synchronize()
        return dict(ids=ids, free_start=len(pre), seconds=time.perf_counter() - t0, prefix_len=Lp)

    def native(self, entry, gen_length=256, block_length=16):
        out = self.generate_suffix(
            entry["prompt"],
            [],
            int(entry.get("seed", 0)),
            gen_length=gen_length,
            block_length=block_length,
        )
        text, e = self.decode(out["ids"])
        return dict(
            native=dict(
                answer_ids=out["ids"],
                response=text,
                prefix_ids=self.prompt_ids(entry["prompt"]),
                seconds=out["seconds"],
                seed=int(entry.get("seed", 0)),
            )
        )

    def decode(self, ids):
        ids = list(ids)
        e = ids.index(EOS) if EOS in ids else len(ids)
        return (self.tok.decode(ids[:e], skip_special_tokens=True), e)

    @torch.no_grad()
    def harmonize_batch(self, entry, cert_answer, frees, *, tstar=0, seeds=(0,), k=None):
        """Leave-k-out re-encoding of the assembled answer: logits for every free position predicted while
        it (and every k-th other free position) is masked and all else is visible. Returns (B, A, V) float32 CPU."""
        if k is None:
            k = int(os.environ.get("PACE_REENC_GROUPS", "5"))
        pids = list(entry["prefix_ids"])
        Lp = len(pids)
        A = len(cert_answer)
        V = 126349
        outs = []
        for free in frees:
            base = torch.tensor([pids + list(cert_answer)], device="cuda")
            logits_all = torch.zeros((A, V), dtype=torch.float32)
            free_idx = [j for j, f in enumerate(free) if f]
            for r in range(k):
                sel = free_idx[r::k]
                if not sel:
                    continue
                x = base.clone()
                x[0, [Lp + j for j in sel]] = MASK_ID
                lg = self.model(x).logits[0, :, :V].float()
                for j in sel:
                    logits_all[j] = lg[Lp + j].cpu()
            for j, f in enumerate(free):
                if not f:
                    logits_all[j] = -30.0
                    logits_all[j, cert_answer[j]] = 0.0
            outs.append(logits_all)
        return torch.stack(outs)
