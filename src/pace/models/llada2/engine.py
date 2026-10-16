from pace.paths import PROJECT_ROOT
import os
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = PROJECT_ROOT


from pace.models.llada2.config import (
    SNAP,
    MASK_ID,
    EOS_ID,
    PAD_ID,
    VOCAB,
    SKIP_IDS,
    GEN_LENGTH,
    BLOCK,
    STEPS,
    THRESHOLD,
    tokenizer,
    contract,
    contract_decode,
    chat_prompt_ids,
)

NEVER_CONTENT = tuple(SKIP_IDS)
DECODING = dict(
    block_length=BLOCK,
    steps=STEPS,
    threshold=THRESHOLD,
    temperature=0.0,
    argmax=True,
    top_p=None,
    top_k=None,
)


def num_transfer_schedule(block_length, steps):
    """== LLaDA2MoeModelLM._get_num_transfer_tokens."""
    base, rem = (block_length // steps, block_length % steps)
    s = [base] * steps
    for i in range(rem):
        s[i] += 1
    return s


def block_causal_mask(n, block, device, dtype=torch.bfloat16):
    """(1, 1, n, n) additive mask: 0 where block(key) <= block(query), -inf elsewhere (as generate())."""
    b = torch.arange(n, device=device) // block
    m = torch.full((n, n), float("-inf"), device=device, dtype=dtype)
    m.masked_fill_(b[None, :] <= b[:, None], 0.0)
    return m[None, None]


class LLaDA2Engine:
    eos_id = EOS_ID
    pad_id = PAD_ID
    canvas_total = 1024
    vocab = VOCAB
    mask_id = MASK_ID
    block = BLOCK

    def __init__(self, load_model=True):
        self.tok = tokenizer()
        self.dec = contract(self.tok)
        self.torch = torch
        self.model = None
        if load_model:
            from transformers import AutoModelForCausalLM

            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    SNAP, trust_remote_code=True, torch_dtype=torch.bfloat16
                )
                .cuda()
                .eval()
            )

    def prompt_ids(self, prompt):
        return chat_prompt_ids(self.tok, prompt)

    def _logits(self, x, attn, pos, rows=None):
        """Model forward; logits (float32) for `rows` (tensor of positions) or all positions."""
        h = self.model.model(input_ids=x, attention_mask=attn, position_ids=pos)[0][0]
        if rows is not None:
            h = h[rows]
        return self.model.lm_head(h).float()

    @torch.no_grad()
    def run_sampler(
        self,
        input_ids,
        gen_length=GEN_LENGTH,
        block_length=BLOCK,
        steps=STEPS,
        threshold=THRESHOLD,
        capture_topk=0,
        continue_after_eos=False,
    ):
        """input_ids: list (prompt [+ fixed answer prefix]). Returns (x list of length len(input)+gen_length, commits
        {abs pos: (top ids int32, top logp fp16, logp[EOS], max logp, global step)}, info)."""
        dev = "cuda"
        L0 = len(input_ids)
        steps = min(steps, gen_length)
        nblk = (L0 + gen_length + block_length - 1) // block_length
        total = nblk * block_length
        attn_full = block_causal_mask(total, block_length, dev)
        pos_full = torch.arange(total, device=dev)[None]
        x = torch.full((1, total), MASK_ID, dtype=torch.long, device=dev)
        x[0, :L0] = torch.tensor(input_ids, device=dev)
        sched = num_transfer_schedule(block_length, steps)
        commits, gstep, eos_block, eos_pos, nfe = ({}, 0, None, None, 0)
        for nb in range(L0 // block_length, nblk):
            end = (nb + 1) * block_length
            cur = x[:, :end]
            for step in range(steps):
                active = cur[0, -block_length:] == MASK_ID
                if not bool(active.any()):
                    break
                lg = self._logits(
                    cur,
                    attn_full[:, :, :end, :end],
                    pos_full[:, :end],
                    rows=torch.arange(end - block_length, end, device=dev),
                )
                nfe += 1
                p = F.softmax(lg, dim=-1)
                x0 = torch.argmax(lg, dim=-1)
                x0_p = torch.gather(p, -1, x0[:, None])[:, 0]
                conf = torch.where(active, x0_p, torch.full_like(x0_p, float("-inf")))
                high = conf > threshold
                transfer = torch.zeros_like(active)
                if int(high.sum()) >= sched[step]:
                    transfer = high
                else:
                    _, idx = torch.topk(conf, k=min(sched[step], int(active.sum())))
                    transfer[idx] = True
                if capture_topk:
                    rr = torch.nonzero(transfer).flatten()
                    lp = F.log_softmax(lg[rr, :VOCAB], dim=-1)
                    tv, ti = torch.topk(lp, capture_topk, dim=-1)
                    ti, tv = (ti.cpu().numpy(), tv.cpu().numpy())
                    eos_lp = lp[:, EOS_ID].cpu().numpy()
                    mx = lp.max(-1).values.cpu().numpy()
                    for r, j in enumerate(rr.tolist()):
                        commits[end - block_length + j] = (
                            ti[r].astype("int32"),
                            tv[r].astype("float16"),
                            float(eos_lp[r]),
                            float(mx[r]),
                            gstep,
                        )
                blk = cur[0, -block_length:]
                blk[transfer] = x0[transfer]
                gstep += 1
            if eos_block is None and bool((x[0, L0:end] == EOS_ID).any()):
                eos_block = nb
                eos_pos = int((x[0, L0:end] == EOS_ID).nonzero()[0, 0]) + L0
                if not continue_after_eos:
                    break
            if (
                eos_block is not None
                and continue_after_eos is not True
                and (end >= eos_pos + 1 + int(continue_after_eos))
            ):
                break
        return (
            x[0, : L0 + gen_length].tolist(),
            commits,
            dict(nfe=nfe, eos_block=eos_block, blocks=nblk),
        )

    def generate_suffix(self, prompt, answer_prefix, seed, *, total=None, gen_length=GEN_LENGTH):
        """Fixed answer prefix (part of the visible context, like the prompt), then `gen_length` masked positions.
        `total` is accepted for interface compatibility and ignored (as LLaDAEngine / DreamEngine). Deterministic (seed
        only seeds torch for interface parity). Positions after the EOS block stay <|mask|> (a no-text special after EOS)."""
        t0 = time.perf_counter()
        pids = self.prompt_ids(prompt)
        Lp = len(pids)
        pre = list(answer_prefix)
        torch.manual_seed(seed)
        x, _, info = self.run_sampler(pids + pre, gen_length=gen_length)
        torch.cuda.synchronize()
        return dict(
            ids=x[Lp:],
            free_start=len(pre),
            seconds=time.perf_counter() - t0,
            prefix_len=Lp,
            nfe=info["nfe"],
        )

    def native(self, entry, gen_length=GEN_LENGTH):
        out = self.generate_suffix(
            entry["prompt"], [], int(entry.get("seed", 0)), gen_length=gen_length
        )
        text, e = self.decode(out["ids"])
        return dict(
            native=dict(
                answer_ids=out["ids"],
                response=text,
                prefix_ids=self.prompt_ids(entry["prompt"]),
                seconds=out["seconds"],
                seed=int(entry.get("seed", 0)),
                eos=e,
            )
        )

    def decode(self, ids):
        return contract_decode(self.dec, ids)

    @torch.no_grad()
    def harmonize_batch(self, entry, cert_answer, frees, *, tstar=0, seeds=(0,), k=5):
        """Leave-k-out re-encoding of the assembled answer under block-diffusion attention (two-stream, see module doc):
        every free position is predicted once while it (and every k-th other free position) is masked, with its own block
        otherwise visible and all earlier blocks clean. Returns (B, A, VOCAB) float32 CPU. Fixed positions get a sharp
        distribution on their own token; no-text specials are floored on free positions."""
        dev = "cuda"
        Bk = BLOCK
        pids = list(entry["prefix_ids"])
        Lp = len(pids)
        A = len(cert_answer)
        L = Lp + A
        clean = torch.tensor(pids + list(cert_answer), device=dev)
        s0 = Lp // Bk * Bk
        n2 = L - s0
        pos = torch.cat([torch.arange(L, device=dev), torch.arange(s0, L, device=dev)])[None]
        bq = torch.cat([torch.arange(L, device=dev), torch.arange(s0, L, device=dev)]) // Bk
        stream2 = torch.cat(
            [
                torch.zeros(L, dtype=torch.bool, device=dev),
                torch.ones(n2, dtype=torch.bool, device=dev),
            ]
        )
        bq_, bk_ = (bq[:, None], bq[None, :])
        s2q, s2k = (stream2[:, None], stream2[None, :])
        allowed = ~s2q & ~s2k & (bk_ <= bq_) | s2q & ~s2k & (bk_ < bq_) | s2q & s2k & (bk_ == bq_)
        attn = torch.full(allowed.shape, float("-inf"), device=dev, dtype=torch.bfloat16)
        attn.masked_fill_(allowed, 0.0)
        attn = attn[None, None]
        outs = []
        for free in frees:
            logits_all = torch.zeros((A, VOCAB), dtype=torch.float32)
            free_idx = [j for j, f in enumerate(free) if f]
            for r in range(k):
                sel = free_idx[r::k]
                if not sel:
                    continue
                noisy = clean[s0:].clone()
                noisy[torch.tensor([Lp + j - s0 for j in sel], device=dev)] = MASK_ID
                x = torch.cat([clean, noisy])[None]
                rows = torch.tensor([L + (Lp + j - s0) for j in sel], device=dev)
                lg = self._logits(x, attn, pos, rows=rows)
                logits_all[sel] = lg[:, :VOCAB].cpu()
            if free_idx:
                logits_all[
                    torch.tensor(free_idx)[:, None], torch.tensor(NEVER_CONTENT)[None, :]
                ] = -10000.0
            for j, f in enumerate(free):
                if not f:
                    logits_all[j] = -30.0
                    logits_all[j, cert_answer[j]] = 0.0
            outs.append(logits_all)
        return torch.stack(outs)
