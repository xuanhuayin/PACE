import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'cah'))
from dream_common import SNAP, MASK_ID, EOS_ID, PAD_ID, VOCAB, SKIP_IDS, fast_tokenizer, contract, contract_decode
DECODING = dict(temperature=0.0, top_p=0.9, top_k=None, alg='entropy', alg_temp=0.0, eps=0.001)
CONF_TEMPERATURE = 0.1
NEVER_CONTENT = tuple(SKIP_IDS)

class DreamEngine:
    eos_id = EOS_ID
    pad_id = PAD_ID
    canvas_total = 1024
    vocab = VOCAB
    mask_id = MASK_ID

    def __init__(self, load_model=True):
        from transformers import AutoModel
        self.tok = fast_tokenizer()
        self.dec = contract()
        self.torch = torch
        self.model = None
        if load_model:
            self.model = AutoModel.from_pretrained(SNAP, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()

    def prompt_ids(self, prompt):
        text = self.tok.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=True, tokenize=False)
        return self.tok(text, add_special_tokens=False)['input_ids']

    @torch.no_grad()
    def run_sampler(self, input_ids, gen_length=256, steps=None, capture_topk=0):
        """Dream diffusion_generate on input_ids (1, L) + gen_length masks. With capture_topk > 0, records for every
        newly unmasked position its commit-time log-probs over the tokenizer vocabulary from the model's raw (untempered,
        shifted) logits at the step that committed it: (top ids int32, top logp fp16, logp[EOS], max logp, step).
        Returns (x (1, L+gen), commits dict)."""
        steps = steps or gen_length
        commits, state = ({}, {})

        def logits_hook(step, x, logits):
            state['pre_mask'] = (x == MASK_ID).clone()
            state['raw'] = logits
            return logits / CONF_TEMPERATURE

        def tokens_hook(step, x, logits):
            if step is None or not capture_topk:
                return x
            new = state['pre_mask'][0] & (x[0] != MASK_ID)
            pos = torch.nonzero(new).flatten()
            if len(pos):
                lp = F.log_softmax(state['raw'][0, pos, :VOCAB].float(), dim=-1)
                tv, ti = torch.topk(lp, capture_topk, dim=-1)
                ti, tv = (ti.cpu().numpy(), tv.cpu().numpy())
                eos_lp = lp[:, EOS_ID].cpu().numpy()
                mx = lp.max(-1).values.cpu().numpy()
                for r, p in enumerate(pos.tolist()):
                    commits[p] = (ti[r].astype('int32'), tv[r].astype('float16'), float(eos_lp[r]), float(mx[r]), step)
            return x
        out = self.model.diffusion_generate(input_ids, attention_mask=None, max_new_tokens=gen_length, steps=steps, mask_token_id=MASK_ID, output_history=False, return_dict_in_generate=True, generation_logits_hook_func=logits_hook, generation_tokens_hook_func=tokens_hook, **DECODING)
        return (out.sequences, commits)

    def generate_suffix(self, prompt, answer_prefix, seed, *, total=None, gen_length=256):
        """Fixed answer prefix (part of the visible context), then `gen_length` masked positions (steps = gen_length).
        `total` is accepted for interface compatibility and ignored (as in LLaDAEngine)."""
        t0 = time.perf_counter()
        pids = self.prompt_ids(prompt)
        Lp = len(pids)
        pre = list(answer_prefix)
        torch.manual_seed(seed)
        x, _ = self.run_sampler(torch.tensor([pids + pre], device='cuda'), gen_length=gen_length)
        ids = x[0, Lp:].tolist()
        torch.cuda.synchronize()
        return dict(ids=ids, free_start=len(pre), seconds=time.perf_counter() - t0, prefix_len=Lp)

    def native(self, entry, gen_length=256):
        out = self.generate_suffix(entry['prompt'], [], int(entry.get('seed', 0)), gen_length=gen_length)
        text, e = self.decode(out['ids'])
        return dict(native=dict(answer_ids=out['ids'], response=text, prefix_ids=self.prompt_ids(entry['prompt']), seconds=out['seconds'], seed=int(entry.get('seed', 0)), eos=e))

    def decode(self, ids):
        return contract_decode(self.dec, ids)

    @torch.no_grad()
    def harmonize_batch(self, entry, cert_answer, frees, *, tstar=0, seeds=(0,), k=5):
        """Leave-k-out re-encoding of the assembled answer: logits for every free position predicted while it (and every
        k-th other free position) is masked and all else is visible (Dream shift: row p comes from output p-1).
        Returns (B, A, VOCAB) float32 CPU. Fixed positions get a sharp distribution on their own token."""
        pids = list(entry['prefix_ids'])
        Lp = len(pids)
        A = len(cert_answer)
        base = torch.tensor([pids + list(cert_answer)], device='cuda')
        outs = []
        for free in frees:
            logits_all = torch.zeros((A, VOCAB), dtype=torch.float32)
            free_idx = [j for j, f in enumerate(free) if f]
            for r in range(k):
                sel = free_idx[r::k]
                if not sel:
                    continue
                x = base.clone()
                x[0, [Lp + j for j in sel]] = MASK_ID
                raw = self.model(x).logits[0]
                rows = torch.tensor([Lp + j - 1 for j in sel], device='cuda')
                logits_all[sel] = raw[rows, :VOCAB].float().cpu()
            if free_idx:
                logits_all[free_idx, NEVER_CONTENT[0]] = -10000.0
                logits_all[free_idx, NEVER_CONTENT[1]] = -10000.0
            for j, f in enumerate(free):
                if not f:
                    logits_all[j] = -30.0
                    logits_all[j, cert_answer[j]] = 0.0
            outs.append(logits_all)
        return torch.stack(outs)
