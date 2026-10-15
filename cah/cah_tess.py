import json
import os
import re
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'assets/tess')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

def read(p):
    return json.loads(Path(p).read_text())

def graph_event(entry):
    import numpy as np
    from compile_tess_ifeval_quotients_v1 import to_event
    path = ROOT / f'commongen_token_compilation_v2/cases/validation_{int(entry['ordinal']):04d}_{int(entry['concept_set_idx'])}/graph.npz'
    with np.load(path, allow_pickle=False) as a:
        graph = {k: a[k] for k in a.files}
    return to_event(graph)

class Tess:

    def __init__(self):
        import torch
        from transformers import AutoTokenizer
        from sdlm.models.mistral.configuration_mistral import MistralDiffusionConfig
        from sdlm.models.mistral.modeling_mistral import MistralForDiffusionLM
        torch.backends.cuda.matmul.allow_tf32 = False
        snapshot = os.environ.get('TESS_MODEL', 'hamishivi/tess2')
        config = MistralDiffusionConfig.from_pretrained(snapshot, local_files_only=False)
        self.model = MistralForDiffusionLM.from_pretrained(snapshot, config=config, torch_dtype=torch.bfloat16, attn_implementation='sdpa', low_cpu_mem_usage=True, local_files_only=False, use_safetensors=True).eval().requires_grad_(False).to('cuda')
        self.tok = AutoTokenizer.from_pretrained(snapshot, use_fast=True, local_files_only=False)
        self.torch = torch

    def native(self, entry):
        """Archived-identical native run; returns record + final bf16 logits (CPU)."""
        from tess_native_final_capture_v1 import generate
        row = dict(key=int(entry['index']), prompt=entry['prompt'])
        out = generate(self.model, self.tok, row, int(entry['seed']))
        return out

    def harmonize(self, entry, cert_answer, free_answer, *, tstar=40, seed=0, sc_init='zeros'):
        torch = self.torch
        from sdlm.schedulers import TokenWiseSimplexDDPMScheduler
        from sdlm.inference.inference_utils import logits_projection
        from sdlm.utils import scale, self_condition_preds, convert_to_simplex
        from tess_bidirectional_attention_adapter import bidirectional_attention
        model = self.model
        prefix = json.loads(entry['prefix_ids']) if isinstance(entry['prefix_ids'], str) else entry['prefix_ids']
        P = len(prefix)
        ids = torch.tensor([list(prefix) + list(cert_answer)], device='cuda')
        span = torch.tensor([[False] * P + list(map(bool, free_answer))], device='cuda')
        L, V = (ids.shape[1], model.config.vocab_size)
        sched = TokenWiseSimplexDDPMScheduler(device='cuda', simplex_value=5.0, num_train_timesteps=100, beta_schedule='squaredcos_improved_ddpm')
        gen = torch.Generator(device='cuda').manual_seed(int(seed))
        torch.manual_seed(int(seed))
        proj = lambda x: logits_projection(x, 'top_p', 0.99, 5.0, 1.0)
        with torch.no_grad(), bidirectional_attention(model), torch.autocast('cuda', dtype=torch.bfloat16):
            clean = convert_to_simplex(ids, 5.0, V).float()
            noise = 5.0 * torch.randn(clean.shape, generator=gen, device='cuda')
            simplex = sched.add_noise(clean, noise, torch.tensor([int(tstar)], device='cuda'))
            if sc_init == 'certified':
                previous = self_condition_preds(model.config.self_condition, clean.to(torch.bfloat16), proj)
            else:
                previous = torch.zeros((1, L, V), device='cuda')
            hidden = None
            logits = None
            for t in range(int(tstar), -1, -1):
                tt = torch.tensor([t], device='cuda').expand(1, L)
                out = model(input_ids=ids, span_mask=span, simplex=simplex, timesteps=scale(tt, 100), previous_pred=previous, classifier_free_guidance=False, reduce_loss='none', max_timestep=100, previous_hidden=hidden)
                logits = out.logits.to(torch.bfloat16)
                hidden = out.hidden_states
                previous = self_condition_preds(model.config.self_condition, logits, proj)
                n = 5.0 * torch.randn(clean.shape, generator=gen, device='cuda')
                simplex = sched.step(proj(logits), tt, tt - 1, n, generator=gen).prev_sample
        answer = logits[0, P:].float().cpu()
        return answer

    def generate_suffix(self, prompt, answer_prefix, seed, *, total=2048, steps=100):
        """Native TESS conditional generation with a fixed answer prefix (prompt + answer_prefix clean,
        remaining positions generated). Returns dict(ids=full answer ids incl. prefix, free_start,
        logits=final logits on free positions (CPU float32), seconds)."""
        import time
        torch = self.torch
        from tess_ifeval_prompt_contract import format_prompt
        from tess_bidirectional_attention_adapter import bidirectional_attention
        from sdlm.pipelines.simplex_ddpm import SimplexDDPMClassifierGuidancePipeline
        from sdlm.schedulers import TokenWiseSimplexDDPMScheduler
        tok, model = (self.tok, self.model)
        prefix = tok.encode(format_prompt(prompt, tok.bos_token), add_special_tokens=False)
        fixed = list(prefix) + list(answer_prefix)
        if len(fixed) >= total - 16:
            raise ValueError('answer prefix too long')
        ids = torch.tensor([fixed + [tok.pad_token_id] * (total - len(fixed))], device='cuda')
        span = torch.tensor([[False] * len(fixed) + [True] * (total - len(fixed))], device='cuda')
        batch = dict(input_ids=ids, span_mask=span)
        torch.manual_seed(seed)
        scheduler = TokenWiseSimplexDDPMScheduler(device='cuda', simplex_value=5.0, num_train_timesteps=steps, beta_schedule='squaredcos_improved_ddpm')
        pipeline = SimplexDDPMClassifierGuidancePipeline(model=model, scheduler=scheduler, simplex_value=5.0, top_p=0.99, sampling_type='top_p', is_conditional_generation=True, tokenizer=tok, classifier_free_uncond_input='empty_token', temperature=1.0, guidance_softmax_combination=False, classifier_model_name_or_path=None)
        pipeline.set_progress_bar_config(disable=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), bidirectional_attention(model), torch.autocast('cuda', dtype=torch.bfloat16):
            for output in pipeline(batch=batch, guidance_scale=0.0, generator=torch.Generator(device='cuda').manual_seed(seed)):
                pass
        torch.cuda.synchronize()
        P = len(prefix)
        free_ids = output.logits.argmax(-1)[0, len(fixed):].cpu().tolist()
        return dict(ids=list(answer_prefix) + free_ids, free_start=len(answer_prefix), logits=output.logits[0, len(fixed):].float().cpu(), seconds=time.perf_counter() - t0, prefix_len=P)

    def harmonize_batch(self, entry, cert_answer, frees, *, tstar=40, seeds=(0,)):
        """Batched harmonization of several proposals (same certified text, same t*), one per free mask/seed."""
        torch = self.torch
        from sdlm.schedulers import TokenWiseSimplexDDPMScheduler
        from sdlm.inference.inference_utils import logits_projection
        from sdlm.utils import scale, self_condition_preds, convert_to_simplex
        from tess_bidirectional_attention_adapter import bidirectional_attention
        model = self.model
        prefix = entry['prefix_ids']
        P = len(prefix)
        B = len(frees)
        ids = torch.tensor([list(prefix) + list(cert_answer)] * B, device='cuda')
        span = torch.tensor([[False] * P + list(map(bool, f)) for f in frees], device='cuda')
        L, V = (ids.shape[1], model.config.vocab_size)
        sched = TokenWiseSimplexDDPMScheduler(device='cuda', simplex_value=5.0, num_train_timesteps=100, beta_schedule='squaredcos_improved_ddpm')
        gens = [torch.Generator(device='cuda').manual_seed(int(s)) for s in seeds]
        proj = lambda x: logits_projection(x, 'top_p', 0.99, 5.0, 1.0)

        def noise():
            return torch.cat([5.0 * torch.randn((1, L, V), generator=g, device='cuda') for g in gens])
        with torch.no_grad(), bidirectional_attention(model), torch.autocast('cuda', dtype=torch.bfloat16):
            clean = convert_to_simplex(ids, 5.0, V).float()
            simplex = sched.add_noise(clean, noise(), torch.full((B,), int(tstar), device='cuda'))
            previous = torch.zeros((B, L, V), device='cuda')
            hidden = None
            for t in range(int(tstar), -1, -1):
                tt = torch.full((B, L), t, device='cuda')
                out = model(input_ids=ids, span_mask=span, simplex=simplex, timesteps=scale(tt, 100), previous_pred=previous, classifier_free_guidance=False, reduce_loss='none', max_timestep=100, previous_hidden=hidden)
                logits = out.logits.to(torch.bfloat16)
                hidden = out.hidden_states
                previous = self_condition_preds(model.config.self_condition, logits, proj)
                simplex = sched.step(proj(logits), tt, tt - 1, noise()).prev_sample
        return logits[:, P:].float().cpu()

    def self_score(self, entry, candidates, region, *, levels=(30, 50, 70), seed=0):
        """DLM-native window score: mean over noise levels of sum_{j in region} log p(y_j | rest),
        with the region forward-noised from each candidate's own simplex at level t (common noise)."""
        torch = self.torch
        from sdlm.utils import scale, convert_to_simplex
        from sdlm.schedulers import TokenWiseSimplexDDPMScheduler
        from tess_bidirectional_attention_adapter import bidirectional_attention
        model = self.model
        prefix = json.loads(entry['prefix_ids']) if isinstance(entry['prefix_ids'], str) else entry['prefix_ids']
        P = len(prefix)
        L, V = (P + len(candidates[0]), model.config.vocab_size)
        sched = TokenWiseSimplexDDPMScheduler(device='cuda', simplex_value=5.0, num_train_timesteps=100, beta_schedule='squaredcos_improved_ddpm')
        span = torch.tensor([[False] * P + list(map(bool, region))], device='cuda')
        pos = torch.nonzero(span[0]).flatten()
        scores = [0.0] * len(candidates)
        with torch.no_grad(), bidirectional_attention(model), torch.autocast('cuda', dtype=torch.bfloat16):
            for t in levels:
                gen = torch.Generator(device='cuda').manual_seed(int(seed) * 1000 + int(t))
                noise = 5.0 * torch.randn((1, L, V), generator=gen, device='cuda')
                for c, cand in enumerate(candidates):
                    ids = torch.tensor([list(prefix) + list(cand)], device='cuda')
                    clean = convert_to_simplex(ids, 5.0, V).float()
                    simplex = sched.add_noise(clean, noise, torch.tensor([int(t)], device='cuda'))
                    tt = torch.tensor([int(t)], device='cuda').expand(1, L)
                    out = model(input_ids=ids, span_mask=span, simplex=simplex, timesteps=scale(tt, 100), previous_pred=torch.zeros((1, L, V), device='cuda'), classifier_free_guidance=False, reduce_loss='none', max_timestep=100)
                    lp = torch.log_softmax(out.logits[0, pos].float(), -1)
                    scores[c] += float(lp.gather(-1, ids[0, pos][:, None]).sum()) / len(levels)
        return scores

    def decode(self, answer_ids):
        eos = self.tok.eos_token_id
        ids = list(answer_ids)
        cut = ids.index(eos) if eos in ids else len(ids)
        return (self.tok.decode(ids[:cut], skip_special_tokens=True, clean_up_tokenization_spaces=False), cut)

def certify(answer_logits, event, fixed_ids=None, fixed_mask=None, truncate=None, eos=2):
    """Exact output-only MAP under the compiled surface automaton (sparse CPU FP64).

    truncate=N solves on the first N positions and keeps the argmax tail; accepted only if the
    truncated optimum contains EOS (decoded text then ends inside the solved range), otherwise
    falls back to the full-length solve.
    """
    import numpy as np
    from branched_token_map_sparse_v1 import SparseBranchedTokenMAP
    x = np.asarray(answer_logits, dtype=np.float64).copy()
    if fixed_mask is not None:
        for j in np.nonzero(np.asarray(fixed_mask))[0]:
            x[j] = -10000.0
            x[j, fixed_ids[j]] = 0.0
    if truncate is not None and truncate < len(x):
        part = SparseBranchedTokenMAP(event, max_trace_bytes=4 * 1024 ** 3).solve(x[:truncate])['token_ids']
        if eos in part:
            return list(part) + x[truncate:].argmax(-1).tolist()
    return SparseBranchedTokenMAP(event, max_trace_bytes=4 * 1024 ** 3).solve(x)['token_ids']
