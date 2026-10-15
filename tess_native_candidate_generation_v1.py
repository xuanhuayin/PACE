import hashlib
import math
import sys
import time
import torch
from tess_ifeval_baseline import ROOT, SETTINGS, encode, decode
sys.path.insert(0, str(ROOT / 'assets/tess'))
from sdlm.pipelines.simplex_ddpm import SimplexDDPMClassifierGuidancePipeline
from sdlm.schedulers import TokenWiseSimplexDDPMScheduler
from tess_bidirectional_attention_adapter import bidirectional_attention
from tess_ifeval_prompt_contract import single_batch

def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

def generate(model, tokenizer, row, seed, trace=False):
    encoded = encode(row, tokenizer, SETTINGS['total_positions'])
    batch = single_batch(encoded, 'cuda')
    original = {k: v.clone() for k, v in batch.items()}
    torch.manual_seed(seed)
    scheduler = TokenWiseSimplexDDPMScheduler(device='cuda', simplex_value=5.0, num_train_timesteps=SETTINGS['steps'], beta_schedule='squaredcos_improved_ddpm')
    pipeline = SimplexDDPMClassifierGuidancePipeline(model=model, scheduler=scheduler, simplex_value=5.0, top_p=0.99, sampling_type='top_p', is_conditional_generation=True, tokenizer=tokenizer, classifier_free_uncond_input='empty_token', temperature=1.0, guidance_softmax_combination=False, classifier_model_name_or_path=None)
    pipeline.set_progress_bar_config(disable=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    hashes = []
    count = 0
    with torch.no_grad(), bidirectional_attention(model), torch.autocast('cuda', dtype=torch.bfloat16):
        for output in pipeline(batch=batch, guidance_scale=0.0, generator=torch.Generator(device='cuda').manual_seed(seed)):
            if tuple(output.logits.shape) != (1, 2048, 32001) or not torch.isfinite(output.logits).all():
                raise ValueError('Invalid native logits')
            count += 1
            if trace:
                hashes.append(tensor_hash(output.logits))
    if count != 100 or any((not torch.equal(batch[k], v) for k, v in original.items())):
        raise ValueError('Step count or input mutation')
    digest = tensor_hash(output.logits)
    ids = output.logits.argmax(-1)[0, batch['span_mask'][0]].cpu().tolist()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    response, eos = decode(ids, tokenizer)
    if not math.isfinite(seconds) or seconds <= 0 or len(ids) != encoded['answer_tokens']:
        raise ValueError('Invalid measurements or answer size')
    return dict(key=row['key'], prompt=row['prompt'], seed=seed, prefix_ids=encoded['prefix_ids'], answer_ids=ids, response=response, eos_position=eos, steps=count, final_logits_sha256=digest, step_logits_sha256=hashes, seconds=seconds, peak_cuda_bytes=torch.cuda.max_memory_allocated())
