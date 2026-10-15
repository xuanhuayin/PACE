import os
from pathlib import Path
REPO = 'inclusionAI/LLaDA2.0-mini'
SNAP = os.environ.get('LLADA2_SNAP', REPO)
MASK_ID = 156895
EOS_ID = 156892
PAD_ID = 156892
BOS_ID = 156891
ADDED_IDS = tuple(range(156891, 157153))
SKIP_IDS = tuple((i for i in ADDED_IDS if i != EOS_ID))
VOCAB = 157153
MODEL_VOCAB = 157184
MODEL_NAME = 'LLaDA2.0-mini'
GEN_LENGTH = int(os.environ.get('LLADA2_GEN', 512))
BLOCK = 32
STEPS = 32
THRESHOLD = 0.95

def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)

def contract(tok=None):
    from llada_decoder import LLaDADecoderContract
    return LLaDADecoderContract(tok or tokenizer(), eos=EOS_ID, extra_special=ADDED_IDS)

def contract_decode(dec, ids):
    """(text, eos_index) under the automaton's decoding contract."""
    ids = list(ids)
    e = ids.index(EOS_ID) if EOS_ID in ids else len(ids)
    return (dec.decode([t for t in ids[:e] if t < VOCAB]), e)

def chat_prompt_ids(tok, prompt):
    text = tok.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=True, tokenize=False)
    return tok(text, add_special_tokens=False)['input_ids']
