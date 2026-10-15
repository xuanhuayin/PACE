import os
from pathlib import Path
SNAP = os.environ.get('DREAM_SNAP', 'Dream-org/Dream-v0-Instruct-7B')
MASK_ID = 151666
EOS_ID = 151643
PAD_ID = 151643
IM_END = 151645
SKIP_IDS = (151665, 151666)
VOCAB = 151667
FAST_DIR = Path(__file__).resolve().parents[1] / 'assets/cache/dream_tokenizer_fast'

def slow_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)

def contract(tok=None):
    from llada_decoder import LLaDADecoderContract
    return LLaDADecoderContract(tok or slow_tokenizer(), eos=EOS_ID, extra_special=(MASK_ID,))

def fast_tokenizer():
    """Qwen2TokenizerFast built from Dream's own vocab/merges (+ the 24 added tokens with Dream's ids/flags and
    chat template). Needed for return_offsets_mapping (program_gen2.certify_text); the slow DreamTokenizer has
    no offsets. Built once and cached in FAST_DIR. Encoding equality vs the slow tokenizer is checked in
    dream_contract_test.py."""
    from transformers import PreTrainedTokenizerFast
    if (FAST_DIR / 'tokenizer.json').exists():
        return PreTrainedTokenizerFast.from_pretrained(str(FAST_DIR))
    from transformers import AddedToken
    from transformers.convert_slow_tokenizer import Qwen2Converter
    slow = slow_tokenizer()
    obj = Qwen2Converter(slow).converted()
    fast = PreTrainedTokenizerFast(tokenizer_object=obj, clean_up_tokenization_spaces=False)
    dec = slow.added_tokens_decoder
    for i in sorted(dec):
        t = dec[i]
        fast.add_tokens([AddedToken(t.content, lstrip=t.lstrip, rstrip=t.rstrip, normalized=t.normalized, single_word=t.single_word, special=t.special)], special_tokens=t.special)
        assert fast.convert_tokens_to_ids(t.content) == i, (i, t.content)
    fast.add_special_tokens(dict(eos_token='<|endoftext|>', pad_token='<|endoftext|>', bos_token='<|beginoftext|>', mask_token='<|mask|>', additional_special_tokens=['<|beginoftext|>', '<|mask|>']))
    fast.chat_template = slow.chat_template
    assert len(fast) == len(slow) == VOCAB, (len(fast), len(slow))
    FAST_DIR.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(str(FAST_DIR))
    return PreTrainedTokenizerFast.from_pretrained(str(FAST_DIR))

def contract_decode(dec, ids):
    """(text, eos_index) under the automaton's decoding contract."""
    ids = list(ids)
    e = ids.index(EOS_ID) if EOS_ID in ids else len(ids)
    return (dec.decode([t for t in ids[:e] if t < VOCAB]), e)
