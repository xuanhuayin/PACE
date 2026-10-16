from functools import lru_cache
import re

CLEANUP = (
    (" .", "."),
    (" ?", "?"),
    (" !", "!"),
    (" ,", ","),
    (" ' ", "'"),
    (" n't", "n't"),
    (" 'm", "'m"),
    (" 's", "'s"),
    (" 've", "'ve"),
    (" 're", "'re"),
)
LETTER = re.compile("[A-Za-z]", re.I)


def replace_step(pending, text, old, new):
    """Streaming, nonoverlapping str.replace; retain only a possible prefix."""
    emitted = []
    for char in text:
        pending += char
        while pending and (not old.startswith(pending)):
            emitted.append(pending[0])
            pending = pending[1:]
        if pending == old:
            emitted.append(new)
            pending = ""
    return (pending, "".join(emitted))


class LexicalStream:
    def __init__(self, word):
        if not re.fullmatch("[a-z]{2,}", word):
            raise ValueError("Lowercase ASCII target of length >=2 required")
        self.word = word
        self.letters = tuple((re.compile(char, re.I) for char in word))
        self.start = (("",) * len(CLEANUP), 0)

    @lru_cache(maxsize=4096)
    def letter_step(self, progress, char):
        if not LETTER.fullmatch(char):
            return (0, int(progress == len(self.word)))
        if progress < len(self.word) and self.letters[progress].fullmatch(char):
            return (progress + 1, 0)
        return (len(self.word) + 1, 0)

    def consume(self, progress, text):
        count = 0
        for char in text:
            progress, increment = self.letter_step(progress, char)
            count += increment
        return (progress, count)

    @lru_cache(maxsize=65536)
    def step(self, state, surface):
        buffers, progress = state
        updated = []
        for pending, (old, new) in zip(buffers, CLEANUP):
            pending, surface = replace_step(pending, surface, old, new)
            updated.append(pending)
        progress, count = self.consume(progress, surface)
        return ((tuple(updated), progress), count)

    @lru_cache(maxsize=65536)
    def finish(self, state):
        buffers, progress = state
        surface = ""
        for pending, (old, new) in zip(buffers, CLEANUP):
            pending, surface = replace_step(pending, surface, old, new)
            surface += pending
        progress, count = self.consume(progress, surface)
        return count + int(progress == len(self.word))

    def count_fragments(self, fragments):
        state = self.start
        count = 0
        for fragment in fragments:
            state, increment = self.step(state, fragment)
            count += increment
        return count + self.finish(state)

    def count_ids(self, ids, surfaces, eos_id=1, stop_at_eos=True):
        state = self.start
        count = 0
        for token in ids:
            if type(token) is not int or not 0 <= token < len(surfaces):
                raise ValueError("Token identity")
            if stop_at_eos and token == eos_id:
                break
            state, increment = self.step(state, surfaces[token])
            count += increment
        return count + self.finish(state)


def token_surfaces(tokenizer):
    """Pinned Metaspace decoder only. Leading-space trim is count-invariant."""
    import json

    decoder = json.loads(tokenizer.backend_tokenizer.to_str())["decoder"]
    if decoder != dict(type="Metaspace", replacement="▁", prepend_scheme="always", split=True):
        raise ValueError("Unsupported decoder")
    special = set(tokenizer.all_special_ids)
    return tuple(
        (
            "" if i in special else tokenizer.convert_ids_to_tokens(i).replace("▁", " ")
            for i in range(len(tokenizer))
        )
    )


def toy_log_probability(logits, word, surfaces, k, eos_id):
    """Exact differentiable toy enumeration by merged finite states.

    Deliberately limited to <=8 positions and <=16 tokens. This is a correctness
    prototype, not a vocabulary-scale implementation or a performance result.
    """
    import torch

    if logits.ndim != 2 or not 0 < logits.shape[0] <= 8 or (not 0 < logits.shape[1] <= 16):
        raise ValueError("Toy dimensions only")
    if (
        len(surfaces) != logits.shape[1]
        or type(k) is not int
        or k < 0
        or (not 0 <= eos_id < len(surfaces))
    ):
        raise ValueError("Toy identity")
    if logits.dtype not in (torch.float32, torch.float64) or not torch.isfinite(logits).all():
        raise ValueError("Finite floating logits")
    machine = LexicalStream(word)
    lp = logits.log_softmax(-1)
    states = {(0, machine.start): logits.sum() * 0}
    for position in range(logits.shape[0]):
        next_states = {}
        for (count, state), weight in states.items():
            if state is None:
                key = (count, None)
                next_states[key] = (
                    torch.logaddexp(next_states[key], weight) if key in next_states else weight
                )
                continue
            for token, surface in enumerate(surfaces):
                if token == eos_id:
                    dest, increment = (None, machine.finish(state))
                else:
                    dest, increment = machine.step(state, surface)
                if count + increment > k:
                    continue
                key = (count + increment, dest)
                value = weight + lp[position, token]
                next_states[key] = (
                    torch.logaddexp(next_states[key], value) if key in next_states else value
                )
        states = next_states
    values = [
        weight
        for (count, state), weight in states.items()
        if count + (0 if state is None else machine.finish(state)) == k
    ]
    if not values:
        return logits.sum() * 0 + logits.new_tensor(-torch.inf)
    return torch.logsumexp(torch.stack(values), 0)
