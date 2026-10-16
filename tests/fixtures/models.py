class FakeTok:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))

    def decode(self, ids, **kw):
        return bytes((i for i in ids if 0 < i < 256)).decode("utf-8", "ignore")

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        b = text.encode("utf-8")
        return dict(input_ids=list(b), offset_mapping=[(i, i + 1) for i in range(len(b))])


class FakeEngine:
    """script(prefix_text, attempt, call_index) -> continuation text."""

    eos_id = 256
    pad_id = 256
    canvas_total = 4096

    def __init__(self, script):
        self.tok = FakeTok()
        self.script = script
        self.calls = []
        self.torch = None

    def decode(self, ids):
        ids = list(ids)
        e = ids.index(256) if 256 in ids else len(ids)
        return (self.tok.decode(ids[:e]), e)

    def generate_suffix(self, prompt, answer_prefix, seed, *, total=None, gen_length=256):
        pre = self.tok.decode(answer_prefix)
        attempt = seed // 1000 if seed >= 1000 else 0
        self.calls.append((pre[-40:], attempt, gen_length))
        cont = self.script(pre, attempt, len(self.calls))
        return dict(
            ids=list(answer_prefix) + self.tok.encode(cont),
            free_start=len(answer_prefix),
            seconds=0.0,
        )


def row(prompt, *instr):
    return dict(
        key=1,
        prompt=prompt,
        instruction_id_list=[i for i, _ in instr],
        kwargs=[k for _, k in instr],
    )
