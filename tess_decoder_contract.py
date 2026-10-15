import json
import re
DECODER = {'type': 'Sequence', 'decoders': [{'type': 'Replace', 'pattern': {'String': '▁'}, 'content': ' '}, {'type': 'ByteFallback'}, {'type': 'Fuse'}, {'type': 'Strip', 'content': ' ', 'start': 1, 'stop': 0}]}
BYTE = re.compile('<0x([0-9a-fA-F]{2})>')

class TessDecoderContract:

    def __init__(self, tokenizer):
        config = json.loads(tokenizer.backend_tokenizer.to_str())
        if config['decoder'] != DECODER or config['model']['type'] != 'BPE' or (not config['model']['byte_fallback']):
            raise ValueError('Unsupported decoder; do not assume token concatenation')
        if len(tokenizer) != 32001 or set(tokenizer.all_special_ids) != {0, 1, 2, 32000} or tokenizer.eos_token_id != 2:
            raise ValueError('Pinned TESS vocabulary/special tokens required')
        self.pieces = tuple((tokenizer.convert_ids_to_tokens(i).replace('▁', ' ') for i in range(len(tokenizer))))
        self.special = frozenset(tokenizer.all_special_ids)
        self.eos = tokenizer.eos_token_id
        self.bytes = tuple((int(m.group(1), 16) if (m := BYTE.fullmatch(piece)) else None for piece in self.pieces))
        if set((v for v in self.bytes if v is not None)) != set(range(256)):
            raise ValueError('Byte vocabulary is incomplete')

    def decode(self, ids):
        if not isinstance(ids, (list, tuple)) or any((type(i) is not int or not 0 <= i < len(self.pieces) for i in ids)):
            raise ValueError('Valid integer token IDs required')
        chunks = []
        pending = []

        def flush():
            if pending:
                try:
                    text = bytes(pending).decode('utf-8', errors='strict')
                except UnicodeDecodeError:
                    text = '�' * len(pending)
                chunks.append(text)
                pending.clear()
        for token in ids:
            if token == self.eos:
                break
            if token in self.special:
                continue
            byte = self.bytes[token]
            if byte is not None:
                pending.append(byte)
            else:
                flush()
                chunks.append(self.pieces[token])
        flush()
        text = ''.join(chunks)
        return text[1:] if text.startswith(' ') else text
