from pace.paths import PROJECT_ROOT
import hashlib, json, os
from pathlib import Path

ROOT = PROJECT_ROOT
INPUT = Path(os.environ.get("ROWS_FILE", ROOT / "assets/ifeval/data/input_data.jsonl"))
REVISION = "684af5623534d9163509df1d37b77412b0737671"
INPUT_SHA = "67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49"
CORE = {
    "transformers": "4.46.3",
    "tokenizers": "0.20.3",
    "huggingface-hub": "0.36.2",
    "diffusers": "0.32.2",
    "peft": "0.13.2",
    "torch": "2.8.0+cu128",
}
SETTINGS = dict(
    total_positions=2048,
    steps=100,
    seed_base=202609281000,
    preflight_seed=202609289999,
    preflight_prompt="Write one sentence explaining why leaves are green.",
    revision=REVISION,
    core=CORE,
    batch_size=1,
    guidance_scale=0.0,
    response_policy="Answer argmax before first EOS; skip special tokens; no cleanup or strip.",
)


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def rows():
    if sha(INPUT) != INPUT_SHA:
        raise ValueError("Official input changed")
    data = [json.loads(line) for line in INPUT.read_text().splitlines()]
    if (
        len(data) != 541
        or len({r["key"] for r in data}) != 541
        or len({r["prompt"] for r in data}) != 541
    ):
        raise ValueError("Population differs")
    return data


def encode(row, tokenizer, total=2048):
    from pace.models.tess.prompts import encode_row, format_prompt

    prefix = tokenizer.encode(
        format_prompt(row["prompt"], tokenizer.bos_token),
        add_special_tokens=False,
        truncation=False,
    )
    if type(total) is not int or total > 4096 or len(prefix) >= total:
        raise ValueError("Invalid total capacity; no prompt truncation")
    result = encode_row(row, tokenizer, total - len(prefix))
    if result["prefix_ids"] != prefix or len(result["input_ids"]) != total:
        raise ValueError("Prefix or length contract differs")
    return result


def decode(ids, tokenizer):
    if (
        not isinstance(ids, list)
        or not ids
        or any((type(v) is not int or not 0 <= v < len(tokenizer) for v in ids))
    ):
        raise ValueError("Invalid answer IDs")
    eos = ids.index(tokenizer.eos_token_id) if tokenizer.eos_token_id in ids else len(ids)
    return (
        tokenizer.decode(ids[:eos], skip_special_tokens=True, clean_up_tokenization_spaces=False),
        eos,
    )
