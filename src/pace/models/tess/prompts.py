import torch


def format_prompt(prompt, bos_token):
    if not isinstance(prompt, str) or not prompt.strip() or (not bos_token):
        raise ValueError("Nonempty prompt and BOS required")
    return bos_token + "<|user|>\n" + prompt.strip() + "\n<|assistant|>\n"


def encode_row(row, tokenizer, answer_tokens=512, capacity=4096):
    if type(answer_tokens) is not int or answer_tokens <= 0:
        raise ValueError("Positive integer answer capacity required")
    text = format_prompt(row["prompt"], tokenizer.bos_token)
    prefix = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if len(prefix) + answer_tokens > capacity:
        raise ValueError("Prompt plus answer exceeds validated capacity; no truncation")
    if tokenizer.pad_token_id is None:
        raise ValueError("Missing pad token")
    return dict(
        key=row["key"],
        original_prompt=row["prompt"],
        formatted_prompt=text,
        prefix_ids=prefix,
        answer_tokens=answer_tokens,
        input_ids=prefix + [tokenizer.pad_token_id] * answer_tokens,
        span_mask=[False] * len(prefix) + [True] * answer_tokens,
    )


def single_batch(encoded, device="cpu"):
    return dict(
        input_ids=torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device),
        span_mask=torch.tensor([encoded["span_mask"]], dtype=torch.bool, device=device),
    )
