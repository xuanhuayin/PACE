from contextlib import contextmanager
from types import MethodType
import torch


def additive_mask(simplex, dtype, key_mask=None):
    if simplex.ndim != 3 or not simplex.is_floating_point():
        raise ValueError("Expected floating B,L,V simplex")
    batch, length, _ = simplex.shape
    if key_mask is None:
        key_mask = torch.ones(batch, length, device=simplex.device, dtype=torch.bool)
    if key_mask.dtype != torch.bool or key_mask.shape != (batch, length):
        raise ValueError("Expected boolean B,L key-validity mask")
    if key_mask.device != simplex.device or not key_mask.any(-1).all():
        raise ValueError("Device mismatch or all-masked keys")
    mask = torch.zeros(batch, 1, length, length, device=simplex.device, dtype=dtype)
    return mask.masked_fill(~key_mask[:, None, None, :], torch.finfo(dtype).min)


@contextmanager
def bidirectional_attention(model):
    if getattr(model, "_pace_bidirectional_active", False):
        raise ValueError("Nested adapter")
    if model.config.is_causal or model.config._attn_implementation not in ("eager", "sdpa"):
        raise ValueError("Only noncausal eager/SDPA full-sequence models supported")
    original = model.forward
    had_local = "forward" in model.__dict__
    local = model.__dict__.get("forward")

    def forward(this, *args, **kwargs):
        if args or "simplex" not in kwargs:
            raise ValueError("Use native keyword-call interface")
        simplex = kwargs["simplex"]
        window = getattr(this.config, "sliding_window", None)
        if window is not None and simplex.shape[1] > window:
            raise ValueError("Sliding-window regime not validated")
        if kwargs.get("use_cache") or kwargs.get("past_key_values") is not None:
            raise ValueError("Cached decoding not supported")
        kwargs["attention_mask"] = additive_mask(
            simplex, this.get_input_embeddings().weight.dtype, kwargs.get("attention_mask")
        )
        return original(**kwargs)

    model._pace_bidirectional_active = True
    model.forward = MethodType(forward, model)
    try:
        yield model
    finally:
        if had_local:
            model.forward = local
        else:
            del model.forward
        del model._pace_bidirectional_active
