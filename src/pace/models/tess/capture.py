import ast
import hashlib
import inspect
from pathlib import Path
import textwrap
import time
import torch
import pace.models.tess.native as native

NATIVE_SHA = "5411b717797d1eef51b8e322ee7a2045b0adfacb4ace1551bfee03c36fb3bfce"


def capture_return(record, logits):
    started = time.perf_counter()
    if (
        logits.ndim != 3
        or logits.shape[0] != 1
        or (not logits.is_floating_point())
        or (not torch.isfinite(logits).all())
    ):
        raise ValueError("Finite single-example native logits required")
    cpu = logits.detach().cpu().clone()
    digest = native.tensor_hash(cpu)
    prefix = len(record["prefix_ids"])
    if (
        digest != record["final_logits_sha256"]
        or not 0 < prefix < cpu.shape[1]
        or cpu.shape[1] - prefix != len(record["answer_ids"])
        or (cpu[0, prefix:].argmax(-1).tolist() != record["answer_ids"])
    ):
        raise ValueError("Captured native final-logit/ID identity differs")
    return dict(
        native=record,
        final_logits=cpu,
        final_logits_sha256=digest,
        capture_seconds=time.perf_counter() - started,
        capture_cost_scope="Additional final CPU copy and identity checks after unchanged native timer",
    )


def transform(source):
    tree = ast.parse(textwrap.dedent(source))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("One native generation function required")
    function = tree.body[0]
    returns = [n for n in ast.walk(function) if isinstance(n, ast.Return)]
    if (
        function.decorator_list
        or len(returns) != 1
        or returns[0] is not function.body[-1]
        or (not isinstance(returns[0].value, ast.Call))
        or (ast.unparse(returns[0].value.func) != "dict")
    ):
        raise ValueError("Exactly one final native dictionary return required")
    returns[0].value = ast.Call(
        func=ast.Name(id="_pace_capture_final_return", ctx=ast.Load()),
        args=[
            returns[0].value,
            ast.Attribute(
                value=ast.Name(id="output", ctx=ast.Load()), attr="logits", ctx=ast.Load()
            ),
        ],
        keywords=[],
    )
    ast.fix_missing_locations(tree)
    return tree


def build():
    if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() != NATIVE_SHA:
        raise ValueError("Native generation source drift")
    tree = transform(inspect.getsource(native.generate))
    scope = dict(native.generate.__globals__, _pace_capture_final_return=capture_return)
    exec(compile(tree, "<pace_native_capture>", "exec"), scope)
    return scope["generate"]


generate = build()
