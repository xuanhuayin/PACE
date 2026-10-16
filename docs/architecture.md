# Architecture

PACE follows a construct-then-certify workflow. The implementation separates the method components from model loading and benchmark execution.

## Method Components

| Operation | Implementation |
|---|---|
| Parse requirements and construct IFEval answers | [`construction/ifeval.py`](../src/pace/construction/ifeval.py), `plan`, `Program` |
| Construct IFBench answers | [`construction/ifbench.py`](../src/pace/construction/ifbench.py), `ifb_plan`, `ProgramIFB` |
| Generate model-conditioned infills and transformations | [`construction/infill.py`](../src/pace/construction/infill.py), [`construction/transforms.py`](../src/pace/construction/transforms.py) |
| Compile content-aware IFEval acceptance conditions | [`constraints/ifeval.py`](../src/pace/constraints/ifeval.py) |
| Compile IFBench acceptance conditions | [`constraints/ifbench/acceptance.py`](../src/pace/constraints/ifbench/acceptance.py) and the family registry |
| Compile finite-state token events | [`automata/context.py`](../src/pace/automata/context.py), [`automata/finite_product.py`](../src/pace/automata/finite_product.py) |
| Run joint constrained inference | [`automata/sparse_token_map.py`](../src/pace/automata/sparse_token_map.py), with dense and exhaustive reference implementations |
| Recompute scores on the constructed answer | `harmonize_batch` in each model's `engine.py` |
| Reject certification edits that damage content | `Program.guard` and [`guards/ifbench.py`](../src/pace/guards/ifbench.py) |

`Program` coordinates construction, answer-conditioned scoring, joint inference, and guarded retries. `ProgramIFB` applies this process to IFBench and checks additional instruction families. The model adapters provide sampling, tokenization, decoding, and answer-conditioned scores. Benchmark pipelines supply inputs, load compiled graphs, and serialize outputs.

## Workflow Boundaries

The command `pace run` dispatches to modules under `pace.pipelines`. Each experiment has three stages:

1. **Native:** generate unmodified model answers and capture prediction records.
2. **Compile:** build finite-state graphs from the structured requirements and selected acceptance predicates.
3. **Generate:** construct answer content, recompute scores, jointly certify the retained predicates, and apply comparative guards.

The launcher shares row order, checkpoint settings, graph paths, and output paths across stages. `pace run --dry-run` prints the selected module and relevant configuration. It does not load model weights or create output directories.

The default execution path performs certification in process. `pace.services.certification` is an optional CPU worker for the TESS IFBench pipeline. `pace.evaluation.oracle` supplies the same official IFBench checks through a subprocess when a separate verifier environment is needed.

## Evaluation and Artifacts

`pace evaluate` reads response records and reports official strict and loose accuracy. Missing responses remain in the denominator. A PACE record overrides the native answer even when the PACE answer is empty. The diagnostic modules also report supported content checks. `pace audit` aggregates independently supplied audit labels into fulfillment estimates and intervals.

Model weights, generated answers, logits, and compiled graphs are stored outside `src/pace`. Public verifier snapshots remain in `assets/` with their upstream notices. Downloaded benchmark prompts and NLTK resources are ignored by Git. `pace.paths.PROJECT_ROOT` centralizes asset resolution; `PACE_PROJECT_ROOT` can point to another checkout containing the same asset layout.

Output record fields and existing algorithm configuration names remain compatible with previously generated corpora. Internal diagnostic prefixes are serialized identifiers, while source modules use names based on their responsibilities.

## Tests

`tests/unit` checks exact inference against exhaustive enumeration, evaluation denominators, audit aggregation, workflow routing, and the TESS capture integrity guard. `tests/regression` exercises construction and comparative guards with synthetic responses and model fixtures. Neither suite requires model weights. The public assets must be downloaded for benchmark-dependent checks.
