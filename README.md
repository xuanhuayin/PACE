# PACE

**Certified Yet Unfulfilled: Construct-Then-Certify Instruction Following with Diffusion Language Models**

[![CPU tests](https://github.com/xuanhuayin/PACE/actions/workflows/cpu-tests.yml/badge.svg)](https://github.com/xuanhuayin/PACE/actions/workflows/cpu-tests.yml)
[![Python](https://img.shields.io/badge/Python-3.12-70445A)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-C65A8B)](LICENSE)

PACE is a training-free framework for instruction following with diffusion language models. It constructs task content, recomputes token scores on that answer, jointly enforces compiled acceptance conditions, and checks certification edits against the constructed content.

**Certified is not fulfilled.** A formal pass can still rely on empty structures, repetition, or meaningless insertions. PACE connects content construction to exact inference over the retained finite-state predicates.

```mermaid
flowchart LR
    X["Request + structured requirements"] --> C["Construct answer"]
    C --> S["Recompute model scores"]
    X --> A["Compile acceptance conditions"]
    S --> J["Joint certification"]
    A --> J
    J --> G["Comparative edit guards"]
    C --> G
    G -->|admit| Y["Return certified answer"]
    G -->|reject| R["Penalize edits and retry"]
    R --> J
    R -->|budget exhausted| F["Return constructed answer"]
    classDef pink fill:#F9E8F0,stroke:#C65A8B,color:#302A32;
    classDef blue fill:#EAF1F7,stroke:#718FA5,color:#302A32;
    class C,G,Y pink;
    class S,A,J blue;
```

## Contents

| Component | Entry points |
|---|---|
| Native generation, compilation, PACE | [`run.py`](run.py) |
| Official benchmark evaluation | [`evaluate.py`](evaluate.py) |
| IFEval construction and guards | [`cah/program_gen4.py`](cah/program_gen4.py) |
| IFBench construction and guards | [`cah/program_ifb4.py`](cah/program_ifb4.py), [`cah/ifb4_guards.py`](cah/ifb4_guards.py) |
| Acceptance predicates | [`cah/predicates_hard.py`](cah/predicates_hard.py), [`cah/predicates_ifb_hard.py`](cah/predicates_ifb_hard.py) |
| Automaton compilation and inference | [`cah/ctx_v2.py`](cah/ctx_v2.py), [`cah/fdp_v2.py`](cah/fdp_v2.py), root automaton modules |
| Model interfaces | `cah/*_engine.py`, [`cah/cah_tess.py`](cah/cah_tess.py) |
| Audit-adjusted fulfillment | [`scripts/audit_metrics.py`](scripts/audit_metrics.py) |
| Reported results | [`results/`](results/) |

## Installation

Use **Python 3.12**. Some files use syntax introduced in Python 3.12.

```bash
git clone https://github.com/xuanhuayin/PACE.git
cd PACE
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python setup_assets.py
```

The setup script fetches public benchmark prompts and NLTK resources. It verifies pinned benchmark data and IFEval sentence-tokenizer hashes. Verifier source snapshots and third-party licenses are included in `assets/`.

For GPU inference, choose one environment:

```bash
# TESS-2, LLaDA, LLaDA-1.5, or Dream
python -m pip install -r requirements-gpu.txt

# LLaDA2.0-mini: use a separate environment
python -m pip install -r requirements-llada2.txt
```

The experiments used NVIDIA GeForce RTX 5090 GPUs with PyTorch 2.8.0 and CUDA 12.8. Install a compatible PyTorch build for your CUDA environment. Compilation and certification also use CPU resources. Memory and runtime depend on graph size, answer length, and backbone.

TESS-2 additionally requires its pinned public implementation:

```bash
python setup_assets.py --tess
```

Model weights are obtained separately or supplied through `--checkpoint`.

## Quick Start

Run **native → compile → generate → evaluate** in order:

```bash
python run.py --model llada --benchmark ifeval --stage native
python run.py --model llada --benchmark ifeval --stage compile -- --workers 6
python run.py --model llada --benchmark ifeval --stage generate
python evaluate.py --model llada --benchmark ifeval --output reports/llada-ifeval.json
```

The default IFEval report uses the **341-prompt holdout with 516 instructions**. Generation and compilation retain the full public prompt order. Development prompts are excluded when held-out results are summarized. Use `--split all` for all 541 prompts.

For IFBench:

```bash
python run.py --model tess --benchmark ifbench --stage native
python run.py --model tess --benchmark ifbench --stage compile -- --workers 6
python run.py --model tess --benchmark ifbench --stage generate
python evaluate.py --model tess --benchmark ifbench --output reports/tess-ifbench.json
```

IFBench uses all **300 prompts and 344 instructions**.

### Supported Backbones

| Backbone | `--model` | Default checkpoint / usage |
|---|---|---|
| TESS-2 | `tess` | `hamishivi/tess2` |
| LLaDA | `llada` | `GSAI-ML/LLaDA-8B-Instruct` |
| LLaDA-1.5 | `llada` | Supply its checkpoint through `--checkpoint` |
| LLaDA2.0-mini | `llada2` | `inclusionAI/LLaDA2.0-mini` |
| Dream | `dream` | `Dream-org/Dream-v0-Instruct-7B` |

For LLaDA-1.5, pass the same checkpoint and a separate working directory to every stage:

```bash
python run.py --model llada --checkpoint checkpoints/llada-1.5 --work-dir outputs-llada15 --stage native
python run.py --model llada --checkpoint checkpoints/llada-1.5 --work-dir outputs-llada15 --stage compile
python run.py --model llada --checkpoint checkpoints/llada-1.5 --work-dir outputs-llada15 --stage generate
python evaluate.py --model llada --work-dir outputs-llada15 --output reports/llada15-ifeval.json
```

Replace the example checkpoint path with a real model directory or public Hugging Face identifier. For the original TESS-2 snapshot, obtain revision `684af5623534d9163509df1d37b77412b0737671` and pass the local snapshot through `--checkpoint`.

Additional script arguments follow `--`. Inspect stage-specific options with `python cah/<script>.py --help`. See [running details](docs/running.md) for outputs and fallback behavior.

## Evaluation

Official acceptance and substantive fulfillment are separate quantities:

* `evaluate.py` reports strict/loose prompt and instruction accuracy, exact counts, missing records, and empty answers. Missing responses remain in the denominator. An explicitly empty PACE answer remains a failure.
* The existing `score_hard` tools additionally report supported content predicates and detector diagnostics.
* Audit-estimated fulfillment multiplies official instruction accuracy by the mean of two binary Genuine judgments among sampled official passes. This requires actual audit labels.

```bash
python evaluate.py --model llada --benchmark ifeval --method native

# Synthetic audit schema example, not experimental results
python scripts/audit_metrics.py examples/audit_fixture.json --output reports/audit-example.json
```

See [evaluation and auditing](docs/evaluation.md) for the audit schema and confidence intervals. Private audit labels, participant identities, weights, and experiment logs are not distributed.

## Reported Results

Audit-estimated fulfillment, in percent, from the reference manuscript:

| Backbone | IFEval Native | IFEval PACE | IFBench Native | IFBench PACE |
|---|---:|---:|---:|---:|
| TESS-2 | 23.1 | **55.0** | 6.1 | **31.5** |
| LLaDA | 55.3 | **73.4** | 9.3 | **24.1** |
| LLaDA-1.5 | 58.3 | **71.3** | 12.9 | **30.5** |
| LLaDA2.0-mini | 67.6 | **81.6** | 22.4 | **37.2** |
| Dream | 52.1 | **68.6** | 11.4 | **28.2** |

The [machine-readable table](results/paper_results.csv) includes matched controls, official accuracy, Genuine shares, and fulfillment intervals. Values are rounded manuscript results with [source provenance](results/provenance.json), not new release runs. Comparisons use the same backbone. PACE does not exceed direct certification in every backbone/benchmark combination, as the LLaDA-1.5 IFEval rows show.

## Tests

After CPU installation and asset setup:

```bash
python scripts/check_integrity.py
python -m unittest test_branched_token_map_v1 test_branched_token_map_sparse_v1 test_tess_branched_quotient_v1
python -m unittest discover -s tests -v
python cah/test_program_gen4.py
python cah/test_program_ifb4.py
```

Solver tests compare inference against exhaustive finite-path enumeration. Construction and guard tests use synthetic fixtures without weights. Release-tool tests cover denominator preservation, empty answers, duplicate keys, shared audit draws, and ranking reversals. GitHub Actions runs the CPU suite. GPU inference is outside this CI job.

## Reproducibility and Scope

Included: PACE inference, all five backbone interfaces, both benchmark evaluators, CPU tests, and audit aggregation. Training is not required. Published baseline repositories and private annotation infrastructure are not bundled. See [reproducibility](docs/reproducibility.md) for exact boundaries.

## License and Attribution

Original PACE code is provided under the [Apache License 2.0](LICENSE). Third-party files retain their upstream notices. Benchmark data and model checkpoints remain subject to upstream terms. See [third-party notices](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@misc{yin2026pace,
  title  = {Certified Yet Unfulfilled: Construct-Then-Certify Instruction Following with Diffusion Language Models},
  author = {Yin, Xuanhua and Wei, Boyu and Mao, Shunqi and Xu, Chuanzhi and Cai, Weidong},
  year   = {2026},
  url    = {https://github.com/xuanhuayin/PACE}
}
```

[CITATION.cff](CITATION.cff) also supports GitHub's citation interface.
