# Running PACE

Run commands from the repository root with Python 3.12. Run `pace setup-assets` before evaluation and install the appropriate GPU requirements for inference.

## Output Layout

```text
outputs/
  <model>/
    <benchmark>/
      native/    # Native answers and prediction records
      graphs/    # Compiled automata
      pace/      # Constructed answers and certification diagnostics
```

Keep `--checkpoint` and `--work-dir` consistent across stages. LLaDA-1.5 shares the LLaDA adapter, so use a separate working directory.

IFEval construction is selective. A missing PACE record keeps the native answer. A present PACE record overrides native text, even if empty. A response missing from both remains an empty failure. Use `--method native` to evaluate native generation alone.

Core records include `key` and `response`. Other fields retain scores, token IDs, construction diagnostics, and certification/fallback status. Keep configuration-specific outputs separate when resuming work. Underlying scripts expose stage-specific options through `--help`.

The launcher fixes public row order and uses in-process certification. External oracle servers or background certification pools are not needed.

## Content-Check Diagnostics

Content diagnostic commands resolve relative directories from the repository root. Absolute paths also work. The first component in a merged specification has priority.

```bash
# IFEval holdout
KEYS_FILE=configs/ifeval_heldout_keys.txt python -m pace.evaluation.content 'native=outputs/llada/ifeval/native:response' 'pace=outputs/llada/ifeval/pace:response|outputs/llada/ifeval/native:response'

# Full IFBench
python -m pace.evaluation.ifbench_content 'native=outputs/llada/ifbench/native:response' 'pace=outputs/llada/ifbench/pace:response|outputs/llada/ifbench/native:response'
```

The IFEval diagnostic merger ignores empty strings when selecting between directories. Use `pace evaluate` for official reporting that preserves an explicitly empty answer as a failure.

## Troubleshooting

* Missing NLTK words/CMUdict: run `pace setup-assets`. Lexical guards need these resources.
* IFEval dependency mismatch: use pinned CPU requirements. Scoring checks package versions and resource hashes.
* TESS imports: run `pace setup-assets --tess` and install GPU requirements.
* LLaDA2 model loading: use its separate requirements/environment.
* Unavailable graph: inspect unsupported/reduced predicate lists and retain fallback outputs in evaluation.
* Memory: compilation and joint inference can require substantial CPU RAM in addition to GPU memory.
