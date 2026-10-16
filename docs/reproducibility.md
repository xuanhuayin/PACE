# Reproducibility

## Included

PACE inference adapters for five backbones, IFEval/IFBench programs, acceptance predicates, automata, joint inference, comparative guards, pinned evaluator snapshots, public asset manifests, third-party notices, held-out IFEval keys, automatic evaluation, audit aggregation, and CPU tests are included. Rounded manuscript results and table-source hashes are in `results/`.

## Supplied Separately

Download weights and benchmark data from upstream sources. `pace setup-assets` fetches public prompts and NLTK resources. No weights are committed.

Existing baseline repositories and raw private labeling infrastructure are not bundled. The main execution path is PACE and native generation. Private historical audit labels and participant records are excluded. Reported audit-adjusted results can be inspected in `results/`, but cannot be recomputed from historical labels using this release alone. The audit utility supports independent labels with the same aggregation definition.

## Determinism and Validation

IFEval scoring pins evaluator code, package versions, sentence-tokenizer resources, and language-detector behavior. Source integrity is recorded in `MANIFEST.sha256`. Generation also depends on model revision, tokenizer, precision, device software, seed, graph budgets, and fallback policy.

CPU CI checks exact finite-path inference and synthetic construction/guard behavior. It does not run the full GPU benchmark matrix. Preparing this release is not a new end-to-end reproduction of all experiments.

After staging changed source files, regenerate the versioned-source manifest:

```bash
git add .
python scripts/check_integrity.py --update
git add MANIFEST.sha256
python scripts/check_integrity.py
```
