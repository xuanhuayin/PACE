# Changelog

## 0.2.0 — 2026-10-16

* Organize the implementation as an installable `src/pace` package, with separate construction, constraints, automata, guards, models, pipelines, and evaluation modules.
* Replace development-stage filenames with descriptive module names and move tests into unit, regression, and fixture directories.
* Add the `pace` CLI, retain the three top-level command wrappers, and provide workflow dry runs.
* Centralize repository paths and update dynamic imports and the IFBench oracle subprocess for package execution.
* Document the algorithm-to-code mapping and validate unchanged official scores on three existing output corpora.

## 0.1.0 — 2026-10-16

Initial public implementation release. Includes five backbone interfaces, IFEval/IFBench programs and verifiers, unified official evaluation, audit aggregation, pinned source manifests, CPU CI, documentation, and rounded manuscript results.
