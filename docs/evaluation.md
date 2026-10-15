# Evaluation and Auditing

## Automatic Metrics

`evaluate.py` retains every request in the selected population, including missing records and empty answers. It reports official prompt/instruction accuracy in strict and loose modes. IFEval defaults to the 341 held-out prompts in `configs/ifeval_heldout_keys.txt`; `--split all` includes all 541 prompts. IFBench always uses all 300 prompts.

Content-check diagnostics apply supported predicates to official passes. Guard/detector diagnostics are method-specific checks, not substitutes for independent fulfillment audits.

## Fulfillment Audit

An audit unit contains a prompt, response, and one instruction index. Repeated types with different arguments remain separate units. Sample uniformly among official instruction passes within each arm. Hide system identities and method assignments. Identical items may be labeled once, but every retained draw keeps its method assignment.

The reference manuscript uses two Claude Opus 5.5 auditor agents and opposite presentation orders. Labels are:

* **GENUINE:** meaningful, relevant compliance with the audited instruction.
* **LOOPHOLE:** a formal pass relying on a technicality, empty structure, padding, duplication, echo, or meaningless insertion.
* **GARBLED:** corrupted or incoherent text for which compliance is not substantive.

The original shared guide appears in [audit_rubric.md](audit_rubric.md). It is a rubric, not a reconstruction of complete agent system prompts or undocumented platform settings. Whole-prompt usefulness is rated from 1 to 5 separately.

For each retained draw, average the two binary Genuine indicators. Multiply their mean by the exact official instruction pass rate. `scripts/audit_metrics.py` independently resamples draws within each arm with official counts fixed. Intervals describe audit-sampling uncertainty for a fixed corpus and supplied judgments.

## Input Schema

Use `examples/audit_fixture.json` as a synthetic example. Each arm has exact `official_passes`, `official_total`, and a `draws` list. Every draw has two `labels`, such as `["GENUINE", "LOOPHOLE"]`. Optional `comparisons` contains pairs such as `["pace", "native"]`.

```bash
python scripts/audit_metrics.py your-audit.json --resamples 5000 --seed 0 --output reports/audit.json
```

Preserve retained draws, including shared-item occurrences, after assigning labels. Exclude bridge-only draws from estimates. Match audited responses to the same official output version. The 5,000-resample percentile endpoint convention matches the paper. Reproducing historical intervals additionally requires the historical labels, arm order, comparison order, and random stream.
