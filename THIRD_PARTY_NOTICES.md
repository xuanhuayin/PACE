# Third-Party Notices

The root Apache License 2.0 applies to original PACE code. Third-party files retain their upstream notices. Public data and weights remain subject to their own terms.

| Artifact | Source | Distribution | Terms |
|---|---|---|---|
| IFEval evaluator | [Google Research](https://github.com/google-research/google-research/tree/e6890f85757dd84e27ca6df2dd30651dafad28e0/instruction_following_eval) | Verifier included, public data fetched | Apache License 2.0 in [`assets/ifeval/LICENSE`](assets/ifeval/LICENSE) |
| IFBench evaluator | [AllenAI IFBench](https://github.com/allenai/IFBench) | Flat-package verifier adaptation included, public data fetched | Apache License 2.0 for evaluation code in [`assets/ifbench/LICENSE`](assets/ifbench/LICENSE) |
| TESS implementation | [TESS-2](https://github.com/hamishivi/tess-2) | Fetched with `setup_assets.py --tess` | Upstream MIT license fetched with the source |
| NLTK resources | [NLTK data](https://github.com/nltk/nltk_data) | Downloaded during setup | Respective upstream resource terms |
| Model weights | Model publishers | Not included | Respective checkpoint terms |

Exact benchmark revisions and hashes appear in `assets/ifeval/sources.json`, `assets/ifbench/sources.json`, and `assets/scoring_assets.json`. The IFBench snapshot retains the flat-package import adaptation used for evaluation.
