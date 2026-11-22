# Original October logs

Working logs written while the experiments ran (Chinese). They record every run, including dead ends and in-place corrections; later sections override earlier ones where they say so.

| File | Content |
|---|---|
| `depth_intervention_PLAN.md` | Round-1 plan: data contract, class definitions, pre-registered checks |
| `depth_intervention_RESULTS.md` | Round-1 log, Oct 16–17: extraction, probes, taxonomy, early exit, loop, label-free features (sections 8–15) |
| `gen_depth_RESULTS.md` | Generation log, Oct 18–19: decoder, depth profiles, early exit, loop, perfect-classifier test (section 16) |
| `summary_2026-10-22.md` | Condensed list of data and experiments across both rounds |

Known corrections to these files:

- **Final-layer rule.** Round-1 question-level numbers before section 9 use the probe at the last layer. The official rule since the evening of Oct 16 uses the real output there (`--last_layer output`); section 9 supersedes the question-level numbers of sections 5 and 7.
- **"Completely untrained".** The label-free exit signal (L0) combines model-internal readouts through a small logistic regression trained on the model's own full-depth output. It needs no gold labels, but it is trained.
- **Single-version O false positives.** The 2026-10-22 summary quotes 57–77% for the original paper's single-version O. The 2026-10-14 re-check could not reproduce that range; under the same rule it found 62% (Qwen3-4B-Base) and 45% (Qwen3.5-9B-Base).
