# Brewing — `unsupervised-metrics` branch

[中文版](README_CN.md)

Brewing studies when an LLM's answer forms across depth. Two readouts bracket it at every layer: a cross-validated linear probe tells whether the answer is already **externally decodable** from the hidden state, and Patchscope-style context-stripped decoding (CSD) tells whether the model's own remaining layers can already decode it (**self-decodable**). The answer reaches the first condition several layers before the second.

**Emission layer.** Each model has one layer at which the answer is written into the output-readable direction in a single step. Before it, a probe can already read the answer, but the answer lies outside the four letter-readout directions; at it, the model's own readout jumps to the answer. The layer is fixed per model: it does not move with the question source, the outcome class, or post-training (Qwen3-4B L24, Qwen2.5-7B L19–20, Llama-3.1-8B L17).

This branch holds the code and data for the October 2026 follow-up to the EACL paper ([arXiv 2606.17648](https://arxiv.org/abs/2606.17648)). The paper used six synthetic code-reasoning tasks; this round moves to cross-domain multiple-choice benchmarks, asks whether label-free signals can replace the trained probe, and tests whether those signals can drive early exit and layer looping. The scripts are the ones that produced the October numbers, with machine paths made configurable. How we got here is in [docs/TRAJECTORY.md](docs/TRAJECTORY.md).

## Results in one table

Answer position, 5,659 questions × 4 cyclic option permutations. A class counts only when all four permutations agree on the same content.

| Model, prompt | Accuracy | R | O | U | M | NB | Probe-readable (median layer) | Self-decodable (median layer) |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B-Base, 1-shot | 66.1% | 47.4% | 1.4% | 4.2% | 0.1% | 46.8% | 21 | 24 |
| Qwen3-4B, 1-shot | 59.1% | 37.6% | 0.5% | 5.4% | 0.0% | 56.4% | 21 | 24 |
| Qwen3-4B, chat | 56.8% | 32.1% | 0.3% | 8.1% | 0.0% | 59.5% | 21 | 24 |
| Qwen2.5-7B, 1-shot | 62.5% | 43.8% | 0.8% | 4.2% | 0.2% | 51.1% | 19 | 20 |
| Qwen2.5-7B-Instruct, chat | 58.6% | 37.5% | 0.4% | 4.7% | 0.0% | 57.3% | 19 | 19 |
| Llama-3.1-8B, 1-shot | 41.0% | 14.8% | 0.9% | 6.7% | 0.0% | 77.5% | 15 | 17 |

R: self-decodable and correct at the output. O: was self-decodable, lost by the output. U: probe-readable, never self-decodable. M: probe-readable, consistently wrong. NB: never probe-readable. Layer medians are over R questions.

What the interventions showed:

- **Label-free readouts work.** A probe trained on the model's own four-permutation consensus, with no gold labels, tracks the gold-label probe layer by layer within 0.6 points on the Qwen3 lines.
- **Early exit at the answer position saves depth, not accuracy.** At 99% agreement with full-depth output, exit features built from the CSD jump stop at layer 25.2 on Qwen3-4B chat (confidence alone: 30.7, oracle: 23.5). Accuracy stays within a point. No rule turns wrong answers right once matched against a decoy control.
- **Looping middle layers barely helps.** Post-trained models gain under a point; base models do not gain. No label-free gate beats looping every question.
- **In generation both actions fail.** With exit decisions that never pick a different token than full depth, MATH-500 accuracy drops from 66.4% to 27.9%. Later tokens read the exited tokens' deep keys and values, which the frozen model never saw. The pre-registered loop arm loses 3.3 points.

The bottleneck is executing the action on a frozen model, not deciding when to act.

## Layout

```
scripts/depth_intervention/   answer-position study: extraction, probes, taxonomy, early exit, loop, label-free features
scripts/gen_depth/            generation study: custom per-token-depth decoder, policies, grading, comparisons
scripts/make_subset.py        stratified question subsets for fast runs
scripts/_workspace.py         data / model path resolution
experiments/depth_intervention/data/bench/   question sets, few-shot demos, per-model few-shot settings
experiments/gen_depth/data/   generation problem pool and the fixed selection lists
vendor/gen_depth_grading/     vendored math_verify grader
docs/                         reproduction guide, research trajectory, original experiment logs
```

Runs write to `experiments/<study>/results/`, which is not tracked.

## Quick start

```bash
pip install -r requirements.txt
export BREWING_MODEL_ROOT=/path/to/models   # holds Qwen3-4B-Base/, Qwen3-4B/, Meta-Llama-3.1-8B-nous/, huggingface/Qwen/...
export BREW_BENCH=core500                   # 500-question subset; unset for the full set

cd experiments/depth_intervention
python scripts/extract.py          --model qwen3-4b-base --fmt base_fewshot
python scripts/save_lens_params.py --model qwen3-4b-base --fmt base_fewshot
python scripts/train_probe.py      --model qwen3-4b-base --fmt base_fewshot
python scripts/classify.py         --model qwen3-4b-base --fmt base_fewshot --probe full --csd raw --last_layer output --out classify
python scripts/early_exit.py       --model qwen3-4b-base --fmt base_fewshot --probe full --classify_dir classify
```

On one RTX 4090 the subset takes about three minutes per model through this point. Every step, the loop and generation studies, expected numbers and how they were checked against October are in [docs/REPRODUCTION.md](docs/REPRODUCTION.md).
