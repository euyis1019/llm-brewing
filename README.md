# From Brewing to Resolution: Tracing the Internal Lifecycle of Code Reasoning in LLMs

Official implementation of **"From Brewing to Resolution: Tracing the Internal Lifecycle of Code Reasoning in LLMs"**.

[[arXiv]](#) &nbsp; [[Paper]](#) &nbsp; [[Project Page]](#)

by Siyue Chen\*, Yifu Guo\*, Yuquan Lu, Zishan Xu, Jiaye Lin, Jianbo Lin, Siyu Zhang, Cheng Yang, Junxin Li, Yujia Li, Yu Huo, Ruixuan Wang†

<sub>\*Equal contribution &nbsp;&nbsp; †Correspondence</sub>

> ⚠️ **Status.** The codebase is **under active refactoring**. Everything here already runs end-to-end, but the interfaces are being reworked into a cleaner, general-purpose framework — **we recommend waiting for the refactored release** if you plan to build on top of it. The current code reproduces all experiments in the paper.

---

## Introduction

![teaser](assets/teaser.png)

Standard accuracy hides more than it reveals. A model that traces variable assignments effortlessly can fall apart on a semantically identical loop — yet both score the same on the surface. We argue that code reasoning inside an LLM is not a last-layer event but an **internal lifecycle** unfolding across depth, and that the interesting differences live in *how* the answer is computed, not just *whether* it is.

We study this lifecycle with a **dual diagnostic lens**:

- **Linear Probing** — *information availability*: is the answer already linearly readable from the hidden state?
- **Context-Stripped Decoding (CSD)** — *information readiness*: can the model itself decode that answer from its own representation?

These two views rarely fire at the same time, and the gap between them is the heart of the story.

### Finding 1 — Models *brew* before they *resolve*

The answer becomes **externally readable** (probing turns correct) many layers before it becomes **self-decodable** (CSD catches up). We call this intermediate stretch **brewing**: the answer is *present* in the representation but not yet in a form the model can act on. Code reasoning has a structured middle, not just an input and an output.

### Finding 2 — Four resolution outcomes, all of them common

Once brewing completes (or fails to), trajectories diverge into four outcomes — illustrated in the four panels of the teaser above:

| Outcome | What happens internally |
| --- | --- |
| **Resolved** | The answer is jointly attested and survives to the output. |
| **Overprocessed** | A correct answer *forms*, then later layers overwrite it — "right, then wrong." |
| **Misresolved** | The model never reaches a joint-correct state but commits confidently to a wrong answer. |
| **Unresolved** | The computation never finishes within the available depth. |

The point is that **none of these is a rare edge case** — every task family carries substantial mass in all four. Two models with the same accuracy can be failing in completely different ways underneath, and surface metrics can't tell them apart.

### Finding 3 — Different code primitives fail in different ways

![task fingerprint](assets/task_fingerprint.png)

When we sweep structure, depth, and operators, each task reveals its *own* bottleneck rather than a shared difficulty axis. Function-call reasoning, for instance, collapses sharply as call depth grows, while value tracking stays robust. The outcome distribution acts like a **fingerprint** of how a primitive is processed — and it explains task-level puzzles (e.g. why explicit loops beat their unrolled equivalents) that accuracy alone leaves mysterious.

We confirm with **causal interventions** — activation patching at the brewing-completion layer, layer-skipping for Overprocessed, and re-injecting early representations for Unresolved — that these outcomes are intervention-sensitive computational states, not post-hoc labels. Notably, the two opposing failure modes pull in opposite directions: Overprocessed wants *less* depth, Unresolved wants *more* — so no single early-exit rule can serve both.

### Finding 4 — A stable scaffold across architectures and scales

![brewing stability](assets/brewing_stability.png)

Across 16 models spanning the Qwen, Llama, and DeepSeek families, the **brewing scaffold is remarkably stable**: the answer always becomes readable well before it becomes decodable, and the normalized brewing duration stays in a tight band regardless of scale or family. What *changes* with capability, scale, and training is the **resolution success** — how often brewing actually completes into a correct, preserved answer. The scaffold looks like a structural regularity of decoder-only Transformers; getting to a good answer is what capability buys you.

## What's in this repo

This repository contains the **inference and evaluation** code: the dual-diagnostic framework (Probing + CSD), the six-family code-reasoning benchmark, the outcome taxonomy, and the figure-generation scripts.

```
brewing/                 # core framework
├── orchestrator.py      #   dataset → hidden-state cache → method runs
├── methods/             #   linear_probing.py, csd.py
├── diagnostics/         #   FPCL / FJC / outcome classification
└── benchmarks/cue_bench/#   six code-reasoning task families
figures/scripts/         # reproduce every paper figure from released data
scripts/                 # smoke tests and pipeline entry points
```

The six task families span **data flow** (value tracking, computing), **control flow** (conditional, function call), and their **combination** (loop, loop-unrolled), each swept over structure, depth, and operators.

## Roadmap

- [ ] **Framework refactor** — rework the framework into a clean, general-purpose layer-wise interpretability toolkit (hot-swappable dataset / method / diagnostic slots). *In progress.*
- [ ] **Scaling experiments** — extend the analysis to larger, state-of-the-art open-source models to stress-test how far the brewing-to-resolution story generalizes and to harden the robustness of the conclusions. *In progress.*
- [ ] Release trained probes and precomputed diagnostic data.
- [ ] arXiv release.

## Installation & Usage

```bash
pip install -e .            # add [model] for torch/transformers/nnsight backends
python scripts/test_e2e_smoke.py   # end-to-end smoke test
```

Experiments are driven by YAML config (one per model); the orchestrator runs dataset build → hidden-state cache → diagnostic methods, with outcome classification as a decoupled post-processing stage.

## Citation

```bibtex
@misc{brewing2026,
  title={From Brewing to Resolution: Tracing the Internal Lifecycle of Code Reasoning in LLMs},
  author={Chen, Siyue and Guo, Yifu and Lu, Yuquan and Xu, Zishan and Lin, Jiaye and
          Lin, Jianbo and Zhang, Siyu and Yang, Cheng and Li, Junxin and Li, Yujia and
          Huo, Yu and Wang, Ruixuan},
  year={2026},
  note={Under review},
}
```
