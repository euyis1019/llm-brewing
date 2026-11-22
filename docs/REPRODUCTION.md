# Reproducing the October 2026 experiments

Two studies, both run from their directory under `experiments/`:

- **Answer position** (`experiments/depth_intervention`): one forward pass per prompt on four-option multiple choice. Readouts, five-class taxonomy, label-free features, early exit and layer looping.
- **Generation** (`experiments/gen_depth`): chain-of-thought decoding with per-token depth control. Token-level depth profiles, early exit and looping inside the decoding loop.

Every number marked "October" below is read from the stored October outputs. Every number marked "core500" was produced from this branch on 2026-11-22. The original working logs, in Chinese, are in [docs/logs](logs/).

## Setup

```bash
pip install -r requirements.txt            # Python 3.11; torch 2.11 / transformers 5.13 were used
export BREWING_MODEL_ROOT=/path/to/models
```

Model keys and the directories they load from `$BREWING_MODEL_ROOT`:

| Key | Directory | Layers |
|---|---|---|
| `qwen3-4b-base` | `Qwen3-4B-Base` | 36 |
| `qwen3-4b` | `Qwen3-4B` | 36 |
| `qwen2.5-7b` | `huggingface/Qwen/Qwen2.5-7B` | 28 |
| `qwen2.5-7b-instruct` | `huggingface/Qwen/Qwen2.5-7B-Instruct` | 28 |
| `llama3.1-8b` | `Meta-Llama-3.1-8B-nous` | 32 |

To add a model, extend `MODELS` in `scripts/depth_intervention/common.py`. Prompt formats are `base_fewshot` (1-shot completion, answer after `Answer:\n`) and `chat0` (system instruction + user turn, non-thinking).

`BREWING_WORKSPACE` defaults to the repository root. Set it to keep results elsewhere; that directory needs the same `experiments/<study>/data` layout.

## Question sets

`experiments/depth_intervention/data/bench/questions.jsonl` is the full set: 5,659 questions from GSM-MC 1,319, MATH-MC 1,000, PythonIO 1,000, MMLU-Pro 1,000 (cut to four options) and C-Eval 1,340. Every question is shown in four cyclic permutations of its options (22,636 prompts), and a question counts as answered only if all four point to the same content. Questions are split into five folds by question group; thresholds are chosen on folds 0–3 and fold 4 is a lockbox.

`BREW_BENCH=<name>` switches every script to `questions_<name>.jsonl` and writes to `results/<model>__<fmt>__<name>`, so subset runs never overwrite full runs. The branch ships `core500`: 500 questions drawn per (source, fold) cell in proportion to the full set, renumbered from 0 (`orig_qid` keeps the original id). Make other sizes with

```bash
python scripts/make_subset.py --n 1000 --name core1000
```

`ext` (11,364 extra GSM, MATH and PythonIO questions) was only used to add O examples on the two Qwen3-4B lines.

## Study 1: answer position

```bash
cd experiments/depth_intervention
export BREW_BENCH=core500          # omit for the full set
M=qwen3-4b-base; F=base_fewshot     # or M=qwen3-4b F=chat0, etc.
```

| Step | Command | core500 time, one 4090 |
|---|---|---|
| Implementation checks; few-shot choice for a new model | `python scripts/smoke.py --model $M --fmt $F` | 1–2 min |
| Hidden states (block outputs at the answer position, bf16), logit lens and CSD readouts | `python scripts/extract.py --model $M --fmt $F` | 45 s |
| Final-norm weights and letter unembedding rows, needed by every CPU step | `python scripts/save_lens_params.py --model $M --fmt $F` | 15 s |
| Cross-fitted lens-anchored probes plus three permuted-label nulls | `python scripts/train_probe.py --model $M --fmt $F` | 45 s |
| Five-class taxonomy | `python scripts/classify.py --model $M --fmt $F --probe full --csd raw --last_layer output --out classify` | 3 s |
| Early-exit rules against the decoy control | `python scripts/early_exit.py --model $M --fmt $F --probe full --classify_dir classify` | 5 s |
| Label-free consensus probe, then its evaluation | `python scripts/consensus_probe.py --model $M --fmt $F` then add `--eval` | 70 s |
| Label-free features and the exit / loop-gate analysis | `python scripts/stage2_features.py ...` then `python scripts/stage2_analysis.py ...` | 1 min |
| Layer looping at the answer position | `python scripts/loop.py --model $M --fmt $F --mode last --out loop_allq` | 1 min for 3 configs |
| Loop evaluation | `python scripts/loop_eval.py --model $M --fmt $F --loop_dir loop_allq --probe full --K 3 --last_layer output --eval_out eval_lo` | 10 s |

Notes that matter for matching October:

- **Run `save_lens_params.py` after `extract.py`.** The probe and every analysis read `extract/lens_params.npz`, which extraction does not write.
- **Always pass `--last_layer output`.** It is the October contract: at the last layer the real model output replaces the probe. Some scripts still default to the earlier rule.
- **The CSV taxonomy has four classes.** `classify/questions.csv` stores R/O/U/M; the five-class report splits off NB, meaning questions whose `FPCL_c_K3` is -1, out of U and M.
- **Few-shot settings are fixed.** `data/bench/fewshot_k__<model>.json` fixes 1-shot for every model, as decided in round 1. `smoke.py` keeps an existing file and writes one only for a new model.
- **Loop configs.** `--configs` takes names such as `none,w18_25_k2,p28_33_k1`: repeat blocks 18–25 twice, versus a late-layer placebo. Leave it out to run all ten configs used in October.
- **Other analyses** are separate scripts on the same caches: `stage2_abne.py` (readout-subspace geometry and the complement push), `stage2_loopgate.py`, `stage2_retroread.py` and `steer.py`. Their October results are in `docs/logs/depth_intervention_RESULTS.md`, sections 8–15.

### Expected results

Qwen3-4B, both lines:

| | Base 1-shot, October | Base 1-shot, core500 | Chat, October | Chat, core500 |
|---|---|---|---|---|
| Single-version accuracy | 66.1% | 66.5% | 56.8% | 53.6% |
| R / O / U / M / NB | 47.4 / 1.4 / 4.2 / 0.1 / 46.8 | 47.4 / 0.8 / 4.0 / 0.0 / 47.8 | 32.1 / 0.3 / 8.1 / 0.0 / 59.5 | 30.0 / 0.2 / 5.6 / 0.0 / 64.2 |
| Self-decodable layer, median over R | 24 | 24 | 24 | 24 |
| Early exit at 99% agreement, mean layer: label-free / confidence / oracle | 29.0 / 31.4 / 24.5 | 29.0 / 31.6 / 24.4 | 25.2 / 30.7 / 23.5 | 25.2 / 30.9 / 23.5 |

The emission layer and the exit depths reproduce on 500 questions. O is about 1% of questions, so the subset holds only a handful, and loop effects below a point have confidence intervals spanning zero on it. Use the full set for those.

The October tables for all six model/prompt lines are in the [README](../README.md#results-in-one-table).

## Study 2: generation

Qwen3-4B, non-thinking, temperature 0.7, top-p 0.8, top-k 20, up to 3,072 new tokens. The decoder in `scripts/engine.py` reuses the HF weights. It lets each token exit early, skip blocks or repeat blocks inside a batch, and draws sampling noise from a hash of (problem, seed, step) so runs pair up problem by problem.

```bash
cd experiments/gen_depth
python -I scripts/test_engine.py --model qwen3-4b          # matches HF: logit diff 0, greedy 639/640
```

Pipeline, with the October problem lists and seeds:

```bash
# 1. Base trajectories on all of MATH-500 (October: seeds 0-3, two shards per seed pair), then teacher-forced
#    replay of seeds 0,1, the token depth profile, the self-supervised exit probe and the tuned lens
python -I scripts/run_gen.py --model qwen3-4b --select source=math500 --seeds 0,1 --policy base --tag base --shard 0/2   # and --shard 1/2
python -I scripts/grade.py   --model qwen3-4b --tag base --pool data/pool.jsonl
python -I scripts/replay.py  --model qwen3-4b --tag base --pool data/pool.jsonl --seeds 0,1 --h_every 8
python -I scripts/analyze_replay.py   --model qwen3-4b --tag base
python -I scripts/train_exit_probe.py --model qwen3-4b --tag base
python -I scripts/train_tuned_lens.py --model qwen3-4b --tag base      # rank 128
python -I scripts/train_tuned_exit.py --model qwen3-4b --tag base      # writes results/qwen3-4b/base/tuned_exit.npz

# 2. Arms: one run_gen + grade per policy. Screening set = data/screen_m500.txt, seeds 0,1, pool data/pool_tmp_math500.jsonl
#    Main evaluation = data/main_v1.txt, seeds 2,3,4,5, pool data/pool.jsonl
python -I scripts/run_gen.py --model qwen3-4b --pool data/pool_tmp_math500.jsonl --pids data/screen_m500.txt --seeds 0,1 \
       --policy 'oracle:read=logit' --tag O_logit
python -I scripts/grade.py --model qwen3-4b --tag O_logit --pool data/pool_tmp_math500.jsonl

# 3. Paired comparison against a baseline tag
python -I scripts/compare.py --model qwen3-4b --base AA_salt1 --runs O_logit,E_t99
```

`scripts/queue_main.py` and `scripts/gpuq.sh` queue many arms across GPUs. Policy strings are documented at the top of `scripts/policies.py`. The arms behind the headline numbers:

| Tag | Set | Policy | October accuracy |
|---|---|---|---|
| `AA_salt1` | screening | `base` with `--salt 1` (fresh sampling noise; the A/A reference) | 66.4% |
| `O_logit` | screening | `oracle:read=logit` (exit never changes the token; deep K/V filled from the exit state) | 27.9% |
| `E_t99` | screening | `exit:rule=tprobe,probe=results/qwen3-4b/base/tuned_exit.npz,thr=0.99` | 26.0% |
| `E_t99f` | screening | same, with `fullkv=1` (deep K/V recomputed at full depth; diagnostic, saves nothing) | 48.6% |
| `M_L_c35` | main | `loop:win=18-25,it=1,cond=commit>=35` (pre-registered loop arm) | 3.3 points below base, p = 0.024 |
| `M_T_c35` | main | `temp:mult=1.5,cond=commit>=35` (temperature control on the same tokens) | not significant against base |

Two cautions from October:

- **Selection bias.** Problems picked by their baseline outcome regress toward the mean: re-running the baseline on the screening set alone gains about 3 points. Compare screening arms against the A/A rerun `AA_salt1`, and use the main set (selected on seeds 0–1, evaluated on 2–5) for claims.
- **Batch-dependent noise.** bf16 results depend on batch composition, so the same job diverges under a different schedule even without intervention. Paired differences stay unbiased, but expect run-to-run noise.

## What was verified on this branch

- **Same code.** The study scripts are byte-identical to the ones that produced October, apart from path configuration and one fix: `smoke.py` used to overwrite the fixed 1-shot setting with 0-shot for Llama.
- **Exact recomputation.** From the October caches and probes, this branch reproduces bit for bit the taxonomy (per-question table and summary), all early-exit statistics, and all 52 label-free feature arrays with their exit-policy analysis, on both Qwen3-4B lines.
- **Fresh extraction.** On core500, Qwen3-4B base, Qwen3-4B chat and Llama-3.1-8B agree with October in 98–99% of per-layer top answers. The residual differences match what changing only the batch size produces, so they are bf16 noise.
- **End to end.** Extraction through loop evaluation and stage-2 analysis runs cleanly on core500 for both Qwen3-4B lines.
- **Generation.** `test_engine.py` passes and the `base`, `oracle`, `exit` and `loop` arms run and grade on a 24-problem check (`data/verify24.txt`).
