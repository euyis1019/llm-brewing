# Data products and script interfaces (round 1)

Everything lives under `ROOT=/nvme0/if/research/brewing/exp/depth_intervention`. Python:
`/nvme0/if/anaconda3/envs/qwen3/bin/python` (torch 2.11, transformers 5.13, sklearn, numpy, scipy).
Shared code: `scripts/common.py` (read it first). The plan is `PLAN.md` (Chinese); this file is the contract.

Runs (one per model × prompt format), directory `RUN = results/{model}__{fmt}`:
- A line: `qwen3-4b-base__base_fewshot` (Qwen3-4B-Base, 1-shot completion prompt, critical token after `Answer:\n`)
- B line: `qwen3-4b__chat0` (Qwen3-4B post-trained, non-thinking chat with a system instruction, critical token = first assistant token)
Both models: L = 36 blocks, d = 2560, letter token ids A–D = 32..35 (no leading space).
Every script takes `--model` and `--fmt` (or `--run`) so the same code serves both lines (and Llama-3.1-8B later, L = 32).

## Benchmark (`data/bench/`)
- `questions.jsonl`: Q = 5659 questions. Fields: `qid` (0..Q-1), `source` (gsm_mc 1319, math_mc 1000, pythonio 1000,
  mmlu_pro 1000, ceval 1340), `group`, `fold` (0..4, by group, stratified by source), `subject`, `question`,
  `options` (4, canonical order), `answer` (canonical index of the correct option).
- Version v ∈ {0,1,2,3}: letter i shows canonical option (i+v)%4. Gold letter of row = (answer − v) % 4.
  Content (canonical option index) chosen when letter i is output in version v = (i + v) % 4.
- Rows: row = 4*qid + v, N = 4Q = 22636. `RUN/extract/rows.json` lists `row, qid, v, source, fold, gold` per row.

## Extraction (`RUN/extract/`, produced by `scripts/extract.py`, already defined)
| file | shape / dtype | meaning |
|---|---|---|
| `hidden_bf16.u16` | memmap uint16 (N, L, d) | bf16 bits of block-l output (pre final norm) at the critical position. Convert with `common.bf16_bits_to_f32` or `torch.from_numpy(x.view(np.int16)).view(torch.bfloat16)`; open with `common.open_hidden(model, fmt)` |
| `lens4.npy` | f32 (N, L, 4) | logit lens at A–D (fp32 final norm + unembedding) |
| `csd_raw4.npy` | f32 (N, L, 4) | raw Patchscope/CSD letter logits (main readiness readout). Layer L−1 equals the real output |
| `csd_base4.npy` | f32 (4,) | bare target prompt letter logits. Sensitivity readout `csd_sub4 = csd_raw4 − csd_base4` |
| `csd_top1.npy`, `csd_mass.npy` | (N, L) | full-vocab argmax id / prob mass on the 4 letters of the CSD readout |
| `out_logits4.npy` | f32 (N, 4) | real output letter logits (fp32) |
| `out_top1.npy`, `out_mass.npy` | (N,) | real output full-vocab argmax id; mass on letters. Compliant row ⇔ out_top1 ∈ letter ids |
| `lens_params.npz` | | `norm_w` (d,), `eps`, `U4` (4, d) letter unembedding rows, `letter_ids` |
| `meta.json` | | N, L, d, letter ids, target prompt, compliance, accuracy, timing |

Lens readout from a state h: `hn = h / sqrt(mean(h²) + eps)`; `lens = (hn * norm_w) @ U4.T`.

## Probe (`RUN/probe/`, produced by `scripts/train_probe.py`)
Form (user's design, implemented in hidden space): `logits4 = lens(h) + hn @ W.T + b`, with `hn` the RMS-normalized
state (no weight), W (4, d), b (4,), per layer; cross-entropy + `lam·||W||²` (b unpenalized), full-batch L-BFGS in
torch on GPU (no Adam/early stopping). Rungs: `lens` (no params), `bias` (lens + b), `full` (lens + W, b).
Cross-fitting: outer folds = `fold` field. For outer fold f, train on rows with fold≠f, predict rows with fold==f.
lam chosen per (outer fold, layer) on an inner split of the training rows (by fold, never the outer fold).
Null probes: 3 seeds; permute `answer` among questions **within source** (question level), derive row labels as
(perm_answer − v) % 4, same pipeline, lam re-selected or fixed to the real probe's (state which).
| file | shape | meaning |
|---|---|---|
| `oof_lens.npy`, `oof_bias.npy`, `oof_full.npy` | f32 (N, L, 4) | out-of-fold logits for each rung (lens = copy of lens4) |
| `oof_null{0,1,2}.npy` | f32 (N, L, 4) | out-of-fold logits of the null full-rung probes |
| `weights_full.npz` | | `W` (5, L, 4, d) f32, `b` (5, L, 4), `lam` (5, L): fold-f weights = trained without fold f |
| `weights_bias.npz` | | `b` (5, L, 4) |
| `summary.json` | | per-layer OOF row accuracy and content-level 4/4 rate per rung and null, lam grid, timings |
Applying fold weights to a new state h of origin layer j for a question in fold f: `lens(h) + hn @ W[f,j].T + b[f,j]`.

## Taxonomy (`RUN/classify/`, produced by `scripts/classify.py`)
Inputs: probe logits (`--probe full` default; `lens` allowed as a stand-in), CSD readout (`--csd raw` default, `sub`
sensitivity), real output. Row-level correctness at layer l: argmax == gold. Content-level (per question, 4 rows):
- `P4[l]`: all 4 versions' probe argmax are correct; `C4[l]`: all 4 CSD argmax correct.
- `FPCL_c(K)`: first l with P4 true on l..l+K−1. `FJC_c(K)`: first l with (P4 & C4) true on l..l+K−1.
  **End truncation (PLAN §6.3, 'held to the final layer or K consecutive layers'):** a run that reaches the last
  layer L−1 counts even if it is shorter than K, i.e. require (P4 & C4) on l..min(l+K−1, L−1). Without this, questions
  whose answer forms only at L−2/L−1 (final 4/4 correct) would fall into U. Same truncation on loop trajectories
  (last step). (Added 2026-10-16 after the first stand-in run: with K=3 untruncated, 716 final-4/4 questions became U.)
  **Last layer = real output (`--last_layer output`, official from 2026-10-16 evening):** at layer L−1 the probe readout
  is replaced by the real output (CSD at L−1 already equals it), so every final-4/4-correct question has FJC_c and is R.
  With the probe at L−1, 103 (A) / 128 (B) final-4/4 questions were U ('U_final4') and inflated loop U→R by 20–27%.
  v0 and per-layer probe accuracy still use the probe itself. loop_eval.py has the same option for the last step.
- `n_final_correct` ∈ 0..4 (real output); `final_wrong_consistent`: all 4 real outputs map to the same wrong content.
- v1 outcome(K): FJC_c exists → `R` if n_final_correct == 4 else `O`; no FJC_c → `M` if final_wrong_consistent else `U`.
  Main K = 3; also K = 1, 2. Sub-tags: U_avail (FPCL_c exists) vs U_none; O by n_final_correct.
  Questions with any non-compliant row are outcome `X` (excluded; count them).
- v0 outcome (first-paper compatible, per row): FPCL = first layer probe correct; FJC = first layer probe & CSD correct;
  FPCL None → `NB`; FJC → R/O by final correctness; else M if CSD tail confidence ≥ 0.5 else U, where tail confidence
  = max over classes of the mean 4-way softmax over layers l ≥ ceil(0.75·L) (match
  `/nvme0/if/llm-brewing/source_code/brewing/diagnostics/{metrics,outcome}.py`).
- Controls: (a) null-probe false positives: same rule with `oof_null*` → FPCL_c/FJC_c/“O” rates;
  (b) matched decoy: for each question and each wrong content c', apply the same FJC_c rule with "correct" replaced by
  "points to c'" (probe and CSD 4/4 on c' for K layers) and final not 4/4 on c' → decoy-O rate (mean over c').
Outputs: `questions.csv` (one row per qid: source, fold, outcome_v1_K1/K2/K3, FPCL_c_K*, FJC_c_K*, n_final_correct,
final contents, u_avail, …), `rows_v0.csv`, `summary.json` (counts by source, controls, bootstrap 95% CIs by question,
2000 resamples). Same for `--csd sub` in `classify_csdsub/`.

## Early exit (`RUN/early_exit/`, `scripts/early_exit.py`, CPU)
Heads: `E-head` (probe OOF logits, main) and `E-csd` (CSD raw logits). Window start w0 = round(0.45·L) (16 for L=36).
Rules choose an exit layer per row **without that row's gold**: `fix` (one layer ≥ w0 chosen on the other 4 folds by
population accuracy), `stab(k,τ)` (first l ≥ w0 whose top-1 is unchanged for k layers and max-softmax ≥ τ; else L−1;
(k,τ) chosen on the other folds), `conf` (argmax confidence in [w0, L−1]). Report: O→R rate (O questions that become
4/4 correct), partial fixes, matched decoy (non-O questions with the same n_final_correct), net single-version accuracy
change on all rows, R harm (R questions no longer 4/4), bootstrap CIs. Acceptance A1 in PLAN §7.

## Depth loop (`RUN/loop/`, `scripts/loop.py` GPU + `scripts/loop_eval.py` CPU)
Layer sequence: 0..b, then (a..b) × k, then b+1..L−1 (same weights). Modes: `all` (every position, `LM.forward_sequence`)
and `last` (critical position only; context keys/values fixed from the normal pass, `LM.run_single` with the sample's
own prefix cache). Damped steps: h + β(block(h) − h) on the repeated steps only.
Configs for L = 36 (scale by L/36 and round for other models): `w16_23_k1, w16_23_k2, w18_25_k1, w18_25_k2,
w20_27_k1, w20_27_k2, d18_25_k2` (β = 0.5), placebo `p4_11_k1, p28_33_k1`, plus `none` (identity, sanity).
Targets (`RUN/loop/targets.json`): all U questions (v1, K=3) + 300 random each of R, M, O (all if fewer), seed fixed.
Per config `RUN/loop/{mode}_{cfg}/`: `rows.npy` (row ids), `seq.npy` (origin block of each step, S steps),
`step_kind.npy` (0 first pass, r = repeat pass r, −1 post), `final4.npy` (n, 4) f32, `final_top1.npy`,
`traj_csd4.npy` (n, S, 4) f32 (CSD of each step state at its origin layer), `traj_lens4.npy` (n, S, 4),
`traj_hidden_bf16.u16` (n, S, d) memmap, `meta.json`.
loop_eval: post-loop v1 class per question, using the extended trajectory (probe of the origin layer with the
question's fold weights, CSD from traj_csd4, FJC_c over consecutive steps) and the final real output. Report the
transition matrix original → post per config and mode, U state-change rate vs placebo, direction (toward correct vs
toward a wrong content), R harm, bootstrap CIs. Acceptance A2 in PLAN §7.

## Rules for every script
- Write `results/.../*.json` with config, n, seconds, script md5. Never overwrite another run's outputs.
- Never use a row's gold label to choose anything applied to that row (exit layers, lam, thresholds): cross-fit by fold.
- fp32 for all readouts; treat |top1 − top2| < 1e-4 logits as a tie (report counts).
- GPU: only `CUDA_VISIBLE_DEVICES` you were given; never kill processes you did not start; no `pkill -f`.
