"""Compact console tables from RUN/{steer}/eval_{dev,final}.json (scripts/steer.py --eval).
Usage: python scripts/explore/steer_tables.py RUN_NAME [eval_final.json] [split] [gate-substring]"""

# Workspace bootstrap; numerical operations and experiment protocols are unchanged.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from _workspace import experiment_root as _experiment_root, model_root as _model_root, normalized_benchmark_root as _normalized_benchmark_root

import json, os, sys
ROOT = str(_experiment_root('depth_intervention'))
run = sys.argv[1]
fn = sys.argv[2] if len(sys.argv) > 2 else 'eval_dev.json'
split = sys.argv[3] if len(sys.argv) > 3 else 'f0-3'
gsub = sys.argv[4] if len(sys.argv) > 4 else '|none'
R = json.load(open(os.path.join(ROOT, 'results', run if '/' in run else os.path.join(run, 'steer'), fn)))
print(run, fn, 'split', split, 'classes', R['config']['classes'], 'none-layer agreement', R['config']['none_layer_agreement'])
print('checks:', {k: R['config']['run_checks'][k] for k in ('alpha0_bit_exact', 'none_vs_loop_path_argmax_agree',
                                                             'none_vs_fwd_argmax_agree', 'fwd_vs_extract_argmax_agree')})
for k, v in R['diag'].items():
    print(f'  diag {k:12s} head acc {v["head_acc_gold"]:.3f} agree_vote {v["head_agree_vote"]:.3f} | none acc {v["none_acc"]:.3f} '
          f'agree_vote {v["none_agree_vote"]:.3f} | c* share outside span(E) {v["cstar_share_outside_spanE"]:.3f} |cos h| '
          f'{v["cstar_abs_cos_h"]:.3f} | pass conf@.7 {v["gate_pass"]["cons_conf@0.7"]:.3f} not_em {v["gate_pass"]["not_emitted"]:.3f}')
print(f'{"config":42s} {"net pp [CI]":>22s} {"dVote":>7s} {"dSelf4":>7s} {"R_harm":>10s} {"U->R":>10s} {"fix/brk":>10s} '
      f'{"R":>6s} {"O":>6s} {"M":>6s} {"U":>6s} {"comp":>6s} {"steer":>6s} {"emis":>6s}')
for k, x in R['grid'].items():
    if gsub not in k or split not in x:
        continue
    m = x[split]
    pp = lambda z: f'{100 * z:+.2f}' if z is not None else '   -'
    print(f'{k:42s} {pp(m["net"][0]):>7s} [{pp(m["net"][1])},{pp(m["net"][2])}] {pp(m["d_agree_vote"]):>7s} {pp(m["d_self4"]):>7s} '
          f'{m["R_harm"][0]:>4d}/{m["R_harm"][1]:<5d} {m["U_to_R"][0]:>4d}/{m["U_to_R"][1]:<5d} {m["fixes"]:>4d}/{m["breaks"]:<5d} '
          f'{pp(m["net_R"]):>6s} {pp(m["net_O"]):>6s} {pp(m["net_M"]):>6s} {pp(m["net_U"]):>6s} {m["compliance"]:.3f} '
          f'{m["steered"]:.3f} {m["emis_shift"] if m["emis_shift"] is not None else float("nan"):+.2f}')
for nm, s in R['selection'].items():
    print(f'[{nm}] final {s["final"]} (dev L1 {100 * s["final_dev_L1"]:+.3f}pp); cross-fit {s["crossfit"]}')
    d = s['dev_crossfit']
    print(f'   dev cross-fitted: net {d["net"]} R_harm {d["R_harm"]} U->R {d["U_to_R"]} fix/brk {d["fixes"]}/{d["breaks"]} '
          f'steered {d["steered_share"]} dVote {d["L1"]["d_agree_vote"]}')
for nm, x in R['selected'].items():
    for sp in ('f0-3', 'f4'):
        if sp not in x:
            continue
        m = x[sp]
        print(f'  {nm:48s} {sp:4s} {x["cfg"]}\n      net {m["net"]} acc {m["acc_before"]:.4f}->{m["acc_after"]:.4f} R_harm {m["R_harm"]} '
              f'{m["R_harm_rate"]} U->R {m["U_to_R"]} {m["U_to_R_rate"]} fix/brk {m["fixes"]}/{m["breaks"]} all4 {m["all4_correct_net"]}'
              f'\n      L1 {m["L1"]} steered {m["steered_share"]} comp {m["compliance_after"]} emis {m["lens_emission_shift"]}'
              f' (steered, answer unchanged {m["lens_emission_shift_steered_unchanged_answer"]})'
              f'\n      by class ' + ' '.join(f'{k}:{v["net"]} ({v["fixes"]}/{v["breaks"]})' for k, v in m['by_class'].items() if v['net'])
              + (f'\n      method - this: {m["method_minus_this"]}' if 'method_minus_this' in m else ''))
