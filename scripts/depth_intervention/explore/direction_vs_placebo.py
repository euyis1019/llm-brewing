"""Direction beyond placebo: per U question (class under the same-batch none run), score +1 if the config moves it to
R/O (correct content formed), -1 if to M (consistently wrong), 0 otherwise; compare each config's mean score with the
placebo's (paired bootstrap over questions). Uses post_classes.csv of a loop_eval output dir."""
import csv, sys, numpy as np
path, placebo = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
P = list(csv.DictReader(open(path)))
cols = [c for c in P[0] if c.startswith(('last_', 'all_')) and not c.endswith('_none') and 'p4_' not in c]
mode = cols[0].split('_')[0]
none = f'{mode}_none'
pl = placebo or [c for c in cols if c.startswith(f'{mode}_p') and 'p4_' not in c][0]
U = [p for p in P if p[none] == 'U']
def score(c): return np.array([1 if p[c] in ('R', 'O') else (-1 if p[c] == 'M' else 0) for p in U], float)
rng = np.random.default_rng(0); B = 2000
idx = rng.integers(0, len(U), (B, len(U)))
sp = score(pl)
print(f'{path}  n_U={len(U)}  placebo={pl}  placebo net={sp.mean()*100:+.2f}pp')
for c in sorted(cols):
    s = score(c); d = s - sp
    bs = d[idx].mean(1)
    print(f'  {c:18s} toC {int((s>0).sum()):4d} toW {int((s<0).sum()):4d} net {s.mean()*100:+.2f}pp  minus placebo {d.mean()*100:+.2f}pp [{np.percentile(bs,2.5)*100:+.2f}, {np.percentile(bs,97.5)*100:+.2f}]')
