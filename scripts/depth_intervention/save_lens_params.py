"""Save final-norm weight/eps and the 4 letter unembedding rows (fp32) for probe/lens readouts on CPU.
Writes results/{model}__{fmt}/extract/lens_params.npz (norm_w (d,), eps (), U4 (4,d), letter_ids (4,)).
Usage: python scripts/save_lens_params.py --model qwen3-4b-base --fmt base_fewshot"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
ap = argparse.ArgumentParser(); ap.add_argument('--model'); ap.add_argument('--fmt'); a = ap.parse_args()
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained(C.MODELS[a.model], dtype=torch.float32, device_map='cpu')
d = os.path.join(C.run_dir(a.model, a.fmt), 'extract'); os.makedirs(d, exist_ok=True)
smoke = json.load(open(os.path.join(C.run_dir(a.model, a.fmt), 'smoke.json'))) if os.path.exists(os.path.join(C.run_dir(a.model, a.fmt), 'smoke.json')) else None
ids = smoke['letter_ids'] if smoke else [32, 33, 34, 35]
np.savez(os.path.join(d, 'lens_params.npz'), norm_w=m.model.norm.weight.detach().numpy().astype(np.float32),
         eps=np.float32(m.model.norm.variance_epsilon), U4=m.lm_head.weight[ids].detach().numpy().astype(np.float32),
         letter_ids=np.array(ids))
print('saved', d, ids, m.model.norm.variance_epsilon)
