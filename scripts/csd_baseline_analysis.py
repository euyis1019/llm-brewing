#!/usr/bin/env python3
"""Compare CSD (baseline-subtracted) vs raw digit argmax.

For each sample: does CSD ever predict correctly at any layer?
Does raw digit argmax ever predict correctly? How many samples
would gain a valid FJC if we dropped baseline subtraction?

Skips NB samples. Early-exits per sample once both modes found correct.
"""
import json, numpy as np, torch, sys, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = Path(__file__).resolve().parent.parent / "brewing_output"
MODEL_CACHE = Path("/path/to/cue/models")
SEED = "seed42"
MODEL_DIR = "Qwen__Qwen2.5-Coder-7B"
MODEL_ID = "Qwen/Qwen2.5-Coder-7B"
TASKS = ["value_tracking","computing","conditional","function_call","loop","loop_unrolled"]
TARGET_PROMPT = '# The value of x is "'

def main():
    from nnsight import LanguageModel
    from brewing.nnsight_ops import get_layer_output, get_next_token_logits, get_layers
    from brewing.schema import Sample

    model = LanguageModel(str(MODEL_CACHE/MODEL_ID), device_map="auto", torch_dtype=torch.float16, dispatch=True)
    tokenizer = model.tokenizer
    digit_ids = [tokenizer.encode(str(d), add_special_tokens=False)[-1] for d in range(10)]
    model_dtype = next(model.parameters()).dtype

    with model.trace(TARGET_PROMPT):
        bl = get_next_token_logits(model).cpu().save()
    baseline = np.array([bl[0].detach().cpu().float().numpy()[tid] for tid in digit_ids])
    print("Baseline digit logits:", [f"{baseline[d]:.2f}" for d in range(10)])

    # Pre-cache layer devices
    layers_list = get_layers(model)
    n_layers = len(layers_list)
    layer_devices = [next(layers_list[l].parameters()).device for l in range(n_layers)]
    print(f"Model: {n_layers} layers\n")

    grand = {"csd_ever": 0, "raw_ever": 0, "raw_only": 0, "both_never": 0, "total": 0}

    for task in TASKS:
        cache_dir = BASE/"caches"/"cuebench"/"eval"/task/SEED/MODEL_DIR
        hs = np.load(cache_dir/"hidden_states.npz")["hidden_states"]
        with open(cache_dir/"meta.json") as f:
            meta = json.load(f)
        with open(BASE/"datasets"/"cuebench"/"eval"/task/SEED/"samples.json") as f:
            samples = {s["id"]: Sample(**s) for s in json.load(f)}
        with open(BASE/"results"/"cuebench"/"eval"/task/SEED/MODEL_DIR/"diagnostics.json") as f:
            diag_map = {d["sample_id"]: d for d in json.load(f)["sample_diagnostics"]}

        sample_ids = meta["sample_ids"]
        csd_ever = 0; raw_ever = 0; raw_only = 0; both_never = 0; total = 0
        raw_only_by_answer = {str(d): 0 for d in range(10)}

        for si in range(len(sample_ids)):
            sid = sample_ids[si]
            if sid not in samples:
                continue
            s = samples[sid]
            d = diag_map.get(sid, {})
            if d.get("fpcl") is None:
                continue  # skip NB
            total += 1

            csd_any = False; raw_any = False
            for layer in range(n_layers):
                h = torch.tensor(hs[si, layer]).to(dtype=model_dtype, device=layer_devices[layer])
                with model.trace(TARGET_PROMPT):
                    get_layer_output(model, layer)[0, -1] = h
                    logits_proxy = get_next_token_logits(model).cpu().save()
                al = np.array([logits_proxy[0].detach().cpu().float().numpy()[tid] for tid in digit_ids])
                del logits_proxy

                if not csd_any and str(np.argmax(al - baseline)) == s.answer:
                    csd_any = True
                if not raw_any and str(np.argmax(al)) == s.answer:
                    raw_any = True
                if csd_any and raw_any:
                    break

            if csd_any: csd_ever += 1
            if raw_any: raw_ever += 1
            if raw_any and not csd_any:
                raw_only += 1
                raw_only_by_answer[s.answer] += 1
            if not raw_any and not csd_any: both_never += 1

            if total % 50 == 0:
                gc.collect(); torch.cuda.empty_cache()
                print(f"  [{task}] {total} samples done, raw_only so far: {raw_only}", flush=True)

        print(f"\n{task:<18} N={total}")
        print(f"  CSD (baseline-sub):  ever-correct={csd_ever}/{total} ({csd_ever/total*100:.1f}%)")
        print(f"  Raw digit argmax:    ever-correct={raw_ever}/{total} ({raw_ever/total*100:.1f}%)")
        print(f"  Raw-only (CSD miss): {raw_only}/{total} ({raw_only/total*100:.1f}%)")
        print(f"  Both never correct:  {both_never}/{total} ({both_never/total*100:.1f}%)")
        nonzero = {k: v for k, v in raw_only_by_answer.items() if v > 0}
        if nonzero:
            print(f"  Raw-only by answer:  {nonzero}")
        print()

        grand["csd_ever"] += csd_ever
        grand["raw_ever"] += raw_ever
        grand["raw_only"] += raw_only
        grand["both_never"] += both_never
        grand["total"] += total
        gc.collect(); torch.cuda.empty_cache()

    t = grand["total"]
    print("="*60)
    print(f"GRAND TOTAL  N={t}")
    print(f"  CSD ever-correct:    {grand['csd_ever']}/{t} ({grand['csd_ever']/t*100:.1f}%)")
    print(f"  Raw ever-correct:    {grand['raw_ever']}/{t} ({grand['raw_ever']/t*100:.1f}%)")
    print(f"  Raw-only (CSD miss): {grand['raw_only']}/{t} ({grand['raw_only']/t*100:.1f}%)")
    print(f"  Both never:          {grand['both_never']}/{t} ({grand['both_never']/t*100:.1f}%)")

if __name__ == "__main__":
    main()
