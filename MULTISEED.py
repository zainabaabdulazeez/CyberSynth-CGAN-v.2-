"""
MULTISEED.py
============
Runs the corrected pipeline across several seeds and produces Table 6:
per-class recall as mean +/- SD, 95% confidence intervals, paired t-tests
with Holm-Bonferroni correction.

This is the single highest-value remaining experiment. Reviewer 2 treats the
absence of variance analysis as disqualifying.

HOW TO RUN
  1. Save next to RUN_ME.py in results/
  2. Open in VS Code, press the Run button
  3. Leave it. 5 seeds x ~48 min = about 4 hours.

It saves after EVERY seed, so if it is interrupted you keep what finished and
can restart with the remaining seeds by editing SEEDS below.

WRITES
  results/multiseed_raw.json        per-seed per-class results
  results/multiseed_table6.md       Table 6, ready to paste
  results/predictions_seed<N>.npz   predictions per seed (upload any one)
"""

import importlib.util
import json
import os
import sys
import time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import precision_score, recall_score

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(HERE, "CyberSynth-CGAN v.2.py")
DATASET_PATH = os.environ.get("CICIDS2017_DIR", os.path.join(HERE, "data", "CIC-IDS-2017"))
OUT_DIR = os.environ.get("RESULTS_DIR", os.path.join(HERE, "results"))
EPOCHS = 2000

SEEDS = [42, 43, 44, 45, 46]          # drop finished ones if restarting

CLASSES = ["Web Attack: Brute Force", "Bot", "DoS slowloris",
           "DoS Slowhttptest", "SSH-Patator", "FTP-Patator", "Web Attack: XSS"]

BASE_KEY = "Real train"                # matches "Real train -> Real test"
AUG_KEY = "Real+Synth train"           # matches "Real+Synth train -> Real test"


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout

    def write(self, s):
        try:
            self.stdout.write(s)
        except Exception:
            pass
        self.f.write(s)

    def flush(self):
        try:
            self.stdout.flush()
        except Exception:
            pass
        self.f.flush()


def load_module():
    spec = importlib.util.spec_from_file_location("cgan", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cgan"] = mod
    spec.loader.exec_module(mod)
    return mod


def pick(results, key):
    """Find the experiment whose name starts with key."""
    for name in results:
        if name.startswith(key):
            return results[name]
    raise KeyError(f"No experiment starting with {key!r}. Got: {list(results)}")


def run_one_seed(seed):
    mod = load_module()

    # Override the module's seed, then re-seed everything.
    if hasattr(mod, "SEED"):
        mod.SEED = seed
    if hasattr(mod, "set_global_seed"):
        mod.set_global_seed(seed)
    else:
        import random
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    t0 = time.time()
    out = mod.main_improved_pipeline(
        dataset_path=DATASET_PATH, epochs=EPOCHS, batch_size=128,
        noise_dim=100, device=torch.device("cpu"), save_plots=False)
    mins = (time.time() - t0) / 60

    res = out["evaluation_results"]
    base, aug = pick(res, BASE_KEY), pick(res, AUG_KEY)

    y_true = np.asarray(base["true_labels"])
    yp_base = np.asarray(base["predictions"])
    yp_aug = np.asarray(aug["predictions"])

    export = {"y_true": y_true, "yp_base": yp_base, "yp_aug": yp_aug}
    for tag, r in (("base", base), ("aug", aug)):
        if "probabilities" in r:
            export[f"proba_{tag}"] = np.asarray(r["probabilities"], dtype=np.float32)
        if "classes" in r:
            export[f"classes_{tag}"] = np.asarray(r["classes"])
    np.savez_compressed(os.path.join(OUT_DIR, f"predictions_seed{seed}.npz"), **export)

    rec = {"seed": seed, "minutes": round(mins, 1), "classes": {}}
    for c in CLASSES:
        if c not in set(y_true.tolist()):
            continue
        rec["classes"][c] = {
            "base_recall": float(recall_score(y_true, yp_base, labels=[c],
                                              average="micro", zero_division=0)),
            "aug_recall": float(recall_score(y_true, yp_aug, labels=[c],
                                             average="micro", zero_division=0)),
            "base_precision": float(precision_score(y_true, yp_base, labels=[c],
                                                    average="micro", zero_division=0)),
            "aug_precision": float(precision_score(y_true, yp_aug, labels=[c],
                                                   average="micro", zero_division=0)),
            "support": int((y_true == c).sum()),
        }
    return rec


def holm_bonferroni(pvals):
    order = sorted(pvals, key=pvals.get)
    m, prev, out = len(order), 0.0, {}
    for i, k in enumerate(order):
        c = min(1.0, max(prev, pvals[k] * (m - i)))
        out[k] = c
        prev = c
    return out


def build_table(runs):
    names = [c for c in CLASSES if c in runs[0]["classes"]]
    raw_p, rows = {}, []

    for c in names:
        b = np.array([r["classes"][c]["base_recall"] for r in runs]) * 100
        a = np.array([r["classes"][c]["aug_recall"] for r in runs]) * 100
        d = a - b

        if len(runs) > 1 and d.std() > 1e-12:
            t, p = stats.ttest_rel(a, b)
            lo, hi = stats.t.interval(0.95, len(d) - 1, d.mean(), stats.sem(d))
        else:
            p, lo, hi = 1.0, d.mean(), d.mean()

        raw_p[c] = float(p)
        rows.append({
            "class": c, "support": runs[0]["classes"][c]["support"],
            "base_mean": b.mean(), "base_sd": b.std(ddof=1) if len(b) > 1 else 0.0,
            "aug_mean": a.mean(), "aug_sd": a.std(ddof=1) if len(a) > 1 else 0.0,
            "delta": d.mean(), "ci_lo": lo, "ci_hi": hi,
        })

    corrected = holm_bonferroni(raw_p)
    for r in rows:
        r["p"] = corrected[r["class"]]
        r["sig"] = "Yes" if r["p"] < 0.05 else "No"

    lines = [
        f"**Table 6.** Per-class recall across {len(runs)} independent seeds "
        "(mean ± SD), with paired t-tests after Holm–Bonferroni correction.",
        "",
        "| Class | Support | Baseline recall (%) | Augmented recall (%) | Δ (pp) | 95% CI | p (corr.) | Sig. |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: -abs(x["delta"])):
        lines.append(
            f"| {r['class']} | {r['support']} | "
            f"{r['base_mean']:.2f} ± {r['base_sd']:.2f} | "
            f"{r['aug_mean']:.2f} ± {r['aug_sd']:.2f} | "
            f"{r['delta']:+.2f} | [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | "
            f"{r['p']:.4f} | {r['sig']} |")
    return "\n".join(lines), rows


def main():
    runs = []
    path_raw = os.path.join(OUT_DIR, "multiseed_raw.json")
    if os.path.exists(path_raw):
        runs = json.load(open(path_raw))
        print(f"[RESUME] {len(runs)} seed(s) already done: "
              f"{[r['seed'] for r in runs]}\n")

    done = {r["seed"] for r in runs}
    todo = [s for s in SEEDS if s not in done]

    for i, seed in enumerate(todo, 1):
        print("=" * 70)
        print(f"SEED {seed}   ({i} of {len(todo)})")
        print("=" * 70, flush=True)
        runs.append(run_one_seed(seed))
        json.dump(runs, open(path_raw, "w"), indent=2)          # save after each
        print(f"[OK] seed {seed} done in {runs[-1]['minutes']} min\n", flush=True)

    table, rows = build_table(runs)
    open(os.path.join(OUT_DIR, "multiseed_table6.md"), "w", encoding="utf-8").write(table)

    print("\n" + "=" * 70)
    print("TABLE 6")
    print("=" * 70)
    print(table)

    bf = next((r for r in rows if r["class"] == "Web Attack: Brute Force"), None)
    if bf:
        print(f"\nHEADLINE: {bf['base_mean']:.2f}% -> {bf['aug_mean']:.2f}% "
              f"({bf['delta']:+.2f} pp, SD {bf['aug_sd']:.2f}, p = {bf['p']:.4f})")
        print("Use this in the Abstract, Section 1.3 and Section 6.")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    sys.stdout = sys.stderr = Tee(os.path.join(OUT_DIR, "multiseed.log"))
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        print("\n[FAILED] Send the error above to Claude.")
