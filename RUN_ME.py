"""
RUN_ME.py
=========
Just open this file in VS Code and press the Run button (the > arrow, top right).
No terminal commands needed.

It checks that everything exists first and tells you plainly if something is
missing, then runs the 2000-epoch pipeline and writes:

    results/cgan_run_2000.log    <- progress, written as it goes
    results/predictions.npz      <- upload THIS to Claude
    results/results_2000.json    <- reports + confusion matrices
"""

import importlib.util
import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# If a path below is wrong, fix it here. Everything else is automatic.
HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(HERE, "CyberSynth-CGAN v.2.py")
DATASET_PATH = os.environ.get("CICIDS2017_DIR", os.path.join(HERE, "data", "CIC-IDS-2017"))
OUT_DIR = os.environ.get("RESULTS_DIR", os.path.join(HERE, "results"))
EPOCHS = 2000
TARGET = "Web Attack: Brute Force"
# ---------------------------------------------------------------------------


class Tee:
    """Print to the screen AND to the log file at the same time."""
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout

    def write(self, s):
        try:
            self.stdout.write(s)
        except Exception:
            pass          # console can't show some characters; log still gets them
        self.f.write(s)

    def flush(self):
        try:
            self.stdout.flush()
        except Exception:
            pass
        self.f.flush()


def main():
    print("=" * 70)
    print("CHECKING PATHS")
    print("=" * 70)

    ok = True
    if not os.path.isfile(SCRIPT_PATH):
        print(f"  [X] Script NOT found:\n      {SCRIPT_PATH}")
        ok = False
    else:
        print(f"  [OK] Script found ({os.path.getsize(SCRIPT_PATH):,} bytes)")

    if not os.path.isdir(DATASET_PATH):
        print(f"  [X] Dataset folder NOT found:\n      {DATASET_PATH}")
        ok = False
    else:
        csvs = [f for f in os.listdir(DATASET_PATH) if f.lower().endswith(".csv")]
        print(f"  [OK] Dataset folder found ({len(csvs)} CSV files)")
        if not csvs:
            print("  [X] ...but it contains no CSV files")
            ok = False

    if not ok:
        print("\nFix the paths at the top of this file, then run it again.")
        return

    for m in ("torch", "sklearn", "pandas", "numpy"):
        try:
            __import__(m)
        except ImportError:
            print(f"  [X] Missing package: {m}   (pip install {m})")
            ok = False
    if not ok:
        return
    print("  [OK] All packages available\n")

    import numpy as np
    import torch
    from sklearn.metrics import classification_report, confusion_matrix

    spec = importlib.util.spec_from_file_location("cgan", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cgan"] = mod
    spec.loader.exec_module(mod)

    print("=" * 70)
    print(f"RUNNING PIPELINE - {EPOCHS} epochs")
    print("This takes a few hours. You can close VS Code's focus and leave it.")
    print("=" * 70, flush=True)

    t0 = time.time()
    out = mod.main_improved_pipeline(
        dataset_path=DATASET_PATH,
        epochs=EPOCHS,
        batch_size=128,
        noise_dim=100,
        device=torch.device("cpu"),
        save_plots=False,
    )
    print(f"\n[TIMING] finished in {(time.time() - t0) / 60:.1f} minutes")

    summary = {"epochs": EPOCHS, "experiments": {}}
    export = {}

    for name, res in out["evaluation_results"].items():
        y_true = np.asarray(res["true_labels"])
        y_pred = np.asarray(res["predictions"])

        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        print(classification_report(y_true, y_pred, digits=4, zero_division=0))

        entry = {"classification_report": classification_report(
            y_true, y_pred, digits=4, zero_division=0, output_dict=True)}

        if TARGET in set(y_true.tolist()):
            cm = confusion_matrix((y_true == TARGET).astype(int),
                                  (y_pred == TARGET).astype(int), labels=[0, 1])
            tp, fn = int(cm[1, 1]), int(cm[1, 0])
            support = int((y_true == TARGET).sum())
            print(f"\n'{TARGET}' confusion matrix (test partition, support={support}):")
            print(f"                  Pred Other   Pred Target")
            print(f"  Actual Other   {cm[0, 0]:>10d}   {cm[0, 1]:>10d}")
            print(f"  Actual Target  {fn:>10d}   {tp:>10d}")
            print(f"  recall = {tp / (tp + fn) * 100:.2f}%")
            entry["brute_force_cm"] = cm.tolist()
            entry["brute_force_support"] = support

        summary["experiments"][name] = entry

        key = (name.replace(" ", "_").replace("->", "to")
                   .replace("+", "plus").replace("/", "_"))
        export[f"y_true__{key}"] = y_true
        export[f"y_pred__{key}"] = y_pred
        if "probabilities" in res:
            export[f"proba__{key}"] = np.asarray(res["probabilities"], dtype=np.float32)

    np.savez_compressed(os.path.join(OUT_DIR, "predictions.npz"), **export)
    with open(os.path.join(OUT_DIR, "results_2000.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[DONE] Written to {OUT_DIR}:")
    print("   predictions.npz     <- upload this one to Claude")
    print("   results_2000.json")
    print("   cgan_run_2000.log")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    sys.stdout = sys.stderr = Tee(os.path.join(OUT_DIR, "cgan_run_2000.log"))
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        print("\n[FAILED] Copy the error above and send it to Claude.")
