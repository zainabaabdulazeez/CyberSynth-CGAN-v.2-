"""
CAPTURE_PROBA.py
================
Your pipeline discards predict_proba output, so ROC and PR curves cannot be
drawn from the saved predictions. This re-runs the pipeline once and captures
the probabilities — WITHOUT editing your source file.

It works by temporarily replacing RandomForestClassifier inside the loaded
module with a subclass that computes predict_proba whenever predict is called.

HOW TO RUN
  1. Save in results/
  2. Open in VS Code, press Run
  3. Takes about 48 minutes (one full pipeline run)
  4. Then run METRICS_AND_FIGURES.py again — the curves will appear

WRITES
  results/predictions_with_proba.npz

If you want per-seed curves later, copy the RecordingRF class and the two
patch lines into MULTISEED.py, just after load_module().
"""

import importlib.util
import os
import sys
import time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier as _RF

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(HERE, "CyberSynth-CGAN v.2.py")
DATASET_PATH = os.environ.get("CICIDS2017_DIR", os.path.join(HERE, "data", "CIC-IDS-2017"))
OUT_DIR = os.environ.get("RESULTS_DIR", os.path.join(HERE, "results"))
EPOCHS = 2000

_CAPTURED = []          # (classes_, proba) in call order


class RecordingRF(_RF):
    """Identical to RandomForestClassifier, but records probabilities."""

    def predict(self, X):
        preds = super().predict(X)
        try:
            proba = super().predict_proba(X).astype(np.float32)
            _CAPTURED.append((np.asarray(self.classes_), proba))
            print(f"  [captured probabilities: {proba.shape}]", flush=True)
        except Exception as e:
            print(f"  [could not capture probabilities: {e}]", flush=True)
        return preds


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


def main():
    spec = importlib.util.spec_from_file_location("cgan", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cgan"] = mod
    spec.loader.exec_module(mod)

    # The patch: two lines.
    mod.RandomForestClassifier = RecordingRF
    print("[PATCH] RandomForestClassifier replaced with recording subclass\n")

    t0 = time.time()
    out = mod.main_improved_pipeline(
        dataset_path=DATASET_PATH, epochs=EPOCHS, batch_size=128,
        noise_dim=100, device=torch.device("cpu"), save_plots=False)
    print(f"\n[TIMING] {(time.time() - t0) / 60:.1f} minutes")

    res = out["evaluation_results"]
    names = list(res)
    print(f"\n[INFO] experiments: {names}")
    print(f"[INFO] probability arrays captured: {len(_CAPTURED)}")

    if len(_CAPTURED) < len(names):
        print("[WARN] Fewer captures than experiments — some classifiers may not")
        print("       route through the patched class. Curves may be incomplete.")

    base_name = next((n for n in names if "Synth" not in n), names[0])
    aug_name = next((n for n in names if "Synth" in n), names[-1])

    y_true = np.asarray(res[base_name]["true_labels"])
    export = {
        "y_true": y_true,
        "yp_base": np.asarray(res[base_name]["predictions"]),
        "yp_aug": np.asarray(res[aug_name]["predictions"]),
    }

    # Captures arrive in the order the experiments were evaluated.
    if len(_CAPTURED) >= 1:
        export["classes_base"], export["proba_base"] = _CAPTURED[0]
    if len(_CAPTURED) >= 2:
        export["classes_aug"], export["proba_aug"] = _CAPTURED[1]

    path = os.path.join(OUT_DIR, "predictions_with_proba.npz")
    np.savez_compressed(path, **export)
    size_mb = os.path.getsize(path) / 1e6
    print(f"\n[DONE] {path}  ({size_mb:.1f} MB)")
    print("Now run METRICS_AND_FIGURES.py again to get Figure 5 and Figure 8.")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    sys.stdout = sys.stderr = Tee(os.path.join(OUT_DIR, "capture_proba.log"))
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        print("\n[FAILED] Send the error above to Claude.")
