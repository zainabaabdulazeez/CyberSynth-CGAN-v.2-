"""
BASELINES.py
============
Regenerates the stale SMOTE and random-oversampling rows of Table VII on the
CORRECTED pipeline. No GAN training is involved, so this takes about 15-20
minutes rather than an hour.

Your current Table VII rows (SMOTE 71.23%, random oversampling 68.45%, baseline
67.77%/61.54%) all came from the pre-correction pipeline and cannot be compared
against your corrected result of 98.34%.

HOW TO RUN
  1. Save in results/
  2. Open in VS Code, press the Run button

WRITES
  results/table7_baselines.md
  results/baselines.log
"""

import importlib.util
import os
import sys
import time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(HERE, "CyberSynth-CGAN v.2.py")
DATASET_PATH = os.environ.get("CICIDS2017_DIR", os.path.join(HERE, "data", "CIC-IDS-2017"))
OUT_DIR = os.environ.get("RESULTS_DIR", os.path.join(HERE, "results"))
SEED = 42
TARGET_PER_CLASS = 6000      # same target the GAN planner uses
CAP_PER_CLASS = 60000        # same cap the pipeline applies to real data

BF, XSS = "Web Attack: Brute Force", "Web Attack: XSS"


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


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def rf():
    return RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=2,
        max_features="sqrt", class_weight="balanced_subsample", bootstrap=True,
        max_samples=0.7, random_state=SEED, n_jobs=-1)


def cap(X, y, n, seed=SEED):
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        keep.append(rng.choice(idx, n, replace=False) if len(idx) > n else idx)
    keep = np.concatenate(keep)
    rng.shuffle(keep)
    return X[keep], y[keep]


def evaluate(name, Xtr, ytr, Xte, yte, elapsed):
    t0 = time.time()
    clf = rf().fit(Xtr, ytr)
    yp = clf.predict(Xte)
    total = elapsed + (time.time() - t0)
    r_bf = recall_score(yte, yp, labels=[BF], average="micro", zero_division=0)
    r_xss = recall_score(yte, yp, labels=[XSS], average="micro", zero_division=0)
    print(f"\n{name}")
    print(f"  training rows : {len(ytr):,}")
    print(f"  Brute Force   : {r_bf*100:.2f}%")
    print(f"  XSS           : {r_xss*100:.2f}%")
    print(f"  time          : {total/60:.1f} min", flush=True)
    return {"method": name, "bf": r_bf * 100, "xss": r_xss * 100,
            "minutes": total / 60}


def main():
    spec = importlib.util.spec_from_file_location("cgan", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cgan"] = mod
    spec.loader.exec_module(mod)

    print("Loading and preprocessing...", flush=True)
    out = mod.load_and_preprocess(DATASET_PATH)

    # The corrected pipeline returns 4 values (raw features, labels, ohe,
    # minority list); the pre-fix version returned 5 with a fitted scaler.
    print(f"[INFO] load_and_preprocess returned {len(out)} values", flush=True)
    if len(out) == 5:
        X_any, Y_onehot, scaler, ohe, _ = out
    elif len(out) == 4:
        X_any, Y_onehot, ohe, _ = out
        scaler = None
    else:
        raise RuntimeError(f"Unexpected return arity: {len(out)}")

    X_any = np.asarray(X_any, dtype=np.float64)
    y = np.array(ohe.categories_[0])[np.asarray(Y_onehot).argmax(1)]

    # Are these raw features or already scaled to [-1, 1]?
    if np.abs(X_any).max() <= 1.0 + 1e-6:
        if scaler is None:
            raise RuntimeError("Features look scaled but no scaler was returned.")
        print("[INFO] features are scaled; inverting to raw", flush=True)
        X_raw = scaler.inverse_transform(X_any)
    else:
        print(f"[INFO] features are raw (max |x| = {np.abs(X_any).max():.3e})", flush=True)
        X_raw = X_any

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.20, stratify=y, random_state=SEED)

    # Prefer the module's own scaler helper so the transform matches the
    # main pipeline exactly; fall back to an equivalent local fit.
    fit_helper = getattr(mod, "fit_log_minmax_scaler", None)
    if fit_helper is not None:
        print("[INFO] using module's fit_log_minmax_scaler", flush=True)
        sc = fit_helper(X_raw[tr])
        Xtr_full, Xte = sc.transform(X_raw[tr]), sc.transform(X_raw[te])
    else:
        print("[INFO] fitting local signed-log + MinMax scaler", flush=True)
        mm = MinMaxScaler((-1, 1)).fit(signed_log(X_raw[tr]))
        Xtr_full = mm.transform(signed_log(X_raw[tr]))
        Xte = mm.transform(signed_log(X_raw[te]))

    ytr_full, yte = y[tr], y[te]
    Xtr, ytr = cap(Xtr_full, ytr_full, CAP_PER_CLASS)

    print(f"  scaled range: [{Xtr.min():.4f}, {Xtr.max():.4f}]")
    print(f"  train {len(ytr):,} (capped)   test {len(yte):,}")
    print(f"  Brute Force test support: {(yte == BF).sum()}   (expect 301)")
    print(f"  XSS test support        : {(yte == XSS).sum()}   (expect 130)\n", flush=True)

    rows = [evaluate("No augmentation", Xtr, ytr, Xte, yte, 0.0)]

    # --- random oversampling -------------------------------------------
    counts = {c: int((ytr == c).sum()) for c in np.unique(ytr)}
    plan = {c: max(n, TARGET_PER_CLASS) for c, n in counts.items()}
    t0 = time.time()
    Xr, yr = RandomOverSampler(sampling_strategy=plan, random_state=SEED).fit_resample(Xtr, ytr)
    rows.append(evaluate("Random oversampling", Xr, yr, Xte, yte, time.time() - t0))

    # --- SMOTE ---------------------------------------------------------
    # SMOTE needs k+1 members, so classes below 6 are left at their real size.
    plan_s = {c: (max(n, TARGET_PER_CLASS) if n >= 6 else n) for c, n in counts.items()}
    t0 = time.time()
    try:
        Xs, ys = SMOTE(sampling_strategy=plan_s, k_neighbors=5,
                       random_state=SEED).fit_resample(Xtr, ytr)
        rows.append(evaluate("SMOTE", Xs, ys, Xte, yte, time.time() - t0))
    except Exception as e:
        print(f"[SMOTE failed] {e}")

    lines = ["**Table VII rows, regenerated on the corrected pipeline (seed 42).**", "",
             "| Method | Brute Force recall | XSS recall | Training time |",
             "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['method']} | {r['bf']:.2f}% | {r['xss']:.2f}% | "
                     f"{r['minutes']:.0f} min |")
    lines += ["| Cyber-Synth-CGAN (unfiltered) | 98.34% | 4.62% | 48 min |", "",
              "*Cyber-Synth-CGAN row carried over from the main run for comparison.*"]

    table = "\n".join(lines)
    open(os.path.join(OUT_DIR, "table7_baselines.md"), "w", encoding="utf-8").write(table)
    print("\n" + "=" * 70)
    print(table)
    print("\nPaste these rows into Table VII, replacing the stale values.")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    sys.stdout = sys.stderr = Tee(os.path.join(OUT_DIR, "baselines.log"))
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        print("\n[FAILED] Send the error above to Claude.")
