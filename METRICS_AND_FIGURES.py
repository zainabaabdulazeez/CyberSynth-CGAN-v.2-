"""
METRICS_AND_FIGURES.py
======================
Produces Table 7 (extended metrics), Figure 5 (ROC) and Figure 8 (PR curves)
from predictions you have ALREADY saved. No re-training, runs in seconds.

Answers Reviewer 2 comment 5 (MCC, Cohen's kappa, balanced accuracy, G-mean,
per-class ROC-AUC, PR-AUC, false positive rates) and Reviewer 1 comment 2.

HOW TO RUN
  1. Save in results/
  2. Open in VS Code, press Run

It looks for predictions.npz, or predictions_seed42.npz, whichever exists.

WRITES
  results/table7_extended_metrics.md
  results/Figure5_ROC.png
  results/Figure8_PR_curves.png
"""

import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("RESULTS_DIR", os.path.join(HERE, "results"))
FOCUS = ["Web Attack: Brute Force", "Web Attack: XSS", "Bot"]


def find_predictions():
    for pat in ("predictions_with_proba.npz", "predictions.npz",
                "predictions_seed42.npz", "predictions_seed*.npz"):
        hits = sorted(glob.glob(os.path.join(OUT_DIR, pat)))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"No predictions file in {OUT_DIR}. Run RUN_ME.py or MULTISEED.py first.")


def load(path):
    d = np.load(path, allow_pickle=True)
    keys = list(d.keys())
    print(f"Loaded {os.path.basename(path)}")
    print(f"  arrays: {keys}\n")

    if "y_true" in keys:
        return (d["y_true"], d["yp_base"], d["yp_aug"],
                d["proba_base"] if "proba_base" in keys else None,
                d["proba_aug"] if "proba_aug" in keys else None,
                d["classes_base"] if "classes_base" in keys else None)

    # RUN_ME.py naming: y_true__<experiment>, y_pred__<experiment>
    tk = [k for k in keys if k.startswith("y_true__")]
    pk = [k for k in keys if k.startswith("y_pred__")]
    base = next((k for k in pk if "Synth" not in k), pk[0])
    aug = next((k for k in pk if "Synth" in k), pk[-1])
    prb = {k.split("__", 1)[1]: d[k] for k in keys if k.startswith("proba__")}
    return (d[tk[0]], d[base], d[aug],
            prb.get(base.split("__", 1)[1]), prb.get(aug.split("__", 1)[1]), None)


def g_mean(y, p, labels):
    r = recall_score(y, p, labels=labels, average=None, zero_division=0)
    return float(np.prod(r) ** (1.0 / len(r)))


def per_class_fpr(y, p, c):
    cm = confusion_matrix((y == c).astype(int), (p == c).astype(int), labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    return fp / (fp + tn) if (fp + tn) else 0.0


def global_metrics(y, p, labels):
    return {
        "Matthews correlation coefficient": matthews_corrcoef(y, p),
        "Cohen's kappa": cohen_kappa_score(y, p),
        "Balanced accuracy": balanced_accuracy_score(y, p),
        "G-mean (per-class recall)": g_mean(y, p, labels),
        "Macro F1": f1_score(y, p, average="macro", zero_division=0),
    }


def main():
    y, yb, ya, pb, pa, cls = load(find_predictions())
    y, yb, ya = np.asarray(y), np.asarray(yb), np.asarray(ya)
    labels = sorted(set(y.tolist()))
    if cls is None:
        cls = np.array(labels)

    rows = ["**Table 7.** Extended evaluation metrics, baseline versus augmented, "
            "on the held-out test partition.", "",
            "| Metric | Baseline | Augmented | Δ |", "|---|---|---|---|"]

    mb, ma = global_metrics(y, yb, labels), global_metrics(y, ya, labels)
    for k in mb:
        rows.append(f"| {k} | {mb[k]:.4f} | {ma[k]:.4f} | {ma[k]-mb[k]:+.4f} |")

    for c in FOCUS:
        if c not in labels:
            continue
        rb = recall_score(y, yb, labels=[c], average="micro", zero_division=0)
        ra = recall_score(y, ya, labels=[c], average="micro", zero_division=0)
        qb = precision_score(y, yb, labels=[c], average="micro", zero_division=0)
        qa = precision_score(y, ya, labels=[c], average="micro", zero_division=0)
        fb, fa = per_class_fpr(y, yb, c), per_class_fpr(y, ya, c)

        rows.append(f"| **{c}** — recall | {rb:.4f} | {ra:.4f} | {ra-rb:+.4f} |")
        rows.append(f"| — precision | {qb:.4f} | {qa:.4f} | {qa-qb:+.4f} |")
        rows.append(f"| — false positive rate | {fb:.6f} | {fa:.6f} | {fa-fb:+.6f} |")

        if pb is not None and pa is not None:
            try:
                j = list(cls).index(c)
                binary = (y == c).astype(int)
                for tag, arr in (("ROC-AUC", roc_auc_score), ("PR-AUC", average_precision_score)):
                    vb, va = arr(binary, pb[:, j]), arr(binary, pa[:, j])
                    rows.append(f"| — {tag} | {vb:.4f} | {va:.4f} | {va-vb:+.4f} |")
            except (ValueError, IndexError):
                pass

    table = "\n".join(rows)
    open(os.path.join(OUT_DIR, "table7_extended_metrics.md"), "w",
         encoding="utf-8").write(table)
    print(table)

    if pb is None or pa is None:
        print("\n[NOTE] No probability arrays saved, so ROC and PR curves were skipped.")
        print("       Add predict_proba output to the pipeline export to enable them.")
        return

    # ---- Figure 5: ROC ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, c in zip(axes, FOCUS[:2]):
        if c not in labels:
            continue
        j = list(cls).index(c)
        binary = (y == c).astype(int)
        for arr, lbl, style in ((pb, "Baseline", "--"), (pa, "Augmented", "-")):
            fpr, tpr, _ = roc_curve(binary, arr[:, j])
            ax.plot(fpr, tpr, style,
                    label=f"{lbl} (AUC = {roc_auc_score(binary, arr[:, j]):.3f})")
        ax.plot([0, 1], [0, 1], ":", color="grey", lw=0.8)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title(c, fontsize=11); ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Figure5_ROC.png"), dpi=300)
    plt.close()

    # ---- Figure 8: precision-recall ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, c in zip(axes, FOCUS[:2]):
        if c not in labels:
            continue
        j = list(cls).index(c)
        binary = (y == c).astype(int)
        for arr, lbl, style in ((pb, "Baseline", "--"), (pa, "Augmented", "-")):
            pr, rc, _ = precision_recall_curve(binary, arr[:, j])
            ax.plot(rc, pr, style,
                    label=f"{lbl} (AP = {average_precision_score(binary, arr[:, j]):.3f})")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title(c, fontsize=11); ax.legend(fontsize=9); ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Figure8_PR_curves.png"), dpi=300)
    plt.close()

    print(f"\n[DONE] Written to {OUT_DIR}:")
    print("   table7_extended_metrics.md")
    print("   Figure5_ROC.png")
    print("   Figure8_PR_curves.png")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        print("\n[FAILED] Send the error above to Claude.")
