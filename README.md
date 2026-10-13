# Cyber-Synth-CGAN

Code for the paper **"Enhanced Network Intrusion Detection through Conditional GAN-based Minority Class Augmentation"** (Babylonian Journal of Artificial Intelligence).

Cyber-Synth-CGAN is a conditional Wasserstein GAN with gradient penalty (CWGAN-GP) that generates synthetic network flows for rare attack classes in CIC-IDS-2017. A sample-size gate decides which classes are augmented. A Random Forest is then trained with and without the synthetic data and evaluated on a held-out test partition that contains only real flows.

## Leakage control

- The data are split 80/20 (stratified, seed 42) **before** any preprocessing.
- The signed log-transform and min–max scaler are fitted on the training partition only.
- The GAN is trained on the training partition only, and synthetic samples are added to the training set only.
- All reported metrics are computed on the untouched real test partition (565,576 flows).

## Repository contents

| File | Purpose |
|---|---|
| `CyberSynth-CGAN v.2.py` | Main pipeline: loading, preprocessing, CWGAN-GP training, augmentation, KS/MMD assessment, evaluation |
| `RUN_ME.py` | One full run (seed 42, 2,000-epoch budget with early stopping). Produces the per-class results of Tables IV and V and the counts in Figure 4 |
| `MULTISEED.py` | Repeats the pipeline for seeds 42–46 and produces Table VI (mean ± SD, 95% CI, paired t-tests with Holm–Bonferroni correction) |
| `BASELINES.py` | No augmentation, random oversampling and SMOTE on the same partition (Table VIII, rows 1–3) |
| `CAPTURE_PROBA.py` | Re-runs the pipeline and saves class probabilities, needed for ROC and precision–recall analysis |
| `METRICS_AND_FIGURES.py` | Extended metrics (MCC, kappa, balanced accuracy, ROC-AUC, PR-AUC, FPR) and the ROC figure (Figure 5) |
| `logs/` | Console logs of the runs reported in the paper |
| `requirements.txt` | Python dependencies |

## Data

Download the **MachineLearningCSV** version of CIC-IDS-2017 (8 CSV files) from the Canadian Institute for Cybersecurity:
https://www.unb.ca/cic/datasets/ids-2017.html

Place the CSV files in `data/CIC-IDS-2017/`, or set the environment variable `CICIDS2017_DIR` to their folder. The dataset is not included in this repository.

## Installation

Python 3.10 or later. A GPU is not required; the reported runs used an Intel Core i7 CPU with 16 GB RAM.

```bash
pip install -r requirements.txt
```

## Reproducing the results

Run each script from the repository folder. Outputs are written to `results/` (or to the folder in the environment variable `RESULTS_DIR`).

```bash
python -u RUN_ME.py               # Tables IV, V, VII and Figure 4 counts  (~48 min on CPU)
python -u MULTISEED.py            # Table VI, seeds 42-46                  (~4 h on CPU)
python -u BASELINES.py            # Table VIII, rows 1-3                    (~15 min)
python -u CAPTURE_PROBA.py        # saves probabilities                     (~50 min)
python -u METRICS_AND_FIGURES.py  # ROC/PR-AUC metrics and Figure 5         (seconds)
```

`MULTISEED.py` saves after every seed and resumes automatically if interrupted.

## Expected results (seed 42)

| Class | Training samples | Baseline recall | Augmented recall | Δ (pp) |
|---|---|---|---|---|
| Web Attack: Brute Force | 1,206 | 68.77% | 98.34% | +29.57 |
| Web Attack: XSS | 522 | 58.46% | 4.62% | −53.84 |
| Bot | 1,565 | 100.00% | 99.49% | −0.51 |

Across seeds 42–46, Brute Force recall improves by 29.90 points (95% CI [27.80, 32.00], Holm-corrected p < 0.0001). Early stopping occurred at epoch 1,531 in the seed-42 run, and the full pipeline took 47.6 minutes on CPU.

Small numerical differences can occur across library versions and hardware, because GAN training is stochastic even with fixed seeds.

## Notes

- The KS and MMD statistics are reported as a diagnostic; synthetic samples are not filtered on them. In the reported runs all six augmented classes fail both tests.
- The 15,000-sample cap and the SMOTE branch for classes below 50 samples are present in the code but were not triggered in the reported runs.
- On CPU the pipeline automatically switches to stability-adjusted hyperparameters (generator learning rate 1.5×10⁻⁴, discriminator 8×10⁻⁵, λ = 5, n_critic = 3, feature-matching weight 0.05); all reported results use these values.

## Citation

If you use this code, please cite the paper (citation details will be added on publication).

## License

Released under the MIT License (see `LICENSE`).
