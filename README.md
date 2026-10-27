# Transformer-based Automatic Modulation Classification (AMC) on Raw I/Q Data

A ViT-style transformer encoder that classifies the modulation scheme of a radio signal directly from raw I/Q samples, trained with a staged transfer-learning pipeline across public RadioML datasets (with real USRP captures planned as the final stage).

**Classes (7):** BPSK, QPSK, 8PSK, QAM16, QAM64, AM-DSB, AM-SSB

---

## Highlights

- Works on **raw time-domain I/Q** only. No spectrograms, constellation images or hand-crafted features.
- **Staged transfer learning:** RadioML2016.10a, then RadioML2018.01A, then real USRP captures (Stage 3, in progress).
- Fine-tuning on RadioML2018.01A improves on zero-shot transfer by roughly **25-30 percentage points at high SNR**.
- Evaluation reports **accuracy vs SNR** and **confusion matrices**, not just a single overall number.
- `AMCTransformerV2` (RoPE, cross-layer fusion, attention pooling) is implemented and smoke-tested; its training results are pending.

---

## Results (Iteration 1 baseline)

| Stage | Dataset | Overall test accuracy | High SNR (>= 10 dB) |
|---|---|---|---|
| Stage 1 (from scratch) | RadioML2016.10a | **61.77%** | ~89-90% |
| Zero-shot (Stage 1 model on 2018 data, no fine-tuning) | RadioML2018.01A | not recorded | ~58-64% |
| Stage 2 (fine-tuned) | RadioML2018.01A | **60.86%** | ~85-93% |

How to read these numbers:

- "Overall" accuracy averages over every SNR (-20 dB to +18 dB for 2016.10a, -20 dB to +30 dB for 2018.01A). At very low SNR the signal is buried in noise and accuracy sits near random guessing (1/7, about 14%), which pulls the average down for any model.
- The overall figures are in the same range as published lightweight raw-I/Q transformers on these benchmarks (for example, MoEformer reports about 64% on RadioML2018.01A).
- The main remaining weakness is **QAM16 vs QAM64** confusion, which persists even at high SNR (about 65-71% per-class on the Stage 1 high-SNR confusion matrix).

Plots are in [`results/`](results/): accuracy-vs-SNR curves, confusion matrices, and the zero-shot vs fine-tuned overlay.

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── Data_Utils.py        # loading, class filtering, splitting, normalization (2016.10a pickle + 2018.01A HDF5)
├── Model_1.py             # AMCTransformer (Iteration 1 baseline)
├── model_v2.py          # AMCTransformerV2 (Iteration 2: RoPE + cross-layer fusion + attention pooling)
├── Train_1.py             # training loop (AdamW, warmup + cosine LR, grad clipping, best-checkpoint saving)
├── evaluate.py          # per-SNR accuracy, confusion matrices, zero-shot evaluation
├── AMCr.ipynb   # Colab notebook that drives the full pipeline
└── results/             # figures
```

---

## Datasets

The datasets are **not** included in this repo (the files are far too large for GitHub). Download them yourself:

| Dataset | Format | Size | Source |
|---|---|---|---|
| RadioML2016.10a | pickle (`RML2016.10a_dict.pkl`) | ~611 MB | DeepSig (deepsig.ai/datasets) |
| RadioML2018.01A | HDF5 (`GOLD_XYZ_OSC.0001_1024.hdf5`) + `classes-fixed.json` | ~20 GB | DeepSig, or Kaggle mirror `pinxau1000/radioml2018` |

Licence note: the RadioML 2018 Kaggle mirror is listed as CC BY-NC-SA 4.0. Check the licence of each dataset before reuse.

### Class mapping across datasets

The two datasets name some classes differently. `data_utils.py` translates them:

| This project | RadioML2016.10a | RadioML2018.01A |
|---|---|---|
| QAM16 | QAM16 | 16QAM |
| QAM64 | QAM64 | 64QAM |
| AM-DSB | AM-DSB | AM-DSB-WC |
| AM-SSB | AM-SSB | AM-SSB-WC |
| BPSK, QPSK, 8PSK | same | same |

> **Known pitfall (fixed):** an early version hard-coded the wrong class order for RadioML2018.01A, which silently selected and labelled the wrong classes. `RADIOML2018_CLASS_ORDER` in `data_utils.py` is now the order verified against the dataset's own `classes-fixed.json`. Always run `inspect_2018_classes()` on your copy before filtering.

---

## Setup

```bash
pip install -r requirements.txt
```

Training was done on a free Google Colab T4 GPU. The code also runs on CPU, just slowly.

---

## Pipeline

### 1. Prepare RadioML2016.10a

```python
from data_utils import prepare_2016_dataset

splits = prepare_2016_dataset(
    pkl_path="RML2016.10a_dict.pkl",
    save_path="radioml2016_prepped.npz",
)
```

This loads the pickle, keeps the 7 target classes (140,000 captures), does a **70/15/15 split stratified by class and SNR**, converts to a 2-channel real array `(N, 2, 128)`, and normalizes each capture to unit average power.

### 2. Stage 1: train from scratch

```python
from data_utils import load_prepped
from train import train_model

splits, class_names = load_prepped("radioml2016_prepped.npz")
model, history = train_model(
    splits, num_classes=len(class_names), seq_len=128,
    epochs=30, checkpoint_path="amc_transformer_stage1.pt",
)
```

Settings: AdamW (lr 3e-4, weight decay 0.05), 3 warmup epochs then cosine decay, batch size 256, gradient clipping at 1.0.

### 3. Prepare RadioML2018.01A

```python
from data_utils import inspect_2018_classes, prepare_2018_dataset

inspect_2018_classes("GOLD_XYZ_OSC.0001_1024.hdf5")   # verify class order first

splits_2018 = prepare_2018_dataset(
    hdf5_path="GOLD_XYZ_OSC.0001_1024.hdf5",
    save_path="radioml2018_prepped.npz",
    max_samples_per_class=2000,     # 14,000 captures total
)
```

### 4. Stage 2: transfer learning

The 2018 captures are 1024 samples long (vs 128), so the Stage 1 positional encoding no longer fits. The transfer therefore:

1. Builds a fresh model with `seq_len=1024` and the same `patch_size=4`, so the patch-embedding weights keep the same shape.
2. Loads every Stage 1 tensor whose shape still matches, and leaves the positional encoding randomly initialised.
3. Freezes the encoder, keeps the positional encoding and CLS token trainable, and unfreezes the last 2 encoder layers.
4. Fine-tunes for 15 epochs at lr 5e-5.

### 5. Evaluate

`evaluate.py` provides `evaluate_model()` (per-SNR accuracy plus confusion matrices) and `zero_shot_evaluate()` (the no-fine-tuning baseline used to quantify the benefit of transfer learning).

---

## Models

### `AMCTransformer` (Iteration 1)

1. Split `(2, L)` I/Q into non-overlapping patches of `patch_size` samples and linearly project each to `d_model`.
2. Prepend a learnable CLS token and add learnable positional encodings.
3. 6 pre-norm transformer encoder layers, 8 heads, GELU, `d_model=128`.
4. Classify from the CLS token with a small MLP head.

About 0.8M parameters.

### `AMCTransformerV2` (Iteration 2)

Three targeted changes, each taken from recent AMC literature:

| Change | Idea from | Why |
|---|---|---|
| **RoPE** replaces learnable positional encoding | MoEformer | Relative, rotation-based position suits the phase rotation in I/Q signals, and it has no length-specific weights to discard when sequence length changes (the problem hit in Stage 1 to Stage 2 transfer) |
| **Cross-layer fusion** of CLS outputs from layers 2, 4 and 6 | MSFF-ViT | Keeps shallow features available to the classifier |
| **Attention pooling** over all tokens | MoEformer | Lets the model weight informative time segments instead of relying only on the CLS token |

About 0.93M parameters. It is a different architecture, so it cannot load Iteration 1 checkpoints and needs its own Stage 1 run.

Deliberately **not** included, because of implementation cost against small documented gains: KAN classifier, MHSSA subspace attention, multi-modal image inputs (constellation, STFT, power spectrum), mixture-of-experts routing.

---

## Status and roadmap

- [x] Data pipeline for both datasets, with verified class mapping
- [x] Stage 1 (RadioML2016.10a)
- [x] Zero-shot baseline and Stage 2 fine-tuning (RadioML2018.01A)
- [x] Accuracy-vs-SNR curves and confusion matrices
- [ ] Iteration 2 (`AMCTransformerV2`): Stage 1 training and comparison with the baseline
- [ ] Stage 3: fine-tuning on real USRP captures
- [ ] Robustness to satellite-link Doppler (large carrier frequency offsets, time-varying Doppler rate). RadioML includes small simulated offsets, but LEO Doppler is much larger, so this is untested.

---

## Limitations

- Overall accuracy of about 61% is bounded by the low-SNR half of the evaluation range; it is not comparable to papers that quote high-SNR-only numbers.
- QAM16 vs QAM64 is still confused at high SNR.
- Stage 2 uses a 14,000-capture subsample of RadioML2018.01A (2,000 per class), not the full dataset.
- Results come from a single run per configuration; no seed variance is reported.
- RadioML data is simulated or collected under specific conditions, so performance on other hardware is not guaranteed. Stage 3 exists to measure exactly that gap.

---

## References

- O'Shea & West, *Radio Machine Learning Dataset Generation with GNU Radio*, 2016 (RadioML2016.10a).
- O'Shea, Roy & Clancy, *Over-the-Air Deep Learning Based Radio Signal Classification*, IEEE JSTSP, 2018 (RadioML2018.01A).
- Wu, Zheng & Liu, *Automatic modulation recognition using vision transformers with cross-layer feature fusion and Kolmogorov-Arnold representation*, Wireless Networks, 2025 (MSFF-ViT).
- Hu et al., *MFCA-Transformer: Modulation Signal Recognition Based on Multidimensional Feature Fusion*, Sensors, 2025.
- *Mixture-of-Experts Transformer for Automatic Modulation Recognition* (MoEformer), arXiv:2606.09085.
- Vaswani et al., *Attention Is All You Need*, 2017.

---


