"""
data_utils.py

Data loading, filtering, and preprocessing utilities for the AMC
(Automatic Modulation Classification) project.

Currently supports RadioML2016.10a (pickle format). A loader for
RadioML2018.01A (HDF5 format) will be added once that dataset is
verified (check exact class name strings before filtering -- see
project notes on AM-DSB-SC vs AM-DSB-WC naming).

Target modulation classes for this project (intersection of both
datasets, GFSK/GMSK excluded per project scope):
    BPSK, QPSK, 8PSK, QAM16, QAM64, AM-DSB, AM-SSB

Usage:
    from data_utils import prepare_2016_dataset

    splits = prepare_2016_dataset(
        pkl_path="RML2016.10a_dict.pkl",
        save_path="radioml2016_prepped.npz",
    )
"""

import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TARGET_CLASSES = ["BPSK", "QPSK", "8PSK", "QAM16", "QAM64", "AM-DSB", "AM-SSB"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_radioml2016(pkl_path: str) -> Dict[Tuple[str, int], np.ndarray]:
    """Load the raw RadioML2016.10a pickle file.

    Returns a dict keyed by (modulation: str, snr_db: int), each value
    an array of shape (1000, 2, 128) -- 1000 captures, 2 channels (I, Q),
    128 time samples.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data


def filter_classes(
    data: Dict[Tuple[str, int], np.ndarray],
    target_classes: List[str] = TARGET_CLASSES,
) -> Dict[Tuple[str, int], np.ndarray]:
    """Keep only the (modulation, snr) entries whose modulation is in
    target_classes. Raises if any requested class is entirely absent,
    since that usually means a naming mismatch rather than missing data.
    """
    found_classes = {k[0] for k in data.keys()}
    missing = set(target_classes) - found_classes
    if missing:
        raise ValueError(
            f"These target classes were not found in the dataset: {missing}. "
            f"Available classes are: {sorted(found_classes)}. "
            "Check for naming mismatches (e.g. 'QAM16' vs '16QAM')."
        )

    filtered = {k: v for k, v in data.items() if k[0] in target_classes}
    return filtered


# ---------------------------------------------------------------------------
# Flattening / formatting
# ---------------------------------------------------------------------------

def flatten_dataset(
    filtered_data: Dict[Tuple[str, int], np.ndarray],
    target_classes: List[str] = TARGET_CLASSES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Flatten the (modulation, snr)-keyed dict into parallel arrays.

    Returns:
        X   : complex IQ array, shape (N, 128)
        y   : integer class labels, shape (N,)
        snr : SNR in dB for each capture, shape (N,)
        class_to_idx : mapping from class name to integer label,
                       sorted alphabetically for reproducibility
    """
    class_to_idx = {c: i for i, c in enumerate(sorted(target_classes))}

    X_list, y_list, snr_list = [], [], []
    for (mod, snr), captures in filtered_data.items():
        # captures shape: (1000, 2, 128) -> complex (1000, 128)
        iq_complex = captures[:, 0, :] + 1j * captures[:, 1, :]
        X_list.append(iq_complex)
        y_list.extend([class_to_idx[mod]] * len(iq_complex))
        snr_list.extend([snr] * len(iq_complex))

    X = np.concatenate(X_list, axis=0)
    y = np.array(y_list)
    snr = np.array(snr_list)
    return X, y, snr, class_to_idx


def to_2channel(X_complex: np.ndarray) -> np.ndarray:
    """Convert complex IQ array (N, L) to real 2-channel array (N, 2, L)
    for feeding into a Conv1d model that expects real-valued input.
    """
    return np.stack([X_complex.real, X_complex.imag], axis=1)


def normalize_power(X_2ch: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize each capture to unit average power.

    X_2ch shape: (N, 2, L). Power is computed per-capture across both
    channels and the time axis, then each capture is scaled so its
    average power is 1.
    """
    power = np.mean(X_2ch ** 2, axis=(1, 2), keepdims=True)
    return X_2ch / np.sqrt(power + eps)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    snr: np.ndarray,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """Split into train/val/test, stratified jointly on (class, snr) so
    every split has balanced representation across both modulation type
    and difficulty level (SNR).
    """
    strat_key = y * 1000 + snr  # combined key, works since snr is bounded

    temp_size = val_size + test_size
    X_train, X_temp, y_train, y_temp, snr_train, snr_temp = train_test_split(
        X, y, snr, test_size=temp_size, stratify=strat_key, random_state=random_state
    )

    relative_test_size = test_size / temp_size
    strat_key_temp = y_temp * 1000 + snr_temp
    X_val, X_test, y_val, y_test, snr_val, snr_test = train_test_split(
        X_temp, y_temp, snr_temp,
        test_size=relative_test_size,
        stratify=strat_key_temp,
        random_state=random_state,
    )

    return {
        "X_train": X_train, "y_train": y_train, "snr_train": snr_train,
        "X_val": X_val, "y_val": y_val, "snr_val": snr_val,
        "X_test": X_test, "y_test": y_test, "snr_test": snr_test,
    }


# ---------------------------------------------------------------------------
# Save / load prepped data
# ---------------------------------------------------------------------------

def save_prepped(splits: Dict[str, np.ndarray], class_to_idx: Dict[str, int], save_path: str) -> None:
    """Save prepped, 2-channel, normalized splits plus the class mapping
    to a single .npz file (upload this to Drive so Colab doesn't need to
    reparse the raw pkl every session).
    """
    np.savez(
        save_path,
        X_train=splits["X_train"], y_train=splits["y_train"], snr_train=splits["snr_train"],
        X_val=splits["X_val"], y_val=splits["y_val"], snr_val=splits["snr_val"],
        X_test=splits["X_test"], y_test=splits["y_test"], snr_test=splits["snr_test"],
        class_names=np.array(sorted(class_to_idx, key=class_to_idx.get)),
    )


def load_prepped(npz_path: str) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Load a previously prepped .npz file. Returns (splits_dict, class_names)."""
    data = np.load(npz_path, allow_pickle=True)
    splits = {k: data[k] for k in data.files if k != "class_names"}
    class_names = list(data["class_names"])
    return splits, class_names


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def prepare_2016_dataset(
    pkl_path: str,
    save_path: str = None,
    target_classes: List[str] = TARGET_CLASSES,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """Full pipeline: load -> filter -> flatten -> split -> 2-channel ->
    normalize -> (optionally) save.

    Returns the splits dict with 2-channel, power-normalized arrays,
    ready to feed directly into a PyTorch Dataset/DataLoader.
    """
    print(f"Loading {pkl_path} ...")
    raw = load_radioml2016(pkl_path)

    print(f"Filtering to target classes: {target_classes}")
    filtered = filter_classes(raw, target_classes)

    print("Flattening dataset ...")
    X, y, snr, class_to_idx = flatten_dataset(filtered, target_classes)
    print(f"Total captures: {len(X)} across {len(class_to_idx)} classes")

    print("Splitting (stratified by class + SNR) ...")
    splits = stratified_split(X, y, snr, val_size=val_size, test_size=test_size, random_state=random_state)

    print("Converting to 2-channel real arrays and normalizing power ...")
    for split_name in ["train", "val", "test"]:
        X_key = f"X_{split_name}"
        splits[X_key] = normalize_power(to_2channel(splits[X_key]))

    for split_name in ["train", "val", "test"]:
        n = len(splits[f"y_{split_name}"])
        print(f"  {split_name}: {n} captures, shape {splits[f'X_{split_name}'].shape}")

    if save_path:
        print(f"Saving prepped data to {save_path} ...")
        save_prepped(splits, class_to_idx, save_path)

    splits["class_to_idx"] = class_to_idx
    return splits


# ---------------------------------------------------------------------------
# RadioML2018.01A (HDF5) support
# ---------------------------------------------------------------------------

# DeepSig's official 24-class order for GOLD_XYZ_OSC.0001_1024.hdf5.
# VERIFIED against the user's actual classes-fixed.json (2026-07-30) --
# this is the real, confirmed order, not a guess.
RADIOML2018_CLASS_ORDER = [
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
    "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
    "FM", "GMSK", "OQPSK",
]

# Mapping from 2016.10a-style names (used in TARGET_CLASSES) to their
# 2018.01A equivalents, where naming differs.
CLASS_NAME_2016_TO_2018 = {
    "QAM16": "16QAM",
    "QAM64": "64QAM",
    "AM-DSB": "AM-DSB-WC",  # "with carrier" variant is the closer analogue; verify against your data
    "AM-SSB": "AM-SSB-WC",  # same caveat
}


def inspect_2018_classes(hdf5_path: str) -> List[str]:
    """Print and return the actual class order stored in a 2018.01A file's
    classes-fixed.json (if present alongside the hdf5) or fall back to the
    hardcoded RADIOML2018_CLASS_ORDER. ALWAYS run this first and eyeball
    the output before filtering -- do not assume the hardcoded list above
    is correct for your specific download.
    """
    import os
    import json

    json_path = os.path.join(os.path.dirname(hdf5_path), "classes-fixed.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            classes = json.load(f)
        print(f"Loaded class order from {json_path}:")
    else:
        classes = RADIOML2018_CLASS_ORDER
        print(
            "No classes-fixed.json found next to the hdf5 file. "
            "Falling back to hardcoded class order -- VERIFY this manually "
            "against DeepSig's documentation before proceeding:"
        )
    for i, c in enumerate(classes):
        print(f"  {i:2d}: {c}")
    return classes


def load_radioml2018(
    hdf5_path: str,
    target_classes_2016_style: List[str] = TARGET_CLASSES,
    class_order: Optional[List[str]] = None,
    max_samples_per_class: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Load and filter RadioML2018.01A directly to your target classes.

    target_classes_2016_style: names using the same convention as
        TARGET_CLASSES (e.g. 'QAM16', 'AM-DSB'). These get translated to
        2018.01A's actual naming via CLASS_NAME_2016_TO_2018 where needed.
    class_order: pass the verified list from inspect_2018_classes() if the
        hardcoded RADIOML2018_CLASS_ORDER doesn't match your file.
    max_samples_per_class: optional cap, useful for a fast first pass given
        this dataset is much larger than 2016.10a.

    Returns (X, y, snr, class_to_idx) in the same format flatten_dataset()
    produces for 2016.10a, so downstream code (splitting, normalization)
    is identical either way.
    """
    import h5py

    class_order = class_order or RADIOML2018_CLASS_ORDER
    class_to_idx_2018 = {c: i for i, c in enumerate(class_order)}

    # translate 2016-style target names to their 2018.01A equivalents
    target_2018_names = [
        CLASS_NAME_2016_TO_2018.get(c, c) for c in target_classes_2016_style
    ]
    missing = [c for c in target_2018_names if c not in class_to_idx_2018]
    if missing:
        raise ValueError(
            f"These translated class names were not found in class_order: {missing}. "
            f"Available: {class_order}. Run inspect_2018_classes() and check "
            "CLASS_NAME_2016_TO_2018 mappings."
        )
    target_indices_2018 = {class_to_idx_2018[c] for c in target_2018_names}

    print(f"Opening {hdf5_path} (this can take a moment for a large file) ...")
    f = h5py.File(hdf5_path, "r")
    Y = f["Y"]  # one-hot, shape (N, 24)
    mod_idx_all = np.argmax(Y, axis=1)  # this loads Y fully into memory; Y is small enough to be fine

    mask = np.isin(mod_idx_all, list(target_indices_2018))
    matched_positions = np.where(mask)[0]
    print(f"Found {len(matched_positions)} captures matching target classes.")

    if max_samples_per_class:
        # subsample per class to keep things manageable
        keep = []
        for idx in target_indices_2018:
            class_positions = matched_positions[mod_idx_all[matched_positions] == idx]
            if len(class_positions) > max_samples_per_class:
                class_positions = np.random.default_rng(42).choice(
                    class_positions, max_samples_per_class, replace=False
                )
            keep.append(class_positions)
        matched_positions = np.sort(np.concatenate(keep))
        print(f"Subsampled to {len(matched_positions)} captures (max {max_samples_per_class}/class).")

    # h5py requires sorted, increasing indices for fancy indexing -- matched_positions
    # from np.where is already sorted, so this is safe.
    X_raw = f["X"][matched_positions]  # (M, 1024, 2) real, [I, Q] last axis
    snr_raw = f["Z"][matched_positions].flatten()
    mod_idx_matched = mod_idx_all[matched_positions]
    f.close()

    X_complex = X_raw[:, :, 0] + 1j * X_raw[:, :, 1]  # (M, 1024)

    # remap 2018-file class indices -> our own alphabetical class_to_idx,
    # consistent with flatten_dataset()'s convention, keyed by the ORIGINAL
    # 2016-style names so labels line up across both datasets.
    class_to_idx = {c: i for i, c in enumerate(sorted(target_classes_2016_style))}
    idx_2018_to_name_2016 = {}
    for name_2016 in target_classes_2016_style:
        name_2018 = CLASS_NAME_2016_TO_2018.get(name_2016, name_2016)
        idx_2018_to_name_2016[class_to_idx_2018[name_2018]] = name_2016

    y = np.array([class_to_idx[idx_2018_to_name_2016[i]] for i in mod_idx_matched])

    return X_complex, y, snr_raw, class_to_idx


def prepare_2018_dataset(
    hdf5_path: str,
    save_path: str = None,
    target_classes: List[str] = TARGET_CLASSES,
    class_order: Optional[List[str]] = None,
    max_samples_per_class: Optional[int] = None,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """Full pipeline for 2018.01A, mirroring prepare_2016_dataset(): load
    + filter -> split -> 2-channel -> normalize -> (optionally) save.
    """
    print(f"Loading and filtering {hdf5_path} ...")
    X, y, snr, class_to_idx = load_radioml2018(
        hdf5_path, target_classes, class_order, max_samples_per_class
    )
    print(f"Total captures: {len(X)} across {len(class_to_idx)} classes")

    print("Splitting (stratified by class + SNR) ...")
    splits = stratified_split(X, y, snr, val_size=val_size, test_size=test_size, random_state=random_state)

    print("Converting to 2-channel real arrays and normalizing power ...")
    for split_name in ["train", "val", "test"]:
        X_key = f"X_{split_name}"
        splits[X_key] = normalize_power(to_2channel(splits[X_key]))

    for split_name in ["train", "val", "test"]:
        n = len(splits[f"y_{split_name}"])
        print(f"  {split_name}: {n} captures, shape {splits[f'X_{split_name}'].shape}")

    if save_path:
        print(f"Saving prepped data to {save_path} ...")
        save_prepped(splits, class_to_idx, save_path)

    splits["class_to_idx"] = class_to_idx
    return splits


if __name__ == "__main__":
    import sys

    pkl_path = sys.argv[1] if len(sys.argv) > 1 else "RML2016.10a_dict.pkl"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "radioml2016_prepped.npz"

    splits = prepare_2016_dataset(pkl_path, save_path=out_path)
    print("\nDone. class_to_idx mapping:")
    print(splits["class_to_idx"])
