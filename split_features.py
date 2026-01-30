#!/usr/bin/env python3

import argparse
import os
from typing import Tuple

import numpy as np


def split_indices(num_samples: int, train_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if num_samples <= 0:
        raise ValueError("No samples found (num_samples <= 0)")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)

    train_size = int(np.floor(train_fraction * num_samples))
    train_size = max(1, min(train_size, num_samples - 1))

    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    return train_idx, test_idx


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly split data/features.npy into X_split_train.npy and X_split_test.npy "
            "(90/10 by default)."
        )
    )
    parser.add_argument(
        "--data-directory",
        default="data",
        help="Directory containing features.npy and where outputs will be written (default: data)",
    )
    parser.add_argument(
        "--input",
        default="features.npy",
        help="Input .npy filename inside --data-directory (default: features.npy)",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.9,
        help="Fraction of samples to put in the train split (default: 0.9)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible shuffling (default: 42)",
    )
    parser.add_argument(
        "--train-output",
        default="X_split_train.npy",
        help="Output filename for the train split inside --data-directory",
    )
    parser.add_argument(
        "--test-output",
        default="X_split_test.npy",
        help="Output filename for the test split inside --data-directory",
    )

    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_directory)
    input_path = os.path.join(data_dir, args.input)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    os.makedirs(data_dir, exist_ok=True)

    try:
        features = np.load(input_path)
    except ValueError:
        features = np.load(input_path, allow_pickle=True)

    if not hasattr(features, "shape") or len(features.shape) == 0:
        raise ValueError(f"Expected an array-like .npy with a leading sample dimension, got: {type(features)}")

    num_samples = int(features.shape[0])
    train_idx, test_idx = split_indices(num_samples, args.train_fraction, args.seed)

    x_train = features[train_idx]
    x_test = features[test_idx]

    train_path = os.path.join(data_dir, args.train_output)
    test_path = os.path.join(data_dir, args.test_output)

    np.save(train_path, x_train)
    np.save(test_path, x_test)

    print(f"Loaded: {input_path} shape={features.shape}")
    print(f"Wrote:  {train_path} shape={x_train.shape}")
    print(f"Wrote:  {test_path} shape={x_test.shape}")
    print(f"Split:  train={len(train_idx)} test={len(test_idx)} seed={args.seed}")


if __name__ == "__main__":
    main()
