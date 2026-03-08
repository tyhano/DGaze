#!/usr/bin/env python3
"""
Convert Unity DGaze_ET collector CSV into trainingX/trainingY/testX/testY numpy files.

Expected CSV header (from DGazeETDataCollector.cs):
  timestamp_ms,gaze_screen_x,gaze_screen_y,gaze_angle_x,gaze_angle_y,head_vel_x,head_vel_y,
  obj1_angle_x,obj1_angle_y,obj1_dist,obj2_angle_x,obj2_angle_y,obj2_dist,obj3_angle_x,obj3_angle_y,obj3_dist
"""

import argparse
import csv
import os
from typing import List

import numpy as np


FEATURE_COLUMNS = [
    "gaze_angle_x",
    "gaze_angle_y",
    "head_vel_x",
    "head_vel_y",
    "obj1_angle_x",
    "obj1_angle_y",
    "obj1_dist",
    "obj2_angle_x",
    "obj2_angle_y",
    "obj2_dist",
    "obj3_angle_x",
    "obj3_angle_y",
    "obj3_dist",
]

LABEL_COLUMNS = ["gaze_angle_x", "gaze_angle_y"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DGaze_ET CSV to NPY dataset files")
    parser.add_argument("--input_csv", required=True, help="Path to CSV exported by DGazeETDataCollector")
    parser.add_argument("--output_dir", required=True, help="Output directory for trainingX/Y and testX/Y")
    parser.add_argument("--seq_length", type=int, default=50, help="History sequence length (default: 50)")
    parser.add_argument(
        "--prediction_offset",
        type=int,
        default=10,
        help="Future frame offset for label (default: 10 for 100Hz -> 100ms)",
    )
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for split")
    return parser.parse_args()


def _safe_float(text: str) -> float:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return value


def load_rows(csv_path: str) -> np.ndarray:
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in FEATURE_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        frames: List[List[float]] = []
        for row in reader:
            features = [_safe_float(row.get(name, "0")) for name in FEATURE_COLUMNS]
            frames.append(features)

    if not frames:
        raise ValueError("No data rows found in CSV")

    return np.asarray(frames, dtype=np.float32)


def build_samples(frames: np.ndarray, seq_length: int, prediction_offset: int):
    if seq_length <= 0:
        raise ValueError("seq_length must be > 0")
    if prediction_offset < 0:
        raise ValueError("prediction_offset must be >= 0")

    total = frames.shape[0]
    # t is the index of current frame (history window end), label uses t + prediction_offset
    start_t = seq_length - 1
    end_t = total - 1 - prediction_offset

    if end_t < start_t:
        raise ValueError(
            f"Not enough frames: got {total}, require at least seq_length + prediction_offset = {seq_length + prediction_offset}"
        )

    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for t in range(start_t, end_t + 1):
        history = frames[t - seq_length + 1 : t + 1]  # [seq_length, 13]
        x_list.append(history.reshape(-1))
        y_list.append(frames[t + prediction_offset, 0:2])  # gaze_angle_x/y

    x = np.asarray(x_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    return x, y


def split_and_save(x: np.ndarray, y: np.ndarray, output_dir: str, test_ratio: float, seed: int):
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be in (0,1)")

    os.makedirs(output_dir, exist_ok=True)
    n = x.shape[0]
    rng = np.random.RandomState(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    test_count = max(1, int(round(n * test_ratio)))
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]

    if train_idx.size == 0:
        raise ValueError("Split produced zero training samples. Increase dataset size or lower test_ratio.")

    training_x = x[train_idx]
    training_y = y[train_idx]
    test_x = x[test_idx]
    test_y = y[test_idx]

    np.save(os.path.join(output_dir, "trainingX.npy"), training_x)
    np.save(os.path.join(output_dir, "trainingY.npy"), training_y)
    np.save(os.path.join(output_dir, "testX.npy"), test_x)
    np.save(os.path.join(output_dir, "testY.npy"), test_y)

    print(f"Saved dataset to: {output_dir}")
    print(f"trainingX: {training_x.shape}, trainingY: {training_y.shape}")
    print(f"testX: {test_x.shape}, testY: {test_y.shape}")


def main():
    args = parse_args()
    frames = load_rows(args.input_csv)
    x, y = build_samples(frames, seq_length=args.seq_length, prediction_offset=args.prediction_offset)
    split_and_save(x, y, args.output_dir, test_ratio=args.test_ratio, seed=args.seed)


if __name__ == "__main__":
    main()
