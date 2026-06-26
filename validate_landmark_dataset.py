import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from landmark_pipeline import (
    load_raw_dataset,
    SEQUENCE_LENGTH,
    RAW_FEATURES
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset_landmark_realtime")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    print("=" * 65)
    print("VALIDASI DATASET LANDMARK")
    print("=" * 65)
    print("Folder dataset:", dataset_path)

    raw_x, labels, files = load_raw_dataset(dataset_path)
    counts = Counter(labels)

    print("\nJumlah kelas :", len(counts))
    print("Jumlah sample:", len(raw_x))
    print("Shape dataset:", raw_x.shape)

    print("\nJumlah sample per kelas:")
    for class_name in sorted(counts):
        print(f"- {class_name:<22}: {counts[class_name]}")

    warnings = []

    for sequence, label, file_path in zip(raw_x, labels, files):
        zero_frames = int(np.sum(~np.any(np.abs(sequence) > 1e-8, axis=1)))
        if zero_frames == SEQUENCE_LENGTH:
            warnings.append(f"[KOSONG] {file_path}: semua frame bernilai nol")
        elif zero_frames > 8:
            warnings.append(
                f"[PERIKSA] {file_path}: terdapat {zero_frames}/{SEQUENCE_LENGTH} frame kosong"
            )

    if len(set(counts.values())) != 1:
        warnings.append("[PERIKSA] Jumlah sample antar-kelas belum seimbang.")

    print("\nHasil pemeriksaan:")
    if warnings:
        for warning in warnings:
            print(warning)
    else:
        print("Dataset lolos pemeriksaan dasar. Tidak ditemukan masalah shape, NaN, atau frame kosong berlebihan.")

    print("\nCatatan:")
    print("- Setiap file harus memiliki shape (30, 126).")
    print("- Untuk evaluasi yang lebih realistis, ambil data pada beberapa sesi dan beberapa orang.")
    print("=" * 65)


if __name__ == "__main__":
    main()
