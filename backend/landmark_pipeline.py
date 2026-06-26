import os
import random
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input, Masking, GRU, LSTM, Bidirectional,
    Dense, Dropout, LayerNormalization
)
from tensorflow.keras.regularizers import l2

SEQUENCE_LENGTH = 30
RAW_FEATURES = 126
PROCESSED_FEATURES = 254
EPS = 1e-8


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def normalize_one_hand(flat_hand):
    """
    Normalisasi 21 landmark tangan:
    - titik wrist menjadi pusat koordinat
    - ukuran tangan dinormalisasi agar lebih tahan terhadap perbedaan jarak kamera
    """
    flat_hand = np.asarray(flat_hand, dtype=np.float32)
    if flat_hand.shape != (63,):
        raise ValueError(f"Landmark satu tangan harus memiliki 63 nilai, ditemukan {flat_hand.shape}")

    if not np.any(np.abs(flat_hand) > EPS):
        return np.zeros(63, dtype=np.float32), 0.0

    points = flat_hand.reshape(21, 3)
    relative = points - points[0]  # wrist sebagai titik pusat

    scale = np.max(np.linalg.norm(relative[1:], axis=1))
    if scale < EPS:
        scale = 1.0

    normalized = relative / scale
    return normalized.reshape(-1).astype(np.float32), 1.0


def preprocess_sequence(raw_sequence):
    """
    Input : (30, 126)
    Output: (30, 254)

    Fitur keluaran:
    - 126 landmark tangan ternormalisasi
    - 2 flag keberadaan tangan kiri/kanan
    - 126 delta/perubahan landmark antar-frame
    """
    sequence = np.asarray(raw_sequence, dtype=np.float32)

    if sequence.shape != (SEQUENCE_LENGTH, RAW_FEATURES):
        raise ValueError(
            f"Shape sequence harus ({SEQUENCE_LENGTH}, {RAW_FEATURES}), "
            f"ditemukan {sequence.shape}"
        )

    normalized_frames = []
    presence_flags = []

    for frame in sequence:
        left, left_present = normalize_one_hand(frame[:63])
        right, right_present = normalize_one_hand(frame[63:])
        normalized_frames.append(np.concatenate([left, right]))
        presence_flags.append([left_present, right_present])

    normalized_frames = np.asarray(normalized_frames, dtype=np.float32)
    presence_flags = np.asarray(presence_flags, dtype=np.float32)

    deltas = np.zeros_like(normalized_frames, dtype=np.float32)
    deltas[1:] = normalized_frames[1:] - normalized_frames[:-1]

    return np.concatenate(
        [normalized_frames, presence_flags, deltas],
        axis=1
    ).astype(np.float32)


def augment_raw_sequence(sequence, noise_std=0.008, max_shift=2, frame_drop_prob=0.04):
    """
    Augmentasi ringan khusus data training.
    Tidak menggunakan horizontal flip karena arah tangan dapat memengaruhi makna gesture.
    """
    seq = np.asarray(sequence, dtype=np.float32).copy()

    valid = np.abs(seq) > EPS
    noise = np.random.normal(0.0, noise_std, size=seq.shape).astype(np.float32)
    seq[valid] += noise[valid]

    if max_shift > 0:
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift > 0:
            seq[shift:] = seq[:-shift]
            seq[:shift] = seq[shift]
        elif shift < 0:
            amount = abs(shift)
            seq[:-amount] = seq[amount:]
            seq[-amount:] = seq[-amount - 1]

    if frame_drop_prob > 0:
        for idx in range(1, len(seq)):
            if np.random.rand() < frame_drop_prob:
                seq[idx] = seq[idx - 1]

    return seq


def load_raw_dataset(dataset_path):
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {dataset_path}")

    class_folders = sorted([item for item in dataset_path.iterdir() if item.is_dir()])
    if not class_folders:
        raise ValueError(f"Tidak ada folder kelas di dalam: {dataset_path}")

    sequences = []
    labels = []
    files = []

    for class_folder in class_folders:
        npy_files = sorted(class_folder.glob("*.npy"))

        for npy_file in npy_files:
            try:
                arr = np.load(npy_file).astype(np.float32)
            except Exception as exc:
                raise ValueError(f"Gagal membaca {npy_file}: {exc}") from exc

            if arr.shape != (SEQUENCE_LENGTH, RAW_FEATURES):
                raise ValueError(
                    f"Shape file tidak valid: {npy_file} memiliki {arr.shape}, "
                    f"seharusnya ({SEQUENCE_LENGTH}, {RAW_FEATURES})"
                )

            if not np.all(np.isfinite(arr)):
                raise ValueError(f"Terdapat NaN atau Infinity pada file: {npy_file}")

            sequences.append(arr)
            labels.append(class_folder.name)
            files.append(str(npy_file))

    if not sequences:
        raise ValueError("Tidak ada file .npy yang ditemukan.")

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(labels),
        files
    )


def preprocess_batch(raw_batch):
    return np.asarray(
        [preprocess_sequence(sequence) for sequence in raw_batch],
        dtype=np.float32
    )


def fit_scaler_preserve_zeros(processed_sequences):
    """
    StandardScaler hanya dilatih pada frame yang memiliki tangan.
    Frame kosong tetap nol agar dapat dikenali oleh Masking.
    """
    x = np.asarray(processed_sequences, dtype=np.float32)
    valid_frames = np.any(np.abs(x) > EPS, axis=2)

    scaler = StandardScaler()
    scaler.fit(x[valid_frames])
    return scaler


def apply_scaler_preserve_zeros(processed_sequences, scaler):
    x = np.asarray(processed_sequences, dtype=np.float32).copy()
    valid_frames = np.any(np.abs(x) > EPS, axis=2)

    if np.any(valid_frames):
        x[valid_frames] = scaler.transform(x[valid_frames]).astype(np.float32)

    return x


def build_model(num_classes, architecture="bigru", sequence_length=SEQUENCE_LENGTH,
                feature_count=PROCESSED_FEATURES):
    architecture = architecture.lower()

    model = Sequential(name=f"landmark_{architecture}")
    model.add(Input(shape=(sequence_length, feature_count)))
    model.add(Masking(mask_value=0.0))

    if architecture == "bigru":
        model.add(Bidirectional(GRU(64, return_sequences=True)))
        model.add(LayerNormalization())
        model.add(Dropout(0.30))
        model.add(Bidirectional(GRU(32)))

    elif architecture == "gru":
        model.add(GRU(96, return_sequences=True))
        model.add(LayerNormalization())
        model.add(Dropout(0.30))
        model.add(GRU(48))

    elif architecture == "lstm":
        model.add(LSTM(96, return_sequences=True))
        model.add(LayerNormalization())
        model.add(Dropout(0.30))
        model.add(LSTM(48))

    else:
        raise ValueError("architecture harus dipilih dari: bigru, gru, atau lstm")

    model.add(Dense(64, activation="relu", kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.35))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
