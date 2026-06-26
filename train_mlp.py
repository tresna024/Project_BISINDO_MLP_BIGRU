import os

# Mengurangi pesan log TensorFlow yang tidak terlalu diperlukan
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Lokasi dataset
DATASETS = {
    "alfabet": "data/dataset_alfabet_landmark.csv",
    "angka": "data/dataset_angka_landmark.csv"
}

# Nama kolom label pada CSV
LABEL_COLUMN = "label"

# Folder penyimpanan model
MODEL_FOLDER = Path("model")

# Folder penyimpanan grafik dan laporan evaluasi
REPORT_FOLDER = Path("hasil_training")

# Konfigurasi pembagian dataset
# 80% training, 10% validation, dan 10% testing
TEST_SIZE = 0.10
VALIDATION_SIZE = 0.10

# Konfigurasi proses training
EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Random seed digunakan agar pembagian data lebih konsisten
RANDOM_STATE = 42


# ============================================================
# RANDOM SEED
# ============================================================
def set_random_seed(seed=42):
    """
    Mengatur random seed agar hasil pembagian dataset dan proses
    training lebih konsisten ketika program dijalankan ulang.
    """

    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# MEMBACA DAN MEMBERSIHKAN DATASET
# ============================================================
def load_dataset(csv_path):
    """
    Membaca dataset landmark tangan dari file CSV.

    Format umum CSV:
    label,left_hand_x0,left_hand_y0,left_hand_z0,...,
    right_hand_x20,right_hand_y20,right_hand_z20

    Semua kolom selain 'label' otomatis dianggap sebagai fitur.
    """

    dataset_path = Path(csv_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"\nDataset tidak ditemukan: {dataset_path}\n"
            "Periksa kembali lokasi file pada konfigurasi DATASETS."
        )

    print("\n" + "=" * 75)
    print(f"MEMBACA DATASET: {dataset_path}")
    print("=" * 75)

    df = pd.read_csv(dataset_path)

    # Membersihkan nama header dari spasi yang tidak disengaja
    df.columns = [
        column.strip()
        for column in df.columns
    ]

    # Memastikan dataset memiliki kolom label
    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Kolom '{LABEL_COLUMN}' tidak ditemukan pada dataset."
        )

    # Menghapus data yang labelnya kosong
    df = df.dropna(
        subset=[LABEL_COLUMN]
    ).copy()

    # Mengubah label menjadi teks
    # Hal ini penting agar label angka tetap dapat diproses
    df[LABEL_COLUMN] = (
        df[LABEL_COLUMN]
        .astype(str)
        .str.strip()
    )

    # Menghapus label yang hanya berisi spasi
    df = df[
        df[LABEL_COLUMN] != ""
    ]

    # Semua kolom selain label digunakan sebagai fitur landmark
    feature_columns = [
        column
        for column in df.columns
        if column != LABEL_COLUMN
    ]

    if len(feature_columns) == 0:
        raise ValueError(
            "Kolom fitur landmark tidak ditemukan."
        )

    # Mengubah seluruh fitur menjadi tipe numerik
    for column in feature_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Nilai infinity dianggap sebagai nilai kosong
    df[feature_columns] = df[feature_columns].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Menghapus baris apabila semua nilai landmark kosong
    jumlah_data_awal = len(df)

    df = df.dropna(
        how="all",
        subset=feature_columns
    )

    jumlah_baris_kosong = jumlah_data_awal - len(df)

    # Menghitung nilai fitur yang kosong sebagian
    jumlah_nilai_kosong = int(
        df[feature_columns]
        .isna()
        .sum()
        .sum()
    )

    # Nilai kosong sebagian diisi dengan 0
    # Berguna apabila salah satu tangan tidak terdeteksi
    df[feature_columns] = (
        df[feature_columns]
        .fillna(0.0)
    )

    # Menghapus baris duplikat yang benar-benar identik
    jumlah_sebelum_hapus_duplikat = len(df)

    df = (
        df.drop_duplicates(
            subset=[LABEL_COLUMN] + feature_columns
        )
        .reset_index(drop=True)
    )

    jumlah_duplikat = (
        jumlah_sebelum_hapus_duplikat - len(df)
    )

    # Memisahkan fitur dan label
    X = df[feature_columns].astype(np.float32)
    y = df[LABEL_COLUMN]

    distribusi_kelas = (
        y.value_counts()
        .sort_index()
    )

    print(f"Jumlah data bersih       : {len(df)}")
    print(f"Jumlah fitur landmark    : {len(feature_columns)}")
    print(f"Jumlah kelas             : {y.nunique()}")
    print(f"Baris kosong dihapus     : {jumlah_baris_kosong}")
    print(f"Nilai kosong diisi nol   : {jumlah_nilai_kosong}")
    print(f"Data duplikat dihapus    : {jumlah_duplikat}")

    print("\nDistribusi data setiap kelas:")
    print(distribusi_kelas.to_string())

    # Validasi minimal jumlah kelas
    if y.nunique() < 2:
        raise ValueError(
            "Dataset harus memiliki minimal dua kelas."
        )

    # Validasi jumlah sampel minimal per kelas
    if distribusi_kelas.min() < 10:
        raise ValueError(
            "\nSetiap kelas harus memiliki minimal 10 sampel.\n"
            "Tambahkan dataset terlebih dahulu agar proses pembagian "
            "training, validation, dan testing dapat dilakukan."
        )

    return X, y, feature_columns, distribusi_kelas


# ============================================================
# MEMBUAT ARSITEKTUR MODEL MLP
# ============================================================

def build_mlp_model(jumlah_fitur, jumlah_kelas):
    """
    Membuat model Multi-Layer Perceptron untuk melakukan
    klasifikasi landmark tangan.
    """

    model = tf.keras.Sequential([
        # Jumlah input menyesuaikan jumlah kolom landmark
        tf.keras.layers.Input(
            shape=(jumlah_fitur,)
        ),

        # Hidden layer pertama
        tf.keras.layers.Dense(
            256,
            activation="relu"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.30),

        # Hidden layer kedua
        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),

        # Hidden layer ketiga
        tf.keras.layers.Dense(
            64,
            activation="relu"
        ),
        tf.keras.layers.Dropout(0.20),

        # Output layer
        tf.keras.layers.Dense(
            jumlah_kelas,
            activation="softmax"
        )
    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# MENYIMPAN GRAFIK ACCURACY DAN LOSS
# ============================================================

def save_training_graph(history, report_path, dataset_name):
    """
    Menyimpan grafik accuracy dan loss selama proses training.
    """

    # Grafik accuracy
    plt.figure(figsize=(10, 6))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title(
        f"Grafik Accuracy Model MLP - {dataset_name.capitalize()}"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        report_path / f"grafik_accuracy_{dataset_name}.png",
        dpi=200
    )

    plt.close()

    # Grafik loss
    plt.figure(figsize=(10, 6))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        f"Grafik Loss Model MLP - {dataset_name.capitalize()}"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        report_path / f"grafik_loss_{dataset_name}.png",
        dpi=200
    )

    plt.close()


# ============================================================
# MENYIMPAN CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    y_test,
    y_pred,
    class_names,
    report_path,
    dataset_name
):
    """
    Menyimpan confusion matrix untuk melihat kelas yang masih
    sering tertukar pada proses prediksi.
    """

    labels = np.arange(
        len(class_names)
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    ukuran_gambar = max(
        8,
        len(class_names) * 0.45
    )

    fig, ax = plt.subplots(
        figsize=(ukuran_gambar, ukuran_gambar)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(
        ax=ax,
        xticks_rotation=90,
        include_values=len(class_names) <= 15,
        cmap="Blues",
        colorbar=False
    )

    plt.title(
        f"Confusion Matrix Model MLP - {dataset_name.capitalize()}"
    )

    plt.tight_layout()

    plt.savefig(
        report_path / f"confusion_matrix_{dataset_name}.png",
        dpi=220
    )

    plt.close()


# ============================================================
# MENYIMPAN FILE MODEL DAN PREPROCESSING
# ============================================================

def save_model_files(
    model,
    scaler,
    label_encoder,
    dataset_name
):
    """
    Menyimpan:
    1. Model lengkap dengan ekstensi .h5
    2. Bobot model dengan ekstensi .weights.h5
    3. StandardScaler dengan ekstensi .save
    4. LabelEncoder dengan ekstensi .save
    """

    model_path = (
        MODEL_FOLDER /
        f"model_{dataset_name}_mlp.h5"
    )

    weights_path = (
        MODEL_FOLDER /
        f"model_{dataset_name}_mlp.weights.h5"
    )

    scaler_path = (
        MODEL_FOLDER /
        f"scaler_{dataset_name}.save"
    )

    label_encoder_path = (
        MODEL_FOLDER /
        f"label_encoder_{dataset_name}.save"
    )

    # Menyimpan arsitektur dan bobot model
    model.save(
        model_path
    )

    # Menyimpan bobot model saja
    model.save_weights(
        weights_path
    )

    # Menyimpan objek normalisasi
    joblib.dump(
        scaler,
        scaler_path
    )

    # Menyimpan objek encoder label
    joblib.dump(
        label_encoder,
        label_encoder_path
    )

    return {
        "model": model_path,
        "weights": weights_path,
        "scaler": scaler_path,
        "label_encoder": label_encoder_path
    }


# ============================================================
# PROSES TRAINING SATU DATASET
# ============================================================

def train_dataset(dataset_name):
    """
    Melatih hanya satu dataset berdasarkan pilihan dari terminal.

    Contoh:
    python train_mlp_landmark.py --dataset alfabet
    """

    csv_path = DATASETS[dataset_name]

    # Membuat folder jika belum tersedia
    MODEL_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path = (
        REPORT_FOLDER /
        dataset_name
    )

    report_path.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n" + "#" * 75)
    print(f"TRAINING MODEL MLP: {dataset_name.upper()}")
    print("#" * 75)

    # --------------------------------------------------------
    # Membaca dataset
    # --------------------------------------------------------

    X, y, feature_columns, distribusi_kelas = load_dataset(
        csv_path
    )

    # --------------------------------------------------------
    # Mengubah label menjadi angka
    # Contoh alfabet:
    # A -> 0, B -> 1, C -> 2, dan seterusnya
    # --------------------------------------------------------

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(
        y
    )

    print("\nDaftar kelas:")
    print(label_encoder.classes_)

    # --------------------------------------------------------
    # Membagi dataset menjadi:
    # 80% training
    # 10% validation
    # 10% testing
    # --------------------------------------------------------

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y_encoded,
        test_size=TEST_SIZE + VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
        shuffle=True
    )

    persentase_test_dari_temp = (
        TEST_SIZE /
        (TEST_SIZE + VALIDATION_SIZE)
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=persentase_test_dari_temp,
        random_state=RANDOM_STATE,
        stratify=y_temp,
        shuffle=True
    )

    print("\nPembagian dataset:")
    print(f"Data training   : {len(X_train)}")
    print(f"Data validation : {len(X_val)}")
    print(f"Data testing    : {len(X_test)}")

    # --------------------------------------------------------
    # Normalisasi fitur landmark
    # --------------------------------------------------------

    scaler = StandardScaler()

    # Scaler hanya mempelajari pola dari data training
    X_train_scaled = scaler.fit_transform(
        X_train
    )

    # Data validation dan testing hanya ditransformasi
    X_val_scaled = scaler.transform(
        X_val
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # --------------------------------------------------------
    # Menghitung bobot setiap kelas
    # Berguna jika jumlah data antar kelas tidak sepenuhnya sama
    # --------------------------------------------------------

    daftar_kelas_training = np.unique(
        y_train
    )

    hasil_bobot = compute_class_weight(
        class_weight="balanced",
        classes=daftar_kelas_training,
        y=y_train
    )

    class_weights = {
        int(class_id): float(weight)
        for class_id, weight in zip(
            daftar_kelas_training,
            hasil_bobot
        )
    }

    # --------------------------------------------------------
    # Membuat model
    # --------------------------------------------------------

    model = build_mlp_model(
        jumlah_fitur=X_train_scaled.shape[1],
        jumlah_kelas=len(label_encoder.classes_)
    )

    print("\nArsitektur Model:")
    model.summary()

    # Model terbaik sementara disimpan menggunakan format .keras
    # Setelah proses selesai, model final tetap disimpan sebagai .h5
    best_model_path = (
        report_path /
        f"best_model_{dataset_name}_mlp.keras"
    )

    # --------------------------------------------------------
    # Callback
    # --------------------------------------------------------

    callbacks = [
        # Menghentikan proses apabila validation loss tidak
        # mengalami peningkatan selama 20 epoch
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),

        # Menyimpan model dengan validation loss terbaik
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),

        # Mengurangi learning rate apabila peningkatan model
        # mulai melambat
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=0.00001,
            verbose=1
        )
    ]

    # --------------------------------------------------------
    # Menjalankan training
    # --------------------------------------------------------

    print("\nProses training dimulai...\n")

    history = model.fit(
        X_train_scaled,
        y_train,
        validation_data=(
            X_val_scaled,
            y_val
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # Memuat kembali model terbaik
    model = tf.keras.models.load_model(
        best_model_path
    )

    # --------------------------------------------------------
    # Evaluasi model menggunakan data testing
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("HASIL EVALUASI MODEL")
    print("=" * 75)

    test_loss, test_accuracy = model.evaluate(
        X_test_scaled,
        y_test,
        verbose=0
    )

    probabilities = model.predict(
        X_test_scaled,
        verbose=0
    )

    y_pred = np.argmax(
        probabilities,
        axis=1
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(
            len(label_encoder.classes_)
        ),
        target_names=label_encoder.classes_,
        digits=4,
        zero_division=0
    )

    print(f"Test Loss      : {test_loss:.4f}")
    print(f"Test Accuracy  : {test_accuracy:.4f}")
    print(f"Accuracy Score : {accuracy:.4f}")

    print("\nClassification Report:")
    print(report)

    # --------------------------------------------------------
    # Menyimpan model dan file preprocessing
    # --------------------------------------------------------

    saved_files = save_model_files(
        model=model,
        scaler=scaler,
        label_encoder=label_encoder,
        dataset_name=dataset_name
    )

    # --------------------------------------------------------
    # Menyimpan classification report
    # --------------------------------------------------------

    classification_report_path = (
        report_path /
        f"classification_report_{dataset_name}.txt"
    )

    with open(
        classification_report_path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(f"Dataset        : {dataset_name}\n")
        file.write(f"Jumlah kelas   : {len(label_encoder.classes_)}\n")
        file.write(f"Jumlah fitur   : {len(feature_columns)}\n")
        file.write(f"Test Loss      : {test_loss:.6f}\n")
        file.write(f"Test Accuracy  : {test_accuracy:.6f}\n")
        file.write(f"Accuracy Score : {accuracy:.6f}\n\n")
        file.write("Classification Report:\n")
        file.write(report)

    # --------------------------------------------------------
    # Menyimpan distribusi kelas
    # --------------------------------------------------------

    distribusi_kelas_path = (
        report_path /
        f"distribusi_kelas_{dataset_name}.csv"
    )

    distribusi_kelas.to_csv(
        distribusi_kelas_path,
        header=["jumlah_data"]
    )

    # --------------------------------------------------------
    # Menyimpan metadata model
    # --------------------------------------------------------

    metadata_path = (
        report_path /
        f"metadata_{dataset_name}.json"
    )

    metadata = {
        "dataset_name": dataset_name,
        "source_csv": str(csv_path),
        "number_of_features": len(feature_columns),
        "feature_columns": feature_columns,
        "number_of_classes": len(label_encoder.classes_),
        "classes": label_encoder.classes_.tolist(),
        "total_samples": int(len(X)),
        "training_samples": int(len(X_train_scaled)),
        "validation_samples": int(len(X_val_scaled)),
        "testing_samples": int(len(X_test_scaled)),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "accuracy_score": float(accuracy)
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Menyimpan grafik
    # --------------------------------------------------------

    save_training_graph(
        history=history,
        report_path=report_path,
        dataset_name=dataset_name
    )

    save_confusion_matrix(
        y_test=y_test,
        y_pred=y_pred,
        class_names=label_encoder.classes_,
        report_path=report_path,
        dataset_name=dataset_name
    )

    # --------------------------------------------------------
    # Informasi hasil akhir
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("TRAINING SELESAI")
    print("=" * 75)

    print("\nFile utama untuk implementasi berhasil disimpan:")

    print(
        f"- Model lengkap : {saved_files['model']}"
    )

    print(
        f"- Bobot model   : {saved_files['weights']}"
    )

    print(
        f"- Scaler        : {saved_files['scaler']}"
    )

    print(
        f"- Label encoder : {saved_files['label_encoder']}"
    )

    print("\nFile laporan evaluasi berhasil disimpan:")

    print(
        f"- Classification report : {classification_report_path}"
    )

    print(
        f"- Metadata              : {metadata_path}"
    )

    print(
        f"- Distribusi kelas       : {distribusi_kelas_path}"
    )

    print(
        f"- Grafik accuracy        : "
        f"{report_path / f'grafik_accuracy_{dataset_name}.png'}"
    )

    print(
        f"- Grafik loss            : "
        f"{report_path / f'grafik_loss_{dataset_name}.png'}"
    )

    print(
        f"- Confusion matrix       : "
        f"{report_path / f'confusion_matrix_{dataset_name}.png'}"
    )


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Program training model MLP untuk klasifikasi "
            "landmark tangan alfabet atau angka."
        )
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "alfabet",
            "angka"
        ],
        help=(
            "Pilih satu dataset yang ingin dilatih: "
            "alfabet atau angka."
        )
    )

    args = parser.parse_args()

    set_random_seed(
        RANDOM_STATE
    )

    train_dataset(
        dataset_name=args.dataset
    )


if __name__ == "__main__":
    main()