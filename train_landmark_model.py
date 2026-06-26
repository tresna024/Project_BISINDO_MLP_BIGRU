import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

from landmark_pipeline import (
    SEQUENCE_LENGTH,
    PROCESSED_FEATURES,
    set_seed,
    load_raw_dataset,
    preprocess_batch,
    augment_raw_sequence,
    fit_scaler_preserve_zeros,
    apply_scaler_preserve_zeros,
    build_model
)


def create_augmented_training_set(raw_x_train, y_train, copies):
    raw_sequences = [sequence for sequence in raw_x_train]
    labels = [label for label in y_train]

    for _ in range(copies):
        for sequence, label in zip(raw_x_train, y_train):
            raw_sequences.append(augment_raw_sequence(sequence))
            labels.append(label)

    return (
        np.asarray(raw_sequences, dtype=np.float32),
        np.asarray(labels, dtype=np.int64)
    )


def save_history_plot(history, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="train_accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training dan Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training dan Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(save_path).replace("accuracy", "loss"), dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset_landmark_realtime")
    parser.add_argument("--output", default="model_landmark_hybrid")
    parser.add_argument(
        "--architecture",
        choices=["bigru", "gru", "lstm"],
        default="bigru"
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--augmentation_copies",
        type=int,
        default=2,
        help="Jumlah salinan augmentasi untuk setiap sample training"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TRAINING MODEL LANDMARK HYBRID STATIC-DYNAMIC")
    print("=" * 70)

    raw_x, string_labels, files = load_raw_dataset(args.dataset)

    encoder = LabelEncoder()
    y = encoder.fit_transform(string_labels)

    print("Jumlah sample :", len(raw_x))
    print("Jumlah kelas  :", len(encoder.classes_))
    print("Daftar kelas  :", list(encoder.classes_))

    # 70% train, 15% validation, 15% test
    raw_x_train, raw_x_temp, y_train, y_temp = train_test_split(
        raw_x,
        y,
        test_size=0.30,
        random_state=args.seed,
        stratify=y
    )

    raw_x_val, raw_x_test, y_val, y_test = train_test_split(
        raw_x_temp,
        y_temp,
        test_size=0.50,
        random_state=args.seed,
        stratify=y_temp
    )

    print("\nPembagian dataset:")
    print("Train      :", len(raw_x_train))
    print("Validation :", len(raw_x_val))
    print("Test       :", len(raw_x_test))

    raw_x_train_aug, y_train_aug = create_augmented_training_set(
        raw_x_train,
        y_train,
        copies=args.augmentation_copies
    )

    print("Train setelah augmentasi:", len(raw_x_train_aug))

    x_train = preprocess_batch(raw_x_train_aug)
    x_val = preprocess_batch(raw_x_val)
    x_test = preprocess_batch(raw_x_test)

    scaler = fit_scaler_preserve_zeros(x_train)
    x_train = apply_scaler_preserve_zeros(x_train, scaler)
    x_val = apply_scaler_preserve_zeros(x_val, scaler)
    x_test = apply_scaler_preserve_zeros(x_test, scaler)

    model = build_model(
        num_classes=len(encoder.classes_),
        architecture=args.architecture
    )

    print("\nArsitektur model:")
    model.summary()

    best_model_path = output_dir / "best_model.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=18,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=6,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history = model.fit(
        x_train,
        y_train_aug,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Load checkpoint terbaik berdasarkan val_accuracy
    model = tf.keras.models.load_model(best_model_path)

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    predictions = model.predict(x_test, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    print("\n" + "=" * 70)
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_accuracy:.4f}")
    print("=" * 70)

    report = classification_report(
        y_test,
        predicted_labels,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        digits=4,
        zero_division=0
    )
    print("\nClassification Report:")
    print(report)

    (output_dir / "classification_report.txt").write_text(
        report,
        encoding="utf-8"
    )

    matrix = confusion_matrix(
        y_test,
        predicted_labels,
        labels=np.arange(len(encoder.classes_))
    )

    fig, ax = plt.subplots(figsize=(13, 11))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=encoder.classes_
    )
    display.plot(ax=ax, xticks_rotation=45, cmap=None, colorbar=False)
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close()

    save_history_plot(history, output_dir / "history_accuracy.png")

    joblib.dump(encoder, output_dir / "label_encoder.pkl")
    joblib.dump(scaler, output_dir / "scaler.pkl")

    config = {
        "architecture": args.architecture,
        "sequence_length": SEQUENCE_LENGTH,
        "processed_features": PROCESSED_FEATURES,
        "classes": list(encoder.classes_),
        "confidence_threshold_recommended": 0.75,
        "stability_window_recommended": 5
    }

    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8"
    )

    print("\nFile model berhasil disimpan di:", output_dir)
    print("- best_model.keras")
    print("- label_encoder.pkl")
    print("- scaler.pkl")
    print("- config.json")
    print("- classification_report.txt")
    print("- confusion_matrix.png")
    print("- history_accuracy.png")
    print("- history_loss.png")


if __name__ == "__main__":
    main()
