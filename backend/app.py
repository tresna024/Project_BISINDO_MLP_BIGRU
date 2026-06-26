from collections import Counter, deque
from pathlib import Path
import base64
import json
import threading
import time

import cv2
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS

from utils.mediapipe_utils import extract_landmarks
from landmark_pipeline import (
    SEQUENCE_LENGTH,
    preprocess_sequence,
    apply_scaler_preserve_zeros,
)


# KONFIGURASI DASAR
BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
CORS(app)


# KONFIGURASI MODEL MLP
MLP_MODEL_PATHS = {
    "angka": {
        "model": BASE_DIR / "model" / "model_angka_mlp.h5",
        "scaler": BASE_DIR / "model" / "scaler_angka.save",
        "encoder": BASE_DIR / "model" / "label_encoder_angka.save",
    },
    "alfabet": {
        "model": BASE_DIR / "model" / "model_alfabet_mlp.h5",
        "scaler": BASE_DIR / "model" / "scaler_alfabet.save",
        "encoder": BASE_DIR / "model" / "label_encoder_alfabet.save",
    },
}

MLP_CONFIDENCE_THRESHOLD = 0.70


# KONFIGURASI MODEL BIGRU
BIGRU_DIR = BASE_DIR / "model_bigru"

BIGRU_MODEL_PATH = BIGRU_DIR / "best_model.keras"
BIGRU_SCALER_PATH = BIGRU_DIR / "scaler.pkl"
BIGRU_ENCODER_PATH = BIGRU_DIR / "label_encoder.pkl"
BIGRU_CONFIG_PATH = BIGRU_DIR / "config.json"

BIGRU_CONFIDENCE_THRESHOLD = 0.75
BIGRU_STABILITY_WINDOW = 3

SESSION_TIMEOUT_SECONDS = 120
MAX_MISSING_HAND_FRAMES = 5


# PENYIMPANAN MODEL DI MEMORI
mlp_models = {}
mlp_scalers = {}
mlp_encoders = {}
mlp_feature_columns = {}

bigru_model = None
bigru_scaler = None
bigru_encoder = None

sequence_sessions = {}
session_lock = threading.Lock()


# MEMBUAT NAMA KOLOM LANDMARK DEFAULT
def create_default_feature_columns():
    feature_columns = []

    for hand_name in ["left_hand", "right_hand"]:
        for index in range(21):
            feature_columns.extend([
                f"{hand_name}_x{index}",
                f"{hand_name}_y{index}",
                f"{hand_name}_z{index}",
            ])

    return feature_columns


DEFAULT_FEATURE_COLUMNS = create_default_feature_columns()


# MEMBUAT ARSITEKTUR MLP
def create_mlp_model(output_units):
    """
    Membuat arsitektur MLP untuk angka dan alfabet.

    Input:
        126 fitur landmark tangan.

    Output:
        Probabilitas setiap kelas.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(126,)),

        tf.keras.layers.Dense(
            256,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(
            64,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            output_units,
            activation="softmax"
        ),
    ])

    return model

# MEMERIKSA KEBERADAAN FILE
def require_file(path):
    """
    Memastikan file model tersedia sebelum backend dijalankan.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {path}"
        )

# LOAD MODEL MLP
def load_mlp_models():
    """
    Memuat model angka dan alfabet dengan cara yang sama seperti
    program testing: load_model(), scaler, label encoder, dan
    urutan kolom fitur dari scaler.feature_names_in_.
    """
    for model_type, paths in MLP_MODEL_PATHS.items():
        require_file(paths["model"])
        require_file(paths["scaler"])
        require_file(paths["encoder"])

        model = tf.keras.models.load_model(
            paths["model"]
        )

        scaler = joblib.load(
            paths["scaler"]
        )

        encoder = joblib.load(
            paths["encoder"]
        )

        if hasattr(scaler, "feature_names_in_"):
            feature_columns = list(
                scaler.feature_names_in_
            )
        else:
            feature_columns = (
                DEFAULT_FEATURE_COLUMNS.copy()
            )

        if hasattr(scaler, "n_features_in_"):
            number_of_features = int(
                scaler.n_features_in_
            )
        else:
            number_of_features = len(
                feature_columns
            )

        if number_of_features != 126:
            raise ValueError(
                f"Scaler '{model_type}' harus menerima 126 fitur, "
                f"tetapi ditemukan {number_of_features}."
            )

        if len(feature_columns) != number_of_features:
            raise ValueError(
                f"Jumlah kolom fitur '{model_type}' tidak sesuai scaler."
            )

        mlp_models[model_type] = model
        mlp_scalers[model_type] = scaler
        mlp_encoders[model_type] = encoder
        mlp_feature_columns[model_type] = feature_columns

        print(
            f"[OK] Model MLP '{model_type}' berhasil dimuat "
            f"dari {paths['model']}."
        )

        print(
            f"     Jumlah kelas: {len(encoder.classes_)}"
        )

        print(
            f"     Daftar kelas: {list(encoder.classes_)}"
        )

# LOAD MODEL BIGRU
def load_bigru_model():
    """
    Memuat model BiGRU untuk mendeteksi kata static dan dynamic.
    """
    global bigru_model
    global bigru_scaler
    global bigru_encoder

    global BIGRU_CONFIDENCE_THRESHOLD
    global BIGRU_STABILITY_WINDOW

    require_file(
        BIGRU_MODEL_PATH
    )

    require_file(
        BIGRU_SCALER_PATH
    )

    require_file(
        BIGRU_ENCODER_PATH
    )

    bigru_model = tf.keras.models.load_model(
        BIGRU_MODEL_PATH
    )

    bigru_scaler = joblib.load(
        BIGRU_SCALER_PATH
    )

    bigru_encoder = joblib.load(
        BIGRU_ENCODER_PATH
    )

    if BIGRU_CONFIG_PATH.exists():
        config = json.loads(
            BIGRU_CONFIG_PATH.read_text(
                encoding="utf-8"
            )
        )

        BIGRU_CONFIDENCE_THRESHOLD = float(
            config.get(
                "confidence_threshold_recommended",
                BIGRU_CONFIDENCE_THRESHOLD,
            )
        )

        BIGRU_STABILITY_WINDOW = int(
            config.get(
                "stability_window_recommended",
                BIGRU_STABILITY_WINDOW,
            )
        )

    print(
        "[OK] Model BiGRU 'kata' berhasil dimuat."
    )

    print(
        f"     Jumlah kelas: {len(bigru_encoder.classes_)}"
    )

    print(
        f"     Daftar kelas: {list(bigru_encoder.classes_)}"
    )

    print(
        f"     Confidence threshold: "
        f"{BIGRU_CONFIDENCE_THRESHOLD}"
    )

    print(
        f"     Stability window: "
        f"{BIGRU_STABILITY_WINDOW}"
    )

# LOAD SELURUH MODEL
def load_all_models():
    """
    Memuat seluruh model ketika Flask pertama kali dijalankan.
    """
    print("\n========================================")
    print("MEMUAT SELURUH MODEL")
    print("========================================")

    load_mlp_models()
    load_bigru_model()

    print("========================================")
    print("SELURUH MODEL BERHASIL DIMUAT")
    print("========================================\n")


# DECODE BASE64 MENJADI GAMBAR
def decode_base64_image(image_data):
    """
    Mengubah gambar Base64 dari frontend menjadi frame OpenCV.

    Frame dibalik secara horizontal agar mekanisme inferensi
    sama seperti proses pengumpulan dataset.
    """
    if not image_data:
        raise ValueError("Gambar tidak ditemukan.")

    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_data)
    except Exception as exc:
        raise ValueError("Format Base64 gambar tidak valid.") from exc

    np_arr = np.frombuffer(
        image_bytes,
        np.uint8
    )

    frame = cv2.imdecode(
        np_arr,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        raise ValueError(
            "Gambar tidak dapat dibaca oleh OpenCV."
        )

    frame = cv2.flip(frame, 1)

    return frame



# VALIDASI SHAPE LANDMARK
def normalize_landmark_shape(landmarks):
    """
    Memastikan landmark memiliki 126 fitur.

    126 fitur berasal dari:
    2 tangan × 21 landmark × 3 koordinat.
    """
    landmarks = np.asarray(
        landmarks,
        dtype=np.float32
    ).reshape(-1)

    if landmarks.shape != (126,):
        raise ValueError(
            "Jumlah landmark harus 126 fitur. "
            f"Ditemukan shape: {landmarks.shape}"
        )

    return landmarks


# MENYUSUN LANDMARK SESUAI URUTAN TRAINING MLP
def landmarks_to_training_dataframe(landmarks, model_type):
    """
    Membuat DataFrame agar nama dan urutan fitur sama persis
    seperti saat scaler model angka atau alfabet dilatih.
    """
    landmarks = normalize_landmark_shape(
        landmarks
    )

    value_by_column = dict(
        zip(
            DEFAULT_FEATURE_COLUMNS,
            landmarks
        )
    )

    ordered_columns = (
        mlp_feature_columns[model_type]
    )

    ordered_values = [
        value_by_column[column]
        for column in ordered_columns
    ]

    return pd.DataFrame(
        [ordered_values],
        columns=ordered_columns
    )


# MEMBUAT SESSION BUFFER BIGRU
def new_sequence_session():
    """
    Membuat buffer baru untuk menyimpan 30 frame terakhir.
    """
    return {
        "frames": deque(
            maxlen=SEQUENCE_LENGTH
        ),

        "prediction_history": deque(
            maxlen=BIGRU_STABILITY_WINDOW
        ),

        "missing_hand_frames": 0,

        "updated_at": time.time(),
    }


# MEMBERSIHKAN SESSION YANG TIDAK AKTIF
def cleanup_expired_sessions():
    """
    Menghapus session yang tidak digunakan agar RAM tidak penuh.
    """
    now = time.time()

    expired_ids = [
        session_id

        for session_id, state
        in sequence_sessions.items()

        if (
            now - state["updated_at"]
            > SESSION_TIMEOUT_SECONDS
        )
    ]

    for session_id in expired_ids:
        del sequence_sessions[
            session_id
        ]


# MENGAMBIL SESSION BERDASARKAN ID BROWSER
def get_sequence_session(session_id):
    """
    Mengambil buffer milik pengguna berdasarkan session_id.
    """
    with session_lock:
        cleanup_expired_sessions()

        if session_id not in sequence_sessions:
            sequence_sessions[
                session_id
            ] = new_sequence_session()

        state = sequence_sessions[
            session_id
        ]

        state["updated_at"] = time.time()

        return state


# MENGHAPUS SESSION
def clear_sequence_session(session_id):
    """
    Menghapus buffer ketika gesture diulang atau kamera dihentikan.
    """
    with session_lock:
        sequence_sessions.pop(
            session_id,
            None
        )


# MEMERIKSA STABILITAS HASIL PREDIKSI BIGRU
def get_stable_prediction(prediction_history):
    """
    Prediksi dianggap stabil jika beberapa hasil terakhir:
    - memiliki label yang sama
    - confidence rata-ratanya mencapai threshold
    """
    if (
        len(prediction_history)
        < BIGRU_STABILITY_WINDOW
    ):
        return None, 0.0

    recent = list(
        prediction_history
    )

    labels = [
        item["label"]
        for item in recent
    ]

    most_common_label, count = (
        Counter(labels)
        .most_common(1)[0]
    )

    if (
        count
        != BIGRU_STABILITY_WINDOW
    ):
        return None, 0.0

    confidences = [
        item["confidence"]

        for item in recent

        if item["label"]
        == most_common_label
    ]

    average_confidence = float(
        np.mean(confidences)
    )

    if (
        average_confidence
        < BIGRU_CONFIDENCE_THRESHOLD
    ):
        return (
            None,
            average_confidence
        )

    return (
        most_common_label,
        average_confidence
    )


# PREDIKSI MLP: ANGKA DAN ALFABET
def predict_mlp(frame, model_type):
    """
    Menjalankan prediksi MLP untuk angka atau alfabet dengan
    pipeline yang sama seperti program testing.
    """
    landmarks = extract_landmarks(
        frame
    )

    if landmarks is None:
        return {
            "label": "Tangan tidak terdeteksi",
            "confidence": 0.0,
            "model_type": model_type,
            "ready": False,
        }

    landmark_data = (
        landmarks_to_training_dataframe(
            landmarks=landmarks,
            model_type=model_type
        )
    )

    scaled_landmarks = (
        mlp_scalers[model_type]
        .transform(landmark_data)
    )

    probabilities = (
        mlp_models[model_type]
        .predict(
            scaled_landmarks,
            verbose=0
        )[0]
    )

    class_id = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[class_id]
    )

    label = str(
        mlp_encoders[model_type]
        .inverse_transform(
            [class_id]
        )[0]
    )

    if confidence < MLP_CONFIDENCE_THRESHOLD:
        return {
            "label": "Gesture kurang yakin",
            "candidate_label": label,
            "confidence": round(
                confidence * 100,
                2
            ),
            "model_type": model_type,
            "ready": False,
        }

    return {
        "label": label,
        "confidence": round(
            confidence * 100,
            2
        ),
        "model_type": model_type,
        "ready": True,
    }


# PREDIKSI BIGRU: KATA STATIC DAN DYNAMIC
def predict_bigru_word(frame, session_id):
    """
    Menjalankan prediksi BiGRU.

    Backend mengumpulkan 30 frame terlebih dahulu sebelum
    menjalankan prediksi gesture kata.
    """
    state = get_sequence_session(
        session_id
    )

    landmarks = extract_landmarks(
        frame
    )

    if landmarks is None:
        state["missing_hand_frames"] += 1

        if (
            state["missing_hand_frames"]
            >= MAX_MISSING_HAND_FRAMES
        ):
            clear_sequence_session(
                session_id
            )

        return {
            "label": "Tangan tidak terdeteksi",

            "confidence": 0.0,

            "model_type": "kata",

            "frames_collected": len(
                state["frames"]
            ),

            "frames_required": SEQUENCE_LENGTH,

            "ready": False,
        }

    state["missing_hand_frames"] = 0

    landmarks = normalize_landmark_shape(
        landmarks
    )

    state["frames"].append(
        landmarks
    )

    frames_collected = len(
        state["frames"]
    )

    if (
        frames_collected
        < SEQUENCE_LENGTH
    ):
        return {
            "label": "Mengumpulkan gerakan...",

            "confidence": 0.0,

            "model_type": "kata",

            "frames_collected": frames_collected,

            "frames_required": SEQUENCE_LENGTH,

            "ready": False,
        }

    raw_sequence = np.asarray(
        state["frames"],
        dtype=np.float32
    )

    processed_sequence = (
        preprocess_sequence(
            raw_sequence
        )
    )

    processed_sequence = np.expand_dims(
        processed_sequence,
        axis=0
    )

    processed_sequence = (
        apply_scaler_preserve_zeros(
            processed_sequence,
            bigru_scaler,
        )
    )

    probabilities = (
        bigru_model
        .predict(
            processed_sequence,
            verbose=0
        )[0]
    )

    class_id = int(
        np.argmax(probabilities)
    )

    candidate_label = str(
        bigru_encoder
        .inverse_transform(
            [class_id]
        )[0]
    )

    candidate_confidence = float(
        probabilities[class_id]
    )

    state["prediction_history"].append({
        "label": candidate_label,

        "confidence": candidate_confidence,
    })

    stable_label, stable_confidence = (
        get_stable_prediction(
            state["prediction_history"]
        )
    )

    if stable_label is None:
        return {
            "label": "Menganalisis gerakan...",

            "confidence": round(
                candidate_confidence
                * 100,
                2
            ),

            "candidate_label": candidate_label,

            "model_type": "kata",

            "frames_collected": frames_collected,

            "frames_required": SEQUENCE_LENGTH,

            "ready": False,
        }

    return {
        "label": stable_label,

        "confidence": round(
            stable_confidence
            * 100,
            2
        ),

        "model_type": "kata",

        "frames_collected": frames_collected,

        "frames_required": SEQUENCE_LENGTH,

        "ready": True,
    }


# ROUTE UTAMA
@app.route(
    "/",
    methods=["GET"]
)
def home():
    """
    Digunakan untuk memeriksa apakah API berhasil berjalan.
    """
    return jsonify({
        "message":
            "API Bahasa Isyarat Hybrid Berjalan",

        "available_models": [
            "kata",
            "angka",
            "alfabet",
        ],

        "architectures": {
            "kata":
                "BiGRU sequence model",

            "angka":
                "MLP single-frame model",

            "alfabet":
                "MLP single-frame model",
        },

        "bigru_sequence_length":
            SEQUENCE_LENGTH,
    })


# ROUTE PREDIKSI
@app.route(
    "/predict",
    methods=["POST"]
)
def predict():
    """
    Menerima gambar dari frontend React dan memilih model
    berdasarkan model_type.
    """
    try:
        data = (
            request.get_json(
                silent=True
            )
            or {}
        )

        image_data = data.get(
            "image"
        )

        model_type = str(
            data.get(
                "model_type",
                "kata"
            )
        ).lower()

        session_id = str(
            data.get(
                "session_id",
                "default"
            )
        )

        if model_type not in {
            "kata",
            "angka",
            "alfabet",
        }:
            return jsonify({
                "error":
                    "model_type tidak valid. "
                    "Gunakan kata, angka, atau alfabet."
            }), 400

        if not image_data:
            return jsonify({
                "error":
                    "Gambar tidak ditemukan."
            }), 400

        frame = decode_base64_image(
            image_data
        )

        if model_type == "kata":
            result = predict_bigru_word(
                frame,
                session_id
            )

        else:
            result = predict_mlp(
                frame,
                model_type
            )

        return jsonify(
            result
        )

    except Exception as exc:
        return jsonify({
            "error": str(exc)
        }), 500


# ROUTE RESET BUFFER BIGRU
@app.route(
    "/reset-sequence",
    methods=["POST"]
)
def reset_sequence():
    """
    Menghapus buffer 30 frame ketika pengguna:
    - menekan tombol ulang gesture
    - mengganti model
    - menghentikan kamera
    - menambahkan kata ke kalimat
    """
    data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    session_id = str(
        data.get(
            "session_id",
            "default"
        )
    )

    clear_sequence_session(
        session_id
    )

    return jsonify({
        "message":
            "Buffer sequence berhasil dibersihkan.",

        "session_id":
            session_id,
    })


# MENJALANKAN FLASK
if __name__ == "__main__":
    load_all_models()

    app.run(
        debug=True,
        host="127.0.0.1",
        port=8000,
        use_reloader=False,
    )