import argparse
import json
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import tensorflow as tf

from landmark_pipeline import (
    SEQUENCE_LENGTH,
    RAW_FEATURES,
    preprocess_sequence,
    apply_scaler_preserve_zeros
)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def extract_hand_keypoints(results):
    left_hand = np.zeros(21 * 3, dtype=np.float32)
    right_hand = np.zeros(21 * 3, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            label = handedness.classification[0].label

            keypoints = []
            for landmark in hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])

            keypoints = np.asarray(keypoints, dtype=np.float32)

            if label == "Left":
                left_hand = keypoints
            elif label == "Right":
                right_hand = keypoints

    return np.concatenate([left_hand, right_hand]).astype(np.float32)


def draw_panel(frame, label, confidence, status):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 118), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"Prediksi: {label}",
        (18, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.2%}",
        (18, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        status,
        (18, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1
    )


def stable_result(history, threshold, stability_window):
    if len(history) < stability_window:
        return "Menunggu gesture...", 0.0

    recent = list(history)[-stability_window:]
    labels = [item[0] for item in recent]
    most_common_label, count = Counter(labels).most_common(1)[0]

    matching_confidences = [
        confidence for label, confidence in recent
        if label == most_common_label
    ]
    avg_confidence = float(np.mean(matching_confidences))

    if count == stability_window and avg_confidence >= threshold:
        return most_common_label, avg_confidence

    return "Belum yakin", avg_confidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="model_landmark_hybrid")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.75)
    parser.add_argument("--stability_window", type=int, default=5)
    parser.add_argument(
        "--predict_every",
        type=int,
        default=3,
        help="Prediksi dilakukan setiap beberapa frame agar lebih ringan"
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    model = tf.keras.models.load_model(model_dir / "best_model.keras")
    encoder = joblib.load(model_dir / "label_encoder.pkl")
    scaler = joblib.load(model_dir / "scaler.pkl")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Kamera tidak dapat dibuka.")

    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    prediction_history = deque(maxlen=args.stability_window)
    frame_counter = 0

    display_label = "Menunggu gesture..."
    display_confidence = 0.0

    print("Tekan Q untuk keluar.")
    print("Tekan C untuk membersihkan buffer prediksi.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

            keypoints = extract_hand_keypoints(results)
            frame_buffer.append(keypoints)
            frame_counter += 1

            if (
                len(frame_buffer) == SEQUENCE_LENGTH
                and frame_counter % args.predict_every == 0
            ):
                raw_sequence = np.asarray(frame_buffer, dtype=np.float32)
                processed = preprocess_sequence(raw_sequence)
                processed = np.expand_dims(processed, axis=0)
                processed = apply_scaler_preserve_zeros(processed, scaler)

                probabilities = model.predict(processed, verbose=0)[0]
                best_index = int(np.argmax(probabilities))
                candidate_label = str(encoder.inverse_transform([best_index])[0])
                candidate_confidence = float(probabilities[best_index])

                prediction_history.append(
                    (candidate_label, candidate_confidence)
                )

                display_label, display_confidence = stable_result(
                    prediction_history,
                    threshold=args.confidence,
                    stability_window=args.stability_window
                )

            status = (
                f"Buffer: {len(frame_buffer)}/{SEQUENCE_LENGTH} | "
                "Q: keluar | C: reset"
            )
            draw_panel(frame, display_label, display_confidence, status)

            cv2.imshow("Real-Time Landmark Gesture Classification", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key == ord("c"):
                frame_buffer.clear()
                prediction_history.clear()
                display_label = "Menunggu gesture..."
                display_confidence = 0.0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
