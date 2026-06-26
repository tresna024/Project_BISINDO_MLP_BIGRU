import threading

import cv2
import mediapipe as mp
import numpy as np


mp_hands = mp.solutions.hands

_hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,

    # Disamakan dengan program testing alfabet dan angka.
    min_detection_confidence=0.70,
    min_tracking_confidence=0.70,
)

_lock = threading.Lock()


def extract_landmarks(frame):
    """
    Menghasilkan 126 fitur dengan urutan tetap:
    63 fitur tangan kiri lalu 63 fitur tangan kanan.

    Catatan penting:
    - Frame sudah di-flip satu kali pada app.py.
    - Jangan melakukan cv2.flip() lagi di file ini.
    - Jika tangan tidak terdeteksi, fungsi mengembalikan None.
    """
    if frame is None:
        return None

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    with _lock:
        results = _hands.process(rgb)

    if (
        not results.multi_hand_landmarks
        or not results.multi_handedness
    ):
        return None

    left_hand = np.zeros(
        21 * 3,
        dtype=np.float32
    )

    right_hand = np.zeros(
        21 * 3,
        dtype=np.float32
    )

    for hand_landmarks, handedness in zip(
        results.multi_hand_landmarks,
        results.multi_handedness,
    ):
        label = (
            handedness
            .classification[0]
            .label
        )

        keypoints = []

        for landmark in hand_landmarks.landmark:
            keypoints.extend([
                landmark.x,
                landmark.y,
                landmark.z,
            ])

        keypoints = np.asarray(
            keypoints,
            dtype=np.float32
        )

        if label == "Left":
            left_hand = keypoints

        elif label == "Right":
            right_hand = keypoints

    landmarks = np.concatenate([
        left_hand,
        right_hand,
    ]).astype(np.float32)

    if np.count_nonzero(landmarks) == 0:
        return None

    return landmarks
