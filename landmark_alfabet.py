import cv2
import mediapipe as mp
import csv
import os
import time
from collections import defaultdict

# ==========================================================
# KONFIGURASI DATASET
# ==========================================================
DATASET_FILE = "dataset_alfabet_landmark.csv"

# Jumlah data maksimal untuk setiap huruf
TARGET_PER_CLASS = 300

# Jeda persiapan sebelum pengambilan data dimulai
PREPARATION_TIME = 15

# Jeda pengambilan setiap sampel.
# 0.10 detik = maksimal sekitar 10 sampel per detik.
CAPTURE_INTERVAL = 0.10

# Daftar kelas alfabet A-Z
ALPHABET_CLASSES = [
    chr(letter)
    for letter in range(ord("A"), ord("Z") + 1)
]

# ==========================================================
# MEDIAPIPE HANDS
# ==========================================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ==========================================================
# MEMBUAT HEADER CSV
# ==========================================================
header = ["label"]

for hand_name in ["left_hand", "right_hand"]:
    for landmark_index in range(21):
        header += [
            f"{hand_name}_x{landmark_index}",
            f"{hand_name}_y{landmark_index}",
            f"{hand_name}_z{landmark_index}"
        ]

if not os.path.exists(DATASET_FILE):
    with open(DATASET_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)

# ==========================================================
# MEMBACA JUMLAH DATA YANG SUDAH TERSIMPAN
# ==========================================================
def load_class_counts():
    counts = defaultdict(int)

    if not os.path.exists(DATASET_FILE):
        return counts

    with open(DATASET_FILE, mode="r", newline="") as file:
        reader = csv.reader(file)

        # Lewati header CSV
        next(reader, None)

        for row in reader:
            if row:
                label = row[0].upper()

                if label in ALPHABET_CLASSES:
                    counts[label] += 1

    return counts


class_counts = load_class_counts()

# ==========================================================
# FUNGSI LANDMARK
# ==========================================================
def create_empty_hand_landmarks():
    """
    Satu tangan memiliki:
    21 landmark × 3 koordinat = 63 nilai.
    """
    return [0.0] * 63


def extract_landmarks(results):
    """
    Mengambil landmark tangan kiri dan kanan secara terpisah.

    Total fitur:
    - Tangan kiri  : 63 nilai
    - Tangan kanan : 63 nilai
    - Total        : 126 nilai
    """
    left_hand_data = create_empty_hand_landmarks()
    right_hand_data = create_empty_hand_landmarks()

    if not results.multi_hand_landmarks or not results.multi_handedness:
        return left_hand_data + right_hand_data

    for hand_landmarks, handedness in zip(
        results.multi_hand_landmarks,
        results.multi_handedness
    ):
        hand_label = handedness.classification[0].label

        landmarks = []

        for landmark in hand_landmarks.landmark:
            landmarks.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        if hand_label == "Left":
            left_hand_data = landmarks
        elif hand_label == "Right":
            right_hand_data = landmarks

    return left_hand_data + right_hand_data


def save_landmark_data(label, landmarks):
    row = [label] + landmarks

    with open(DATASET_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(row)

# ==========================================================
# FUNGSI TAMPILAN
# ==========================================================
def draw_text(
    frame,
    text,
    position,
    font_scale=0.65,
    color=(255, 255, 255),
    thickness=2
):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness
    )

# ==========================================================
# KAMERA
# ==========================================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera tidak dapat dibuka.")
    hands.close()
    raise SystemExit

# ==========================================================
# VARIABEL STATUS PROGRAM
# ==========================================================
selected_label = None

# True ketika data sedang disimpan otomatis
is_collecting = False

# True selama countdown 15 detik berlangsung
is_preparing = False

# Waktu mulai countdown
preparation_start_time = 0

# Waktu terakhir penyimpanan sampel
last_capture_time = 0

print("=" * 70)
print("PROGRAM PENGUMPULAN DATASET LANDMARK ALFABET")
print("=" * 70)
print("Tekan huruf A-Z : Memilih kelas alfabet")
print("Tekan SPACE     : Memulai countdown 15 detik")
print("Tekan SPACE     : Membatalkan countdown atau menjeda pengambilan data")
print("Tekan ESC       : Keluar dari program")
print("=" * 70)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame kamera gagal dibaca.")
        break

    # Membalik frame agar tampak seperti cermin
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Menggambar landmark tangan
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # ======================================================
    # COUNTDOWN PERSIAPAN 15 DETIK
    # ======================================================
    if selected_label and is_preparing:
        elapsed_time = time.time() - preparation_start_time
        remaining_time = PREPARATION_TIME - elapsed_time

        if remaining_time <= 0:
            is_preparing = False
            is_collecting = True
            last_capture_time = 0

            print(
                f"Countdown selesai. "
                f"Mulai mengambil data kelas {selected_label}."
            )

    # ======================================================
    # PENGAMBILAN DATA OTOMATIS
    # ======================================================
    if selected_label and is_collecting:
        current_count = class_counts[selected_label]

        if current_count >= TARGET_PER_CLASS:
            is_collecting = False

            print(
                f"Kelas {selected_label} selesai. "
                f"Jumlah data: {current_count}/{TARGET_PER_CLASS}"
            )

        elif results.multi_hand_landmarks:
            current_time = time.time()

            if current_time - last_capture_time >= CAPTURE_INTERVAL:
                landmarks = extract_landmarks(results)

                save_landmark_data(selected_label, landmarks)

                class_counts[selected_label] += 1
                last_capture_time = current_time

                print(
                    f"Data {selected_label} tersimpan: "
                    f"{class_counts[selected_label]}/{TARGET_PER_CLASS}"
                )

    # ======================================================
    # INFORMASI PADA LAYAR KAMERA
    # ======================================================
    draw_text(
        frame,
        "A-Z: pilih huruf | SPACE: mulai/jeda | ESC: keluar",
        (10, 30),
        font_scale=0.60,
        color=(0, 255, 0)
    )

    if selected_label:
        current_count = class_counts[selected_label]

        draw_text(
            frame,
            f"Kelas dipilih : {selected_label}",
            (10, 65),
            color=(255, 255, 0)
        )

        draw_text(
            frame,
            f"Jumlah data   : {current_count}/{TARGET_PER_CLASS}",
            (10, 95),
            color=(255, 255, 0)
        )

        if is_preparing:
            elapsed_time = time.time() - preparation_start_time
            remaining_time = max(
                0,
                int(PREPARATION_TIME - elapsed_time) + 1
            )

            draw_text(
                frame,
                f"Persiapan     : {remaining_time} detik",
                (10, 125),
                color=(0, 165, 255)
            )

            draw_text(
                frame,
                "Silakan perbaiki dan tahan posisi tangan",
                (10, 160),
                font_scale=0.70,
                color=(0, 165, 255)
            )

        elif is_collecting:
            draw_text(
                frame,
                "Status        : MENGAMBIL DATA",
                (10, 125),
                color=(0, 255, 0)
            )

            draw_text(
                frame,
                "Pertahankan posisi tangan sampai selesai",
                (10, 160),
                font_scale=0.65,
                color=(0, 255, 0)
            )

        else:
            draw_text(
                frame,
                "Status        : DIJEDA",
                (10, 125),
                color=(0, 165, 255)
            )

    else:
        draw_text(
            frame,
            "Pilih salah satu huruf A-Z terlebih dahulu",
            (10, 70),
            color=(0, 165, 255)
        )

    cv2.imshow("Dataset Alfabet Landmark Tangan", frame)

    key = cv2.waitKey(1) & 0xFF

    # ======================================================
    # KONTROL KEYBOARD
    # ======================================================

    # ESC untuk keluar
    if key == 27:
        break

    # SPACE untuk memulai atau menghentikan proses
    if key == ord(" "):
        if selected_label is None:
            print("Pilih huruf A-Z terlebih dahulu.")

        elif class_counts[selected_label] >= TARGET_PER_CLASS:
            print(
                f"Kelas {selected_label} sudah memiliki "
                f"{TARGET_PER_CLASS} data."
            )

        elif is_preparing:
            # Membatalkan countdown
            is_preparing = False

            print(
                f"Countdown kelas {selected_label} dibatalkan."
            )

        elif is_collecting:
            # Menjeda proses penyimpanan data
            is_collecting = False

            print(
                f"Pengambilan data kelas {selected_label} dijeda."
            )

        else:
            # Memulai countdown sebelum pengambilan data
            is_preparing = True
            preparation_start_time = time.time()

            print(
                f"Persiapan kelas {selected_label} dimulai. "
                f"Pengambilan data dimulai dalam "
                f"{PREPARATION_TIME} detik."
            )

    # Memilih kelas alfabet dengan tombol A-Z
    if ord("a") <= key <= ord("z"):
        selected_label = chr(key).upper()

        # Menghentikan proses kelas sebelumnya
        is_collecting = False
        is_preparing = False

        print(
            f"Kelas dipilih: {selected_label} | "
            f"Data tersedia: "
            f"{class_counts[selected_label]}/{TARGET_PER_CLASS}"
        )

# ==========================================================
# MENUTUP PROGRAM
# ==========================================================
cap.release()
hands.close()
cv2.destroyAllWindows()

print("=" * 70)
print("Program selesai.")
print(f"Dataset tersimpan pada file: {DATASET_FILE}")
print("=" * 70)