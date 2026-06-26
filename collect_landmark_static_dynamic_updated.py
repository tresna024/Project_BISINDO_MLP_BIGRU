import os
import cv2
import time
import argparse
import numpy as np
import mediapipe as mp


# ============================================================
# KONFIGURASI DATASET
# ============================================================
DATASET_PATH = "dataset_landmark_realtime"

SEQUENCE_LENGTH = 30
FEATURE_PER_FRAME = 126  # 2 tangan x 21 landmark x 3 koordinat (x, y, z)

# Kata tambahan yang diperlakukan sebagai pose STATIC.
# Folder dataset memakai underscore agar lebih aman digunakan saat training.
STATIC_CLASSES = [
    "benar",
    "kamu",
    "kapan",
    "makan",
    "minum",
    "motor",
    "sama_sama",
    "terima_kasih",
]

# Alias memudahkan penulisan melalui terminal.
CLASS_ALIASES = {
    "sama-sama": "sama_sama",
    "sama sama": "sama_sama",
    "sama_sama": "sama_sama",
    "terimakasih": "terima_kasih",
    "terima-kasih": "terima_kasih",
    "terima kasih": "terima_kasih",
    "terima_kasih": "terima_kasih",
}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def create_folder(path):
    """Membuat folder jika belum tersedia."""
    os.makedirs(path, exist_ok=True)


def normalize_class_name(class_name):
    """
    Menyamakan format nama kelas agar folder dataset konsisten.
    Contoh:
    - sama-sama    -> sama_sama
    - terima kasih -> terima_kasih
    """
    normalized = class_name.strip().lower()

    if normalized in CLASS_ALIASES:
        return CLASS_ALIASES[normalized]

    normalized = normalized.replace("-", "_").replace(" ", "_")

    while "__" in normalized:
        normalized = normalized.replace("__", "_")

    return normalized


def get_display_name(class_name):
    """Mengubah nama folder menjadi tulisan yang nyaman dibaca."""
    return class_name.replace("_", " ").title()


def extract_hand_keypoints(results):
    """
    Mengambil landmark tangan kiri dan kanan.
    Jika salah satu tangan tidak terlihat, posisinya diisi angka nol.

    Output:
        array dengan shape (126,)
    """
    left_hand = np.zeros(21 * 3, dtype=np.float32)
    right_hand = np.zeros(21 * 3, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            label = handedness.classification[0].label

            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

            keypoints = np.array(keypoints, dtype=np.float32)

            if label == "Left":
                left_hand = keypoints
            elif label == "Right":
                right_hand = keypoints

    return np.concatenate([left_hand, right_hand]).astype(np.float32)


def pad_or_sample_sequence(sequence, target_length=SEQUENCE_LENGTH):
    """
    Menyesuaikan sequence dinamis menjadi jumlah frame yang tetap.
    """
    sequence = np.asarray(sequence, dtype=np.float32)

    if len(sequence) == 0:
        return np.zeros((target_length, FEATURE_PER_FRAME), dtype=np.float32)

    if len(sequence) == target_length:
        return sequence

    if len(sequence) > target_length:
        indices = np.linspace(0, len(sequence) - 1, target_length).astype(int)
        return sequence[indices]

    while len(sequence) < target_length:
        sequence = np.vstack([sequence, sequence[-1]])

    return sequence.astype(np.float32)


def get_existing_sample_count(class_folder):
    """Menghitung jumlah file dataset .npy yang sudah tersimpan."""
    if not os.path.exists(class_folder):
        return 0

    return len([
        file_name
        for file_name in os.listdir(class_folder)
        if file_name.endswith(".npy")
    ])


def get_next_sample_index(class_folder):
    """Menentukan nomor file sample berikutnya."""
    existing_files = [
        file_name
        for file_name in os.listdir(class_folder)
        if file_name.endswith(".npy")
    ]

    numbers = []

    for file_name in existing_files:
        try:
            number = int(
                file_name.replace("sample_", "").replace(".npy", "")
            )
            numbers.append(number)
        except ValueError:
            pass

    return max(numbers, default=0) + 1


def draw_panel(frame, title, subtitle="", color=(0, 255, 0)):
    """Menampilkan informasi proses pada bagian atas kamera."""
    width = frame.shape[1]

    cv2.rectangle(frame, (0, 0), (width, 115), (0, 0, 0), -1)

    cv2.putText(
        frame,
        title,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        color,
        2
    )

    if subtitle:
        cv2.putText(
            frame,
            subtitle,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.57,
            (255, 255, 255),
            2
        )


def draw_landmarks(frame, results):
    """Menggambar landmark tangan pada tampilan kamera."""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )


def countdown(cap, hands, seconds):
    """Hitung mundur sebelum proses pengambilan data dimulai."""
    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            return False

        frame = cv2.flip(frame, 1)

        elapsed = time.time() - start_time
        remaining = max(1, int(seconds - elapsed) + 1)

        if elapsed >= seconds:
            return True

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        draw_landmarks(frame, results)

        draw_panel(
            frame,
            f"Mulai dalam {remaining} detik",
            "Siapkan posisi tangan kamu...",
            color=(0, 255, 255)
        )

        cv2.imshow("Collect Landmark Dataset", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False


def collect_static_auto(
    cap,
    hands,
    class_folder,
    class_name,
    total_samples,
    saved_count,
    delay_seconds=0.3
):
    """
    Mode STATIC:
    - Tekan S satu kali.
    - Program otomatis mengambil beberapa sample hingga target terpenuhi.
    - Satu sample diambil dari satu frame landmark tangan.
    - Landmark tersebut diulang menjadi 30 frame agar format dataset
      static dan dynamic sama-sama memiliki shape (30, 126).
    """
    display_name = get_display_name(class_name)

    print("\n=== Mode STATIC otomatis dimulai ===")
    print(f"Kelas: {display_name}")
    print("Ubah sedikit posisi, jarak, atau sudut tangan selama pengambilan data.")
    print("Tekan Q untuk menghentikan proses.\n")

    while saved_count < total_samples:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        draw_landmarks(frame, results)

        keypoints = extract_hand_keypoints(results)

        if np.count_nonzero(keypoints) == 0:
            draw_panel(
                frame,
                f"Kelas: {display_name} | Landmark belum terbaca",
                f"Tersimpan: {saved_count}/{total_samples} | Pastikan tangan terlihat jelas",
                color=(0, 0, 255)
            )

            cv2.imshow("Collect Landmark Dataset", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            continue

        sequence = np.repeat(
            keypoints[np.newaxis, :],
            repeats=SEQUENCE_LENGTH,
            axis=0
        ).astype(np.float32)

        sample_index = get_next_sample_index(class_folder)
        save_path = os.path.join(
            class_folder,
            f"sample_{sample_index:03d}.npy"
        )

        np.save(save_path, sequence)
        saved_count += 1

        print(
            f"[STATIC] {display_name}: "
            f"{saved_count}/{total_samples} tersimpan -> {save_path}"
        )

        draw_panel(
            frame,
            f"Kelas: {display_name} | Mengambil data STATIC otomatis",
            f"Tersimpan: {saved_count}/{total_samples} | Gerakkan posisi tangan sedikit",
            color=(0, 255, 0)
        )

        cv2.imshow("Collect Landmark Dataset", frame)

        key = cv2.waitKey(max(1, int(delay_seconds * 1000))) & 0xFF

        if key == ord("q"):
            break

    print("\n=== Mode STATIC otomatis selesai ===")
    return saved_count


def collect_dynamic_sequence(cap, hands, capture_seconds):
    """
    Mode DYNAMIC:
    Mengambil landmark tangan selama beberapa detik, kemudian menyesuaikannya
    menjadi sequence dengan shape (30, 126).
    """
    sequence = []
    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        elapsed = time.time() - start_time
        remaining = capture_seconds - elapsed

        if elapsed >= capture_seconds:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        draw_landmarks(frame, results)

        keypoints = extract_hand_keypoints(results)
        sequence.append(keypoints)

        draw_panel(
            frame,
            "Sedang mengambil landmark gerakan...",
            f"Sisa waktu: {remaining:.1f} detik | Frame terkumpul: {len(sequence)}",
            color=(0, 255, 255)
        )

        cv2.imshow("Collect Landmark Dataset", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    return pad_or_sample_sequence(sequence, SEQUENCE_LENGTH)


def save_dynamic_sequence(sequence, class_folder, saved_count):
    """Menyimpan satu sequence landmark dinamis."""
    if sequence is None:
        print("[WARNING] Data gagal diambil.")
        return saved_count

    if np.count_nonzero(sequence) == 0:
        print("[WARNING] Tidak ada landmark tangan. Data tidak disimpan.")
        return saved_count

    sample_index = get_next_sample_index(class_folder)
    save_path = os.path.join(
        class_folder,
        f"sample_{sample_index:03d}.npy"
    )

    np.save(save_path, sequence.astype(np.float32))
    saved_count += 1

    print(f"[DYNAMIC] Data tersimpan: {save_path}")
    print("Shape:", sequence.shape)

    return saved_count


def collect_dataset(
    class_name,
    data_type,
    total_samples,
    capture_seconds,
    countdown_seconds,
    static_delay
):
    """Mengumpulkan dataset untuk satu kelas."""
    class_name = normalize_class_name(class_name)
    display_name = get_display_name(class_name)

    class_folder = os.path.join(DATASET_PATH, class_name)

    create_folder(DATASET_PATH)
    create_folder(class_folder)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Kamera tidak bisa dibuka.")
        return False

    saved_count = get_existing_sample_count(class_folder)

    print("\n======================================")
    print("PENGUMPULAN DATASET LANDMARK")
    print("Kelas        :", display_name)
    print("Nama folder  :", class_name)
    print("Tipe data    :", data_type)
    print("Total target :", total_samples)
    print("Sudah ada    :", saved_count)
    print("Folder       :", class_folder)
    print("======================================")

    if saved_count >= total_samples:
        print("Target data sudah terpenuhi. Tidak perlu mengambil data lagi.")
        cap.release()
        cv2.destroyAllWindows()
        return True

    if data_type == "static":
        print("Tekan S satu kali untuk mengambil seluruh sample static secara otomatis.")
    else:
        print("Tekan S setiap kali ingin mengambil satu sample gerakan dynamic.")

    print("Tekan Q untuk keluar.")
    print("======================================")

    continue_to_next_class = True

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while saved_count < total_samples:
            ret, frame = cap.read()

            if not ret:
                print("[ERROR] Frame kamera tidak dapat dibaca.")
                continue_to_next_class = False
                break

            frame = cv2.flip(frame, 1)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            draw_landmarks(frame, results)

            if data_type == "static":
                draw_panel(
                    frame,
                    f"Kelas: {display_name} | Mode: STATIC AUTO",
                    f"Tersimpan: {saved_count}/{total_samples} | Tekan S sekali untuk mulai | Q keluar"
                )
            else:
                draw_panel(
                    frame,
                    f"Kelas: {display_name} | Mode: DYNAMIC",
                    f"Tersimpan: {saved_count}/{total_samples} | Tekan S untuk rekam gerakan | Q keluar"
                )

            cv2.imshow("Collect Landmark Dataset", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                continue_to_next_class = False
                break

            if key != ord("s"):
                continue

            lanjut = countdown(cap, hands, countdown_seconds)

            if not lanjut:
                continue_to_next_class = False
                break

            if data_type == "static":
                saved_count = collect_static_auto(
                    cap=cap,
                    hands=hands,
                    class_folder=class_folder,
                    class_name=class_name,
                    total_samples=total_samples,
                    saved_count=saved_count,
                    delay_seconds=static_delay
                )

            elif data_type == "dynamic":
                sequence = collect_dynamic_sequence(
                    cap=cap,
                    hands=hands,
                    capture_seconds=capture_seconds
                )

                saved_count = save_dynamic_sequence(
                    sequence=sequence,
                    class_folder=class_folder,
                    saved_count=saved_count
                )

    cap.release()
    cv2.destroyAllWindows()

    print("\nSelesai untuk kelas:", display_name)
    print("Total data sekarang:", saved_count)
    print("Folder dataset     :", class_folder)

    return continue_to_next_class


def collect_all_static_classes(
    total_samples,
    capture_seconds,
    countdown_seconds,
    static_delay
):
    """
    Mengumpulkan delapan kata static secara berurutan.
    Tekan Enter sebelum berpindah ke kelas berikutnya.
    """
    print("\n======================================")
    print("MODE PENGUMPULAN SEMUA KELAS STATIC")
    print("======================================")
    print("Kelas yang akan dikumpulkan:")

    for index, class_name in enumerate(STATIC_CLASSES, start=1):
        print(f"{index}. {get_display_name(class_name)}")

    print("\nSetiap kelas akan disimpan di folder yang berbeda.")
    print("Untuk setiap kelas, tekan S satu kali agar pengambilan sample dimulai.")

    for index, class_name in enumerate(STATIC_CLASSES, start=1):
        print("\n--------------------------------------")
        print(
            f"Kelas {index}/{len(STATIC_CLASSES)}: "
            f"{get_display_name(class_name)}"
        )
        print("--------------------------------------")

        input("Tekan Enter untuk membuka kamera dan mulai kelas ini...")

        lanjut = collect_dataset(
            class_name=class_name,
            data_type="static",
            total_samples=total_samples,
            capture_seconds=capture_seconds,
            countdown_seconds=countdown_seconds,
            static_delay=static_delay
        )

        if not lanjut:
            print("\nPengumpulan seluruh kelas dihentikan oleh pengguna.")
            return

    print("\n======================================")
    print("SELURUH DATA STATIC SELESAI DIKUMPULKAN")
    print("======================================")


def print_static_classes():
    """Menampilkan daftar kelas static yang tersedia."""
    print("\nDaftar kelas STATIC:")
    for index, class_name in enumerate(STATIC_CLASSES, start=1):
        print(f"{index}. {get_display_name(class_name)} -> folder: {class_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Program pengumpulan dataset landmark tangan static dan dynamic"
    )

    parser.add_argument(
        "--class_name",
        type=str,
        help="Nama kelas, contoh: benar, sama-sama, terima kasih, atau selamat_malam"
    )

    parser.add_argument(
        "--type",
        type=str,
        choices=["static", "dynamic"],
        help="Pilih static untuk pose diam atau dynamic untuk gerakan"
    )

    parser.add_argument(
        "--collect_all_static",
        action="store_true",
        help="Kumpulkan seluruh kata static bawaan secara berurutan"
    )

    parser.add_argument(
        "--list_static",
        action="store_true",
        help="Tampilkan daftar kata static bawaan"
    )

    parser.add_argument(
        "--total_samples",
        type=int,
        default=50,
        help="Jumlah total sample untuk setiap kelas"
    )

    parser.add_argument(
        "--capture_seconds",
        type=float,
        default=3,
        help="Durasi pengambilan gerakan untuk data dynamic"
    )

    parser.add_argument(
        "--countdown_seconds",
        type=float,
        default=3,
        help="Durasi hitung mundur sebelum pengambilan data"
    )

    parser.add_argument(
        "--static_delay",
        type=float,
        default=0.3,
        help="Jeda antar-sample static otomatis dalam detik"
    )

    args = parser.parse_args()

    if args.list_static:
        print_static_classes()
        return

    if args.collect_all_static:
        collect_all_static_classes(
            total_samples=args.total_samples,
            capture_seconds=args.capture_seconds,
            countdown_seconds=args.countdown_seconds,
            static_delay=args.static_delay
        )
        return

    if not args.class_name or not args.type:
        parser.error(
            "Gunakan --class_name dan --type, "
            "atau gunakan --collect_all_static."
        )

    class_name = normalize_class_name(args.class_name)

    if class_name in STATIC_CLASSES and args.type != "static":
        parser.error(
            f"Kelas '{class_name}' termasuk kata static. "
            "Gunakan --type static."
        )

    collect_dataset(
        class_name=class_name,
        data_type=args.type,
        total_samples=args.total_samples,
        capture_seconds=args.capture_seconds,
        countdown_seconds=args.countdown_seconds,
        static_delay=args.static_delay
    )


if __name__ == "__main__":
    main()
