import React, { useEffect, useRef, useState } from "react";
import {
  Camera,
  Square,
  Plus,
  Trash2,
  RotateCcw,
} from "lucide-react";
import { motion } from "framer-motion";

import {
  predictSign,
  resetWordSequence,
} from "../services/api";

const WORD_INTERVAL_MS = 100;
const MLP_INTERVAL_MS = 700;

const INVALID_LABELS = new Set([
  "-",
  "Tangan tidak terdeteksi",
  "Mengumpulkan gerakan...",
  "Menganalisis gerakan...",
  "Belum yakin",
]);

function createSessionId() {
  const existingId = sessionStorage.getItem("sequence_session_id");

  if (existingId) {
    return existingId;
  }

  const newId =
    typeof crypto !== "undefined" && crypto.randomUUID
      ? crypto.randomUUID()
      : `session-${Date.now()}-${Math.random()
          .toString(36)
          .slice(2)}`;

  sessionStorage.setItem("sequence_session_id", newId);

  return newId;
}

export default function DetectionBox() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const timerRef = useRef(null);
  const requestRunningRef = useRef(false);
  const cameraActiveRef = useRef(false);
  const modelTypeRef = useRef("kata");
  const sessionIdRef = useRef(createSessionId());

  const [isDetecting, setIsDetecting] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);

  const [modelType, setModelType] = useState("kata");

  const [prediction, setPrediction] = useState("-");
  const [confidence, setConfidence] = useState(0);

  const [candidateLabel, setCandidateLabel] = useState("");

  const [sentence, setSentence] = useState([]);

  const [framesCollected, setFramesCollected] = useState(0);
  const [framesRequired, setFramesRequired] = useState(30);

  const [ready, setReady] = useState(false);


  // RESET INFORMASI HASIL PREDIKSI
  const resetPredictionDisplay = () => {
    setPrediction("-");
    setConfidence(0);
    setCandidateLabel("");

    setFramesCollected(0);
    setFramesRequired(30);

    setReady(false);
  };

  const resetSequenceBuffer = async () => {
    try {
      await resetWordSequence(sessionIdRef.current);
    } catch (error) {
      console.error(
        "Gagal membersihkan buffer BiGRU:",
        error
      );
    }

    resetPredictionDisplay();
  };


  const startCamera = async () => {
    try {
      if (cameraActiveRef.current) {
        return;
      }

      const stream =
        await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      cameraActiveRef.current = true;
      setCameraActive(true);

      // Membersihkan frame lama ketika kamera baru dinyalakan.
      await resetSequenceBuffer();
    } catch (error) {
      console.error(
        "Kamera gagal diaktifkan:",
        error
      );

      alert(
        "Kamera tidak dapat diaktifkan. " +
          "Periksa izin kamera pada browser."
      );
    }
  };


  // ============================================================
  // MENGHENTIKAN TIMER PENGIRIMAN FRAME
  // ============================================================
  const clearDetectionTimer = () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  };


  // ============================================================
  // MENGHENTIKAN DETEKSI REAL-TIME
  // ============================================================
  const stopRealTimeDetection = async () => {
    clearDetectionTimer();

    requestRunningRef.current = false;

    setIsDetecting(false);

    // Reset buffer hanya diperlukan oleh BiGRU.
    if (modelTypeRef.current === "kata") {
      await resetSequenceBuffer();
    }
  };


  // ============================================================
  // MEMATIKAN KAMERA
  // ============================================================
  const stopCamera = async () => {
    await stopRealTimeDetection();

    const stream =
      streamRef.current ||
      videoRef.current?.srcObject;

    if (stream) {
      stream
        .getTracks()
        .forEach((track) => track.stop());
    }

    streamRef.current = null;

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    cameraActiveRef.current = false;

    setCameraActive(false);

    resetPredictionDisplay();
  };


  // ============================================================
  // MENGAMBIL SATU GAMBAR DARI VIDEO
  const captureFrame = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (
      !video ||
      !canvas ||
      video.videoWidth === 0 ||
      video.videoHeight === 0
    ) {
      return null;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");

    context.drawImage(
      video,
      0,
      0,
      canvas.width,
      canvas.height
    );

    // Nilai 0.8 digunakan untuk mengecilkan ukuran request.
    return canvas.toDataURL(
      "image/jpeg",
      0.8
    );
  };


  // MENAMPILKAN HASIL DARI BACKEND
  const applyPredictionResult = (result) => {
    if (result.error) {
      console.error(
        "Backend error:",
        result.error
      );

      return;
    }

    setPrediction(result.label ?? "-");

    setConfidence(
      Number(result.confidence ?? 0)
    );

    setCandidateLabel(
      result.candidate_label ?? ""
    );

    setReady(Boolean(result.ready));

    if (
      typeof result.frames_collected ===
      "number"
    ) {
      setFramesCollected(
        result.frames_collected
      );
    }

    if (
      typeof result.frames_required ===
      "number"
    ) {
      setFramesRequired(
        result.frames_required
      );
    }
  };


  // ============================================================
  // MENGIRIM FRAME KE BACKEND
  // ============================================================
  const detectSign = async () => {
    /*
      requestRunningRef mencegah pengiriman request baru
      ketika request sebelumnya masih diproses backend.
    */
    if (
      !cameraActiveRef.current ||
      requestRunningRef.current
    ) {
      return;
    }

    const image = captureFrame();

    if (!image) {
      return;
    }

    requestRunningRef.current = true;

    try {
      const result = await predictSign(
        image,
        modelTypeRef.current,
        sessionIdRef.current
      );

      applyPredictionResult(result);
    } catch (error) {
      console.error(
        "Prediksi gagal:",
        error
      );
    } finally {
      requestRunningRef.current = false;
    }
  };


  // ============================================================
  // MENJADWALKAN REQUEST BERIKUTNYA
  // ============================================================
  const scheduleNextDetection = () => {
    clearDetectionTimer();

    const interval =
      modelTypeRef.current === "kata"
        ? WORD_INTERVAL_MS
        : MLP_INTERVAL_MS;

    timerRef.current = setTimeout(
      async () => {
        await detectSign();

        if (
          cameraActiveRef.current &&
          timerRef.current
        ) {
          scheduleNextDetection();
        }
      },
      interval
    );
  };


  // ============================================================
  // MEMULAI DETEKSI REAL-TIME
  // ============================================================
  const startRealTimeDetection = () => {
    if (!cameraActiveRef.current) {
      alert(
        "Aktifkan kamera terlebih dahulu"
      );

      return;
    }

    if (timerRef.current) {
      return;
    }

    setIsDetecting(true);

    timerRef.current = setTimeout(
      async () => {
        await detectSign();
        scheduleNextDetection();
      },
      0
    );
  };


  // ============================================================
  // MENGGANTI JENIS MODEL
  const handleModelChange = async (
    event
  ) => {
    const nextModelType =
      event.target.value;

    clearDetectionTimer();

    requestRunningRef.current = false;

    setIsDetecting(false);

    if (
      modelTypeRef.current === "kata" ||
      nextModelType === "kata"
    ) {
      await resetSequenceBuffer();
    } else {
      resetPredictionDisplay();
    }

    modelTypeRef.current =
      nextModelType;

    setModelType(nextModelType);
  };


  // MENAMBAHKAN HASIL KE KALIMAT
  const addWord = async () => {
    if (INVALID_LABELS.has(prediction)) {
      return;
    }

    if (
      modelType === "kata" &&
      !ready
    ) {
      return;
    }

    setSentence(
      (currentSentence) => [
        ...currentSentence,
        prediction,
      ]
    );

    if (modelType === "kata") {
      await resetSequenceBuffer();
    }
  };

  const clearSentence = async () => {
    setSentence([]);

    await resetSequenceBuffer();
  };

  useEffect(() => {
    modelTypeRef.current = modelType;
  }, [modelType]);

  useEffect(() => {
    return () => {
      clearDetectionTimer();

      const stream =
        streamRef.current ||
        videoRef.current?.srcObject;

      if (stream) {
        stream
          .getTracks()
          .forEach((track) =>
            track.stop()
          );
      }
    };
  }, []);

  const progress =
    framesRequired > 0
      ? Math.min(
          100,
          Math.round(
            (framesCollected /
              framesRequired) *
              100
          )
        )
      : 0;

  return (
    <motion.section
      initial={{
        opacity: 0,
        y: 90,
      }}
      whileInView={{
        opacity: 1,
        y: 0,
      }}
      viewport={{
        once: true,
        amount: 0.15,
      }}
      transition={{
        duration: 0.8,
        ease: "easeOut",
      }}
      className="
        scroll-mt-28
        bg-gradient-to-b
        from-white
        to-slate-100
        px-8
        py-28
      "
    >
      <div className="mx-auto max-w-7xl">
        {/* TITLE */}
        <div className="mb-16 text-center">
          <h2
            className="
              mb-4
              text-xl
              font-extrabold
              uppercase
              tracking-[0.2em]
              text-violet-600
            "
          >
            REAL-TIME DETECTION
          </h2>
        </div>


        {/* MAIN CARD */}
        <div
          className="
            grid
            gap-10
            rounded-[36px]
            border
            border-violet-100
            bg-white
            p-8
            shadow-2xl
            md:grid-cols-[1.7fr_1fr]
          "
        >
          {/* LEFT SIDE */}
          <div>
            {/* CAMERA STATUS */}
            <div className="mb-5 flex items-center gap-3">
              <span className="relative flex h-4 w-4">
                <span
                  className={`
                    absolute
                    inline-flex
                    h-full
                    w-full
                    rounded-full
                    opacity-75
                    ${
                      cameraActive
                        ? "animate-ping bg-green-400"
                        : "bg-slate-300"
                    }
                  `}
                />

                <span
                  className={`
                    relative
                    inline-flex
                    h-4
                    w-4
                    rounded-full
                    ${
                      cameraActive
                        ? "bg-green-500"
                        : "bg-slate-400"
                    }
                  `}
                />
              </span>

              <span
                className="
                  text-sm
                  font-extrabold
                  tracking-wide
                  text-slate-700
                "
              >
                {cameraActive
                  ? "LIVE DETECTION ACTIVE"
                  : "CAMERA OFF"}
              </span>
            </div>


            {/* VIDEO */}
            <div
              className="
                overflow-hidden
                rounded-[32px]
                border
                border-violet-100
                bg-black
                shadow-xl
              "
            >
              <video
                ref={videoRef}
                autoPlay
                muted
                playsInline
                className="
                  h-[620px]
                  w-full
                  -scale-x-100
                  object-cover
                "
              />
            </div>

            <canvas
              ref={canvasRef}
              className="hidden"
            />


            {/* CAMERA BUTTONS */}
            <div className="mt-6 flex flex-wrap gap-4">
              <button
                onClick={startCamera}
                className="
                  flex
                  items-center
                  gap-3
                  rounded-full
                  bg-violet-600
                  px-7
                  py-4
                  text-lg
                  font-bold
                  text-white
                  shadow-lg
                  shadow-violet-300
                  transition
                  duration-300
                  hover:scale-105
                  hover:bg-violet-700
                "
              >
                <Camera size={22} />
                Mulai Kamera
              </button>


              <button
                onClick={
                  isDetecting
                    ? stopRealTimeDetection
                    : startRealTimeDetection
                }
                className={`
                  rounded-full
                  px-7
                  py-4
                  text-lg
                  font-bold
                  text-white
                  shadow-lg
                  transition
                  duration-300
                  hover:scale-105
                  ${
                    isDetecting
                      ? "bg-orange-500 shadow-orange-300 hover:bg-orange-600"
                      : "bg-green-600 shadow-green-300 hover:bg-green-700"
                  }
                `}
              >
                {isDetecting
                  ? "Stop Deteksi"
                  : "Mulai Deteksi Real-Time"}
              </button>


              <button
                onClick={stopCamera}
                className="
                  flex
                  items-center
                  gap-3
                  rounded-full
                  bg-red-600
                  px-7
                  py-4
                  text-lg
                  font-bold
                  text-white
                  shadow-lg
                  shadow-red-300
                  transition
                  duration-300
                  hover:scale-105
                  hover:bg-red-700
                "
              >
                <Square size={22} />
                Stop
              </button>
            </div>
          </div>


          {/* RIGHT SIDE */}
          <div className="flex flex-col justify-between">
            <div>
              <h2 className="mb-8 text-4xl font-black text-slate-900">
                Hasil Deteksi
              </h2>


              {/* SELECT MODEL */}
              <div>
                <label className="text-lg font-bold text-slate-700">
                  Pilih Jenis Deteksi
                </label>

                <select
                  value={modelType}
                  onChange={handleModelChange}
                  className="
                    mt-3
                    w-full
                    rounded-2xl
                    border
                    border-slate-300
                    bg-white
                    px-5
                    py-4
                    text-lg
                    font-bold
                    outline-none
                    transition
                    focus:border-violet-600
                  "
                >
                  <option value="kata">
                    Deteksi Kata
                  </option>

                  <option value="angka">
                    Deteksi Angka
                  </option>

                  <option value="alfabet">
                    Deteksi Alfabet
                  </option>
                </select>
              </div>


              {/* PREDICTION */}
              <div
                className="
                  mt-8
                  rounded-[28px]
                  border
                  border-violet-100
                  bg-violet-50
                  p-8
                  shadow-sm
                "
              >
                <p className="text-lg font-bold text-slate-500">
                  Prediksi
                </p>

                <h1
                  className="
                    my-5
                    break-words
                    text-5xl
                    font-black
                    text-violet-600
                  "
                >
                  {prediction}
                </h1>

                <p className="text-lg text-slate-600">
                  Confidence:{" "}
                  <span className="font-bold">
                    {confidence}%
                  </span>
                </p>


                {/* PROGRESS BUFFER BIGRU */}
                {modelType === "kata" && (
                  <div className="mt-5">
                    <div
                      className="
                        mb-2
                        flex
                        justify-between
                        text-sm
                        font-bold
                        text-slate-600
                      "
                    >
                      <span>Buffer gerakan</span>

                      <span>
                        {framesCollected}/
                        {framesRequired} frame
                      </span>
                    </div>

                    <div
                      className="
                        h-3
                        overflow-hidden
                        rounded-full
                        bg-violet-100
                      "
                    >
                      <div
                        className="
                          h-full
                          rounded-full
                          bg-violet-600
                          transition-all
                        "
                        style={{
                          width: `${progress}%`,
                        }}
                      />
                    </div>

                    {candidateLabel &&
                      !ready && (
                        <p
                          className="
                            mt-3
                            text-sm
                            font-semibold
                            text-slate-500
                          "
                        >
                          Kandidat sementara:{" "}
                          {candidateLabel}
                        </p>
                      )}
                  </div>
                )}
              </div>


              {/* SENTENCE */}
              <div
                className="
                  mt-6
                  rounded-[28px]
                  border
                  border-violet-100
                  bg-white
                  p-8
                  shadow-sm
                "
              >
                <p className="text-lg font-bold text-slate-500">
                  Kalimat Terdeteksi
                </p>

                <h3
                  className="
                    mt-5
                    min-h-24
                    text-2xl
                    font-bold
                    leading-10
                    text-slate-800
                  "
                >
                  {sentence.length > 0
                    ? sentence.join(" ")
                    : "Belum ada kata"}
                </h3>
              </div>
            </div>


            {/* ACTION BUTTONS */}
            <div className="mt-8 flex flex-wrap gap-4">
              <button
                onClick={addWord}
                className="
                  flex
                  items-center
                  gap-3
                  rounded-full
                  bg-violet-600
                  px-7
                  py-4
                  text-lg
                  font-bold
                  text-white
                  shadow-lg
                  shadow-violet-300
                  transition
                  duration-300
                  hover:scale-105
                  hover:bg-violet-700
                "
              >
                <Plus size={22} />
                Tambah Kata
              </button>


              <button
                onClick={resetSequenceBuffer}
                className="
                  flex
                  items-center
                  gap-3
                  rounded-full
                  bg-slate-600
                  px-7
                  py-4
                  text-lg
                  font-bold
                  text-white
                  shadow-lg
                  shadow-slate-300
                  transition
                  duration-300
                  hover:scale-105
                  hover:bg-slate-700
                "
              >
                <RotateCcw size={22} />
                Ulang Gesture
              </button>


              <button
                onClick={clearSentence}
                className="
                  flex
                  items-center
                  gap-3
                  rounded-full
                  bg-red-600
                  px-7
                  py-4
                  text-lg
                  font-bold
                  text-white
                  shadow-lg
                  shadow-red-300
                  transition
                  duration-300
                  hover:scale-105
                  hover:bg-red-700
                "
              >
                <Trash2 size={22} />
                Hapus
              </button>
            </div>
          </div>
        </div>
      </div>
    </motion.section>
  );
}