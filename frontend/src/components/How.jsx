import React from "react";
import { motion } from "framer-motion";
import { Brain, Camera, Languages } from "lucide-react";

export default function HowItWorks() {
  return (
    <motion.section
      id="about"
      initial={{ opacity: 0, y: 80 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.2 }}
      transition={{ duration: 0.8, ease: "easeOut" }}
      className="scroll-mt-28 bg-gradient-to-b from-slate-100 to-white px-8 py-28 text-center"
    >
      {/* TITLE */}
      <div className="mx-auto max-w-3xl">
        <h2 className="text-5xl font-black tracking-tight text-slate-900 md:text-6xl">
          How It Works
        </h2>

        <p className="mt-6 text-xl leading-9 text-slate-600">
          Sistem bekerja dalam tiga langkah utama untuk menerjemahkan bahasa
          isyarat menjadi teks secara real-time menggunakan teknologi AI dan
          Machine Learning.
        </p>
      </div>

      {/* CARD */}

      <div className="mx-auto mt-20 grid max-w-7xl gap-8 md:grid-cols-3">
        {/* CARD 1 */}
        <motion.div
          whileHover={{ scale: 1.05, y: -12 }}
          transition={{ type: "spring", stiffness: 250 }}
          className="group rounded-[30px] border border-slate-200 bg-white p-10 text-left shadow-lg"
        >
          <div className="mb-8 flex h-20 w-20 items-center justify-center rounded-3xl bg-violet-100 text-violet-600 transition duration-500 group-hover:bg-violet-600 group-hover:text-white">
            <Camera size={42} strokeWidth={2.3} />
          </div>

          <h3 className="text-3xl font-black text-slate-900">1. Capture</h3>

          <p className="mt-6 text-lg leading-9 text-slate-600">
            Kamera menangkap gerakan tangan user secara real-time menggunakan
            webcam dengan proses tracking yang stabil dan responsif.
          </p>
        </motion.div>

        {/* CARD 2 */}
        <motion.div
          whileHover={{ scale: 1.05, y: -12 }}
          transition={{ type: "spring", stiffness: 250 }}
          className="group rounded-[30px] border border-slate-200 bg-white p-10 text-left shadow-lg"
        >
          <div className="mb-8 flex h-20 w-20 items-center justify-center rounded-3xl bg-violet-100 text-violet-600 transition duration-500 group-hover:bg-violet-600 group-hover:text-white">
            <Brain size={42} strokeWidth={2.3} />
          </div>

          <h3 className="text-3xl font-black text-slate-900">2. Process</h3>

          <p className="mt-6 text-lg leading-9 text-slate-600">
            MediaPipe mengambil landmark tangan, lalu data diproses menggunakan
            model MLP untuk mengenali pola gerakan bahasa isyarat.
          </p>
        </motion.div>

        {/* CARD 3 */}
        <motion.div
          whileHover={{ scale: 1.05, y: -12 }}
          transition={{ type: "spring", stiffness: 250 }}
          className="group rounded-[30px] border border-slate-200 bg-white p-10 text-left shadow-lg"
        >
          <div className="mb-8 flex h-20 w-20 items-center justify-center rounded-3xl bg-violet-100 text-violet-600 transition duration-500 group-hover:bg-violet-600 group-hover:text-white">
            <Languages size={42} strokeWidth={2.3} />
          </div>

          <h3 className="text-3xl font-black text-slate-900">3. Translate</h3>

          <p className="mt-6 text-lg leading-9 text-slate-600">
            Hasil prediksi dari model AI diterjemahkan menjadi teks dan
            ditampilkan langsung pada website secara cepat dan akurat.
          </p>
        </motion.div>
      </div>
    </motion.section>
  );
}
