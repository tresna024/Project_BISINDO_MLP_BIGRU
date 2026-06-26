import React from "react";
import { motion } from "framer-motion";
import heroImage from "/assets/hero.png";

export default function HeroSection() {
  return (
    <motion.section
      id="home"
      initial={{ opacity: 0, scale: 0.9, y: 60 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ duration: 0.8, ease: "easeOut" }}
      className="scroll-mt-28 mx-auto grid min-h-[720px] max-w-7xl items-center gap-20 px-8 py-24 md:grid-cols-2"
    >
      <div className="relative">
        <div className="relative h-[420px] overflow-hidden rounded-[32px] border border-violet-200 bg-white shadow-2xl">
          {/* Gambar */}
          <img
            src={heroImage}
            alt="Sign language detection"
            className="h-full w-full object-cover"
          />

          {/* Live Detection Badge */}
          <div className="absolute left-6 top-6 z-20">
            <div className="flex items-center gap-3 rounded-full bg-white px-5 py-3 shadow-lg">
              <span className="relative flex h-3.5 w-3.5">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-green-400 opacity-75"></span>
                <span className="relative inline-flex h-3.5 w-3.5 rounded-full bg-green-500 shadow-[0_0_12px_rgba(34,197,94,0.9)]"></span>
              </span>

              <span className="text-sm font-extrabold tracking-wide text-slate-800">
                LIVE DETECTION ACTIVE
              </span>
            </div>
          </div>
        </div>

        {/* FLOATING BUBBLES */}
        <div className="absolute right-[-10px] top-20 hidden rounded-2xl bg-white px-7 py-5 text-2xl font-bold text-violet-600 shadow-xl md:block">
          Sign
        </div>

        <div className="absolute right-[-25px] top-44 hidden rounded-2xl bg-white px-7 py-5 text-2xl font-bold text-violet-600 shadow-xl md:block">
          Language
        </div>

        <div className="absolute right-[-5px] top-72 hidden rounded-2xl bg-white px-7 py-5 text-2xl font-bold text-violet-600 shadow-xl md:block">
          AI Detection
        </div>
      </div>

      <div>
        <p className="mb-5 text-sm font-extrabold uppercase tracking-[0.25em] text-violet-600">
          New: Real-Time ML Processing
        </p>

        <h1 className="text-6xl font-black leading-[1.05] tracking-tight text-slate-900 md:text-7xl">
          Sign Language to Text,{" "}
          <span className="text-violet-600">Instantly.</span>
        </h1>

        <p className="mt-8 max-w-2xl text-xl leading-10 text-slate-600">
          Website ini menjadi jembatan antara user dan model bahasa isyarat.
          Gerakan tangan dari kamera akan diproses menggunakan model MLP dan
          ditampilkan menjadi teks secara real-time.
        </p>

        <div className="mt-10 flex flex-wrap gap-5">
          <a
            href="#deteksi"
            className="rounded-full bg-violet-600 px-10 py-5 text-lg font-bold text-white shadow-xl shadow-violet-300 transition duration-300 hover:scale-105 hover:bg-violet-700"
          >
            Get Started
          </a>

          <a
            href="#about"
            className="rounded-full border border-slate-300 bg-white px-10 py-5 text-lg font-bold text-violet-600 transition duration-300 hover:border-violet-400 hover:bg-violet-50"
          >
            Watch Demo
          </a>
        </div>
      </div>
    </motion.section>
  );
}
