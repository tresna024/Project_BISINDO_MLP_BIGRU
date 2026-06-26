import React from "react";
import { Hand, Mail, Phone } from "lucide-react";

export default function Footer() {
  return (
    <footer
      id="contact"
      className="mt-24 overflow-hidden bg-gradient-to-r from-violet-700 to-violet-500 px-8 py-16 text-white"
    >
      <div className="mx-auto grid max-w-7xl gap-12 md:grid-cols-4">

        {/* LOGO */}
        <div>
          <div className="mb-6 flex items-center gap-4">
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-white/20">
              <Hand size={30} />
            </div>

            <h2 className="text-3xl font-black">
              Sign Language
            </h2>
          </div>

          <p className="max-w-sm text-lg leading-8 text-violet-100">
            Website berbasis AI untuk menerjemahkan bahasa
            isyarat menjadi teks secara real-time menggunakan
            teknologi Machine Learning.
          </p>
        </div>

        {/* COMPANY */}
        <div>
          <h3 className="mb-6 text-2xl font-bold">
            Company
          </h3>

          <ul className="space-y-4 text-lg text-violet-100">
            <li className="transition hover:translate-x-1 hover:text-white">
              <a href="#home">Home</a>
            </li>

            <li className="transition hover:translate-x-1 hover:text-white">
              <a href="#about">Features</a>
            </li>

            <li className="transition hover:translate-x-1 hover:text-white">
              <a href="#deteksi">Detection</a>
            </li>

            <li className="transition hover:translate-x-1 hover:text-white">
              <a href="#contact">Contact</a>
            </li>
          </ul>
        </div>

        {/* FEATURES */}
        <div>
          <h3 className="mb-6 text-2xl font-bold">
            Features
          </h3>

          <ul className="space-y-4 text-lg text-violet-100">
            <li>Real-Time Detection</li>
            <li>MLP Model</li>
            <li>MediaPipe Tracking</li>
            <li>Sign Translation</li>
          </ul>
        </div>

        {/* NEWSLETTER */}
        <div>
          <h3 className="mb-6 text-2xl font-bold">
            Newsletter
          </h3>

          <p className="mb-5 text-lg leading-8 text-violet-100">
            Dapatkan update terbaru tentang pengembangan
            AI Sign Language Detection.
          </p>

          {/* INPUT */}
          <div className="flex overflow-hidden rounded-full bg-white p-2 shadow-lg">

            <input
              type="email"
              placeholder="Enter your email"
              className="w-full bg-transparent px-4 text-slate-700 outline-none"
            />

            <button className="rounded-full bg-violet-600 px-6 py-3 font-bold text-white transition hover:bg-violet-700">
              Subscribe
            </button>
          </div>

          {/* PHONE */}
          <div className="mt-8 flex items-center gap-4 text-lg font-semibold">
            <Phone size={22} />

            <span>+62 852-6951-7114</span>
          </div>
        </div>
      </div>

      {/* BOTTOM */}
      <div className="mx-auto mt-16 flex max-w-7xl flex-col items-center justify-between gap-4 border-t border-white/20 pt-8 text-violet-100 md:flex-row">

        <p>
          © 2026 Sign Language (Tresna Hidayah). All rights reserved.
        </p>

        <div className="flex items-center gap-6">
          <a href="#" className="transition hover:text-white">
            Privacy Policy
          </a>

          <a href="#" className="transition hover:text-white">
            Terms & Conditions
          </a>
        </div>
      </div>
    </footer>
  );
}