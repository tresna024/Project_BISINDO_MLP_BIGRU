import React, { useEffect, useState } from "react";
import { Hand } from "lucide-react";

import Navbar from "./components/Navbar";
import HeroSection from "./components/HeroSection";
import HowItWorks from "./components/How";
import GestureGuide from "./components/GestureGuide";
import DetectionBox from "./components/DetectionBox";
import Footer from "./components/Footer";

export default function App() {

  const [loading, setLoading] = useState(true);

  useEffect(() => {

    const timer = setTimeout(() => {
      setLoading(false);
    }, 1800);

    return () => clearTimeout(timer);

  }, []);

  // LOADING SCREEN
  if (loading) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center bg-white">
        <div className="relative flex h-28 w-28 items-center justify-center rounded-3xl bg-violet-100 text-violet-600 shadow-2xl shadow-violet-200 animate-bounce">
          <div className="absolute inset-0 rounded-3xl bg-violet-300 opacity-20 blur-2xl"></div>
          <Hand
            size={64}
            strokeWidth={2.5}
            className="relative z-10"
          />

        </div>
        <h1 className="mt-8 text-5xl font-black tracking-tight text-violet-600">
          Sign Language
        </h1>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <Navbar />
      <HeroSection />
      <HowItWorks />
      <GestureGuide />
      <section id="deteksi" className="px-8 py-24">
        <DetectionBox />
      </section>
      <Footer />
    </div>
  );
}