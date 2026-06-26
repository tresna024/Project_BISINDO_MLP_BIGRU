import React, { useEffect, useState } from "react";
import { Hand } from "lucide-react";

export default function Navbar() {

  const [activeSection, setActiveSection] = useState("home");

  useEffect(() => {

    const sections = document.querySelectorAll("section");

    const observer = new IntersectionObserver(
      (entries) => {

        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.id);
          }
        });

      },
      {
        threshold: 0.6,
      }
    );

    sections.forEach((section) => {
      observer.observe(section);
    });

    return () => observer.disconnect();

  }, []);

  const navLinkClass = (section) =>
    `relative text-xl font-semibold transition duration-300 
    ${
      activeSection === section
        ? "text-violet-600 after:w-full"
        : "text-slate-500 hover:text-violet-600 after:w-0"
    }
    after:absolute after:-bottom-2 after:left-0 after:h-0.5 
    after:bg-violet-600 after:transition-all after:duration-300 hover:after:w-full`;

  return (
    <header className="sticky top-0 z-50 px-6 py-5">
      <nav className="mx-auto flex max-w-7xl items-center justify-between rounded-full border border-slate-100 bg-white px-8 py-4 shadow-xl shadow-slate-200/70">

        {/* Logo */}
        <a href="#home" className="flex items-center gap-4">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-violet-100 text-violet-600">
            <Hand size={30} strokeWidth={2.5} />
          </div>

          <span className="text-2xl font-extrabold tracking-tight text-slate-800">
            Sign Language
          </span>
        </a>

        {/* Menu */}
        <div className="hidden items-center gap-12 md:flex">

          <a
            href="#home"
            className={navLinkClass("home")}
          >
            Home
          </a>

          <a
            href="#about"
            className={navLinkClass("about")}
          >
            Cara Kerja
          </a>

          <a
            href="#panduan-gerakan"
            className={navLinkClass("panduan-gerakan")}
          >
            Panduan
          </a>

          <a
            href="#deteksi"
            className={navLinkClass("deteksi")}
          >
            Detection
          </a>

        </div>

        {/* Button */}
        <a
          href="#deteksi"
          className="rounded-full bg-violet-600 px-8 py-4 text-base font-bold text-white shadow-lg shadow-violet-300 transition duration-300 hover:scale-105 hover:bg-violet-700"
        >
          Launch App
        </a>

      </nav>
    </header>
  );
}