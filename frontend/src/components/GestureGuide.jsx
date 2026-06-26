import React, { useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  BookOpen,
  Hand,
  Hash,
  MessageSquareText,
  PlayCircle,
} from "lucide-react";


// ============================================================
// DATA ALFABET A-Z
// ============================================================
const ALPHABET_GESTURES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  .split("")
  .map((label) => ({
    label,
    type: "image",
    src: `/gesture-guide/alfabet/${label}.jpg`,
  }));


// ============================================================
// DATA ANGKA 0-9
// ============================================================
const NUMBER_GESTURES = Array.from(
  { length: 10 },
  (_, index) => ({
    label: String(index),
    type: "image",
    src: `/gesture-guide/angka/${index}.jpg`,
  })
);


// ============================================================
// DATA KATA STATIC
// ============================================================
const STATIC_WORD_GESTURES = [
  {
    label: "Benar",
    type: "image",
    src: "/gesture-guide/kata_static/benar.jpg",
  },
  {
    label: "Bertemu",
    type: "image",
    src: "/gesture-guide/kata_static/bertemu.jpg",
  },
  {
    label: "Kamu",
    type: "image",
    src: "/gesture-guide/kata_static/kamu.jpg",
  },
  {
    label: "Kapan",
    type: "image",
    src: "/gesture-guide/kata_static/kapan.jpg",
  },
  {
    label: "Bis",
    type: "image",
    src: "/gesture-guide/kata_static/bis.jpg",
  },
  {
    label: "Mobil",
    type: "image",
    src: "/gesture-guide/kata_static/mobil.jpg",
  },
  {
    label: "Makan",
    type: "image",
    src: "/gesture-guide/kata_static/makan.jpg",
  },
  {
    label: "Minum",
    type: "image",
    src: "/gesture-guide/kata_static/minum.jpg",
  },
  {
    label: "Motor",
    type: "image",
    src: "/gesture-guide/kata_static/motor.jpg",
  },
  {
    label: "Sama-sama",
    type: "image",
    src: "/gesture-guide/kata_static/sama-sama.jpg",
  },
  {
    label: "Terima Kasih",
    type: "image",
    src: "/gesture-guide/kata_static/terimakasih.jpg",
  },
];


// ============================================================
// DATA KATA DYNAMIC
// ============================================================
const DYNAMIC_WORD_GESTURES = [
  {
    label: "Selamat Pagi",
    type: "video",
    src: "/gesture-guide/kata_dynamic/Selamat_Pagi.mp4",
  },
  {
    label: "Selamat Siang",
    type: "video",
    src: "/gesture-guide/kata_dynamic/Selamat_siang.mp4",
  },
  {
    label: "Selamat Sore",
    type: "video",
    src: "/gesture-guide/kata_dynamic/Selamat_sore.mp4",
  },
  {
    label: "Selamat Malam",
    type: "video",
    src: "/gesture-guide/kata_dynamic/Selamat_malam.mp4",
  },
];


// ============================================================
// KONFIGURASI TAB
// ============================================================
const TABS = [
  {
    id: "alfabet",
    label: "Alfabet A-Z",
    icon: Hand,
  },
  {
    id: "angka",
    label: "Angka 0-9",
    icon: Hash,
  },
  {
    id: "static",
    label: "Kata Statis",
    icon: MessageSquareText,
  },
  {
    id: "dynamic",
    label: "Kata Dinamis",
    icon: PlayCircle,
  },
];


export default function GestureGuide() {
  const [activeTab, setActiveTab] = useState("alfabet");
  const [searchTerm, setSearchTerm] = useState("");

  const currentGestures = useMemo(() => {
    if (activeTab === "angka") {
      return NUMBER_GESTURES;
    }

    if (activeTab === "static") {
      return STATIC_WORD_GESTURES;
    }

    if (activeTab === "dynamic") {
      return DYNAMIC_WORD_GESTURES;
    }

    return ALPHABET_GESTURES;
  }, [activeTab]);

  const filteredGestures = useMemo(() => {
    const keyword = searchTerm.trim().toLowerCase();

    if (!keyword) {
      return currentGestures;
    }

    return currentGestures.filter((gesture) =>
      gesture.label.toLowerCase().includes(keyword)
    );
  }, [currentGestures, searchTerm]);

  return (
    <motion.section
      id="panduan-gerakan"
      initial={{ opacity: 0, y: 70 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.1 }}
      transition={{ duration: 0.7, ease: "easeOut" }}
      className="
        scroll-mt-24
        bg-gradient-to-b
        from-slate-100
        to-white
        px-6
        py-24
        md:px-8
      "
    >
      <div className="mx-auto max-w-7xl">
        {/* TITLE */}
        <div className="mx-auto mb-12 max-w-3xl text-center">
          <div
            className="
              mx-auto
              mb-5
              flex
              h-16
              w-16
              items-center
              justify-center
              rounded-2xl
              bg-violet-100
              text-violet-600
            "
          >
            <BookOpen size={34} />
          </div>

          <h2
            className="
              text-4xl
              font-black
              tracking-tight
              text-slate-900
              md:text-5xl
            "
          >
            Panduan Gerakan BISINDO
          </h2>

          <p
            className="
              mt-5
              text-lg
              leading-8
              text-slate-600
            "
          >
            Pelajari bentuk tangan alfabet, angka, dan kata
            sebelum menggunakan fitur deteksi secara real-time.
          </p>
        </div>


        {/* TAB BUTTONS */}
        <div
          className="
            mb-8
            flex
            flex-wrap
            justify-center
            gap-3
          "
        >
          {TABS.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;

            return (
              <button
                key={tab.id}
                onClick={() => {
                  setActiveTab(tab.id);
                  setSearchTerm("");
                }}
                className={`
                  flex
                  items-center
                  gap-2
                  rounded-full
                  px-5
                  py-3
                  font-bold
                  transition
                  duration-300
                  ${
                    isActive
                      ? "bg-violet-600 text-white shadow-lg shadow-violet-200"
                      : "border border-violet-100 bg-white text-slate-700 hover:bg-violet-50"
                  }
                `}
              >
                <Icon size={19} />
                {tab.label}
              </button>
            );
          })}
        </div>


        {/* SEARCH */}
        <div className="mx-auto mb-10 max-w-xl">
          <input
            type="text"
            value={searchTerm}
            onChange={(event) =>
              setSearchTerm(event.target.value)
            }
            placeholder="Cari alfabet, angka, atau kata..."
            className="
              w-full
              rounded-2xl
              border
              border-slate-200
              bg-white
              px-5
              py-4
              text-base
              font-semibold
              text-slate-700
              outline-none
              shadow-sm
              transition
              focus:border-violet-500
              focus:ring-4
              focus:ring-violet-100
            "
          />
        </div>


        {/* GESTURE GRID */}
        {filteredGestures.length > 0 ? (
          <div
            className="
              grid
              gap-6
              sm:grid-cols-2
              lg:grid-cols-3
              xl:grid-cols-4
            "
          >
            {filteredGestures.map((gesture) => (
              <motion.article
                key={`${activeTab}-${gesture.label}`}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.25 }}
                className="
                  overflow-hidden
                  rounded-[26px]
                  border
                  border-violet-100
                  bg-white
                  shadow-lg
                  shadow-slate-200
                  transition
                  duration-300
                  hover:-translate-y-2
                  hover:shadow-xl
                "
              >
                <div
                  className="
                    flex
                    h-64
                    items-center
                    justify-center
                    bg-slate-100
                  "
                >
                  {gesture.type === "video" ? (
                    <video
                      src={gesture.src}
                      controls
                      loop
                      muted
                      playsInline
                      className="
                        h-full
                        w-full
                        object-cover
                      "
                    />
                  ) : (
                    <img
                      src={gesture.src}
                      alt={`Gerakan ${gesture.label}`}
                      loading="lazy"
                      className="
                        h-full
                        w-full
                        object-contain
                        p-4
                      "
                    />
                  )}
                </div>

                <div className="p-5">
                  <p
                    className="
                      text-sm
                      font-extrabold
                      uppercase
                      tracking-[0.15em]
                      text-violet-500
                    "
                  >
                    {gesture.type === "video"
                      ? "Gerakan Dinamis"
                      : "Pose Tangan"}
                  </p>

                  <h3
                    className="
                      mt-2
                      text-2xl
                      font-black
                      text-slate-900
                    "
                  >
                    {gesture.label}
                  </h3>
                </div>
              </motion.article>
            ))}
          </div>
        ) : (
          <div
            className="
              rounded-3xl
              border
              border-dashed
              border-slate-300
              bg-white
              px-6
              py-14
              text-center
            "
          >
            <p className="text-lg font-bold text-slate-500">
              Gerakan tidak ditemukan.
            </p>
          </div>
        )}
      </div>
    </motion.section>
  );
}