/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0D0F12",
        surface: "#161820",
        raised: "#1E2028",
        border: "#2A2D3A",
        brand: "#7C6AF7",
        "brand-dim": "#3D3578",
        confirmed: "#2ECC71",
        inconclusive: "#6B7280",
        refuted: "#E74C3C",
        running: "#3B82F6",
        warning: "#F59E0B",
        "text-primary": "#E8E9ED",
        "text-secondary": "#9CA3AF",
        "text-muted": "#4B5563",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["'JetBrains Mono'", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      keyframes: {
        pulseSoft: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.55" },
        },
        fadeInUp: {
          "0%": { opacity: "0", transform: "translateY(4px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        pulseSoft: "pulseSoft 2s ease-in-out infinite",
        fadeInUp: "fadeInUp 150ms ease-out",
      },
    },
  },
  plugins: [],
};
