/** @type {import('tailwindcss').Config} */
// Colors are wired to CSS variables (see index.css) so the whole palette flips
// with [data-theme]. Tailwind just injects the var; alpha is baked into the token.
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: ["selector", '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        // surfaces
        desk: "var(--desk)",
        rail: "var(--railBg)",
        center: "var(--centerBg)",
        right: "var(--rightBg)",
        // text (ink)
        ink: {
          DEFAULT: "var(--text)",
          2: "var(--text2)",
          3: "var(--text3)",
          4: "var(--text4)",
        },
        // lines
        line: "var(--divider)",
        edge: "var(--border)",
        // fills
        chip: "var(--chip)",
        rowsel: "var(--rowSel)",
        rowhover: "var(--rowHover)",
        mention: "var(--mention)",
        // semantic status
        pos: { DEFAULT: "var(--green)", dim: "var(--greenDim)" },
        neg: { DEFAULT: "var(--red)", dim: "var(--redDim)" },
      },
      boxShadow: {
        win: "var(--winShadow)",
        tab: "0 1px 2px rgba(0,0,0,.12)",
      },
      fontFamily: {
        sans: ["Inter", "'Helvetica Neue'", "Helvetica", "Arial", "sans-serif"],
        mono: ["'JetBrains Mono'", "ui-monospace", "Menlo", "monospace"],
      },
      keyframes: {
        ppulse: { "0%,100%": { opacity: "1" }, "50%": { opacity: ".3" } },
        pdots: { "0%": { opacity: ".2" }, "20%": { opacity: "1" }, "100%": { opacity: ".2" } },
        ptick: {
          from: { opacity: "0", transform: "translateY(3px)" },
          to: { opacity: "1", transform: "none" },
        },
      },
      animation: {
        ppulse: "ppulse 1.5s ease-in-out infinite",
        pdots: "pdots 1.2s infinite",
        ptick: "ptick .4s ease",
      },
    },
  },
  plugins: [],
};
