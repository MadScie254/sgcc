/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx,js,jsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#F7F8FA",
        surface: "#FFFFFF",
        "surface-alt": "#F1F3F6",
        border: "#E2E5EA",
        primary: "#131720",
        secondary: "#5B6472",
        muted: "#8B93A1",
        accent: "#2B5FAD",
        "accent-bg": "#EAF1FB",
        danger: "#C0392B",
        "danger-bg": "#FBEAEA",
        warning: "#B7791F",
        "warning-bg": "#FBF2E1",
        success: "#1E7A5F",
        "success-bg": "#E7F5F0",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      borderRadius: {
        md: "8px",
        lg: "12px",
        xl: "16px",
      },
    },
  },
  plugins: [],
};