import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";
import { Badge } from "@/components/ui/primitives";

const navItems = [
  { to: "/", label: "Dashboard" },
  { to: "/eda", label: "EDA" },
  { to: "/train", label: "Train" },
  { to: "/predict", label: "Predict" },
  { to: "/explain", label: "Explain" },
  { to: "/upload", label: "Upload" },
  { to: "/compare", label: "Compare" },
  { to: "/monitor", label: "Monitor" },
  { to: "/research-validation", label: "Research Validation" },
];

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen text-gp-text">
      <div className="mx-auto flex min-h-screen max-w-[1600px] gap-6 p-4 lg:p-6">
        <aside className="hidden w-72 shrink-0 flex-col rounded-[28px] border border-gp-border bg-[rgba(10,16,24,0.84)] p-5 shadow-[0_28px_80px_rgba(0,0,0,0.45)] backdrop-blur xl:flex">
          <div className="mb-8">
            <Badge tone="signal">SGCC Theft Detector</Badge>
            <h1 className="mt-4 font-display text-3xl font-semibold tracking-tight">Grid Pulse</h1>
            <p className="mt-2 text-sm leading-6 text-gp-text-muted">
              Detection, explainability, monitoring, and training in one operational console.
            </p>
          </div>

          <nav className="space-y-2">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  [
                    "block rounded-2xl border px-4 py-3 text-sm font-medium transition",
                    isActive
                      ? "border-gp-signal/50 bg-gp-signal-dim text-gp-signal"
                      : "border-transparent text-gp-text-muted hover:border-gp-border hover:bg-gp-panel-alt hover:text-gp-text",
                  ].join(" ")
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </aside>

        <main className="flex min-w-0 flex-1 flex-col gap-6">
          <header className="rounded-[28px] border border-gp-border bg-[rgba(9,14,20,0.72)] px-5 py-4 shadow-[0_22px_60px_rgba(0,0,0,0.35)] backdrop-blur lg:px-7">
            <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
              <div>
                <p className="font-data text-xs uppercase tracking-[0.24em] text-gp-text-dim">SGCC OPERATIONS CONSOLE</p>
                <h2 className="mt-1 font-display text-2xl font-semibold tracking-tight">Theft detection with live model telemetry</h2>
              </div>
              <div className="flex flex-wrap gap-2 text-xs text-gp-text-muted">
                <Badge tone="muted">FastAPI API</Badge>
                <Badge tone="muted">React 19</Badge>
                <Badge tone="muted">Tailwind v4</Badge>
              </div>
            </div>
          </header>

          <section className="flex-1">{children}</section>
        </main>
      </div>
    </div>
  );
}