import { useState, type ReactNode } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Activity, Gauge, LayoutGrid, Menu, Search, Settings, SlidersHorizontal, Workflow, X, FileText } from "lucide-react";
import { getModelMetrics, getPipelineRuns } from "@/lib/api";
import { fmtRelative } from "@/lib/format";
import { cn } from "@/lib/cn";

const NAV = [
  { to: "/", label: "Command center", icon: LayoutGrid, end: true },
  { to: "/cases", label: "Case files", icon: Search },
  { to: "/pipeline", label: "Pipeline", icon: Workflow },
  { to: "/threshold", label: "Threshold studio", icon: SlidersHorizontal },
  { to: "/model", label: "Model performance", icon: Gauge },
  { to: "/reports", label: "Reports & scoring", icon: FileText },
];

function Logo() {
  return (
    <div className="flex items-center gap-2.5 px-2">
      <svg width="28" height="28" viewBox="0 0 28 28" fill="none" stroke="#F3F2EE" strokeWidth="1.6" aria-hidden>
        <path d="M14 2 L25 8 V20 L14 26 L3 20 V8 Z" />
        <path d="M15.5 7 L10 15 H14 L12.5 21 L18 13 H14 Z" fill="#E0873A" stroke="none" />
      </svg>
      <div className="flex flex-col">
        <span className="font-display text-lg font-semibold text-ground">GridSentinel</span>
        <span className="text-[11px] uppercase tracking-[0.08em] text-night-muted">Revenue protection</span>
      </div>
    </div>
  );
}

function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  const metrics = useQuery({ queryKey: ["model-metrics"], queryFn: getModelMetrics });
  const runs = useQuery({ queryKey: ["pipeline-runs"], queryFn: () => getPipelineRuns(1), refetchInterval: 30_000 });
  const lastRun = runs.data?.[0];

  return (
    <div className="flex h-full flex-col gap-7 px-4 py-7">
      <Logo />
      <nav aria-label="Primary" className="flex flex-col gap-0.5">
        {NAV.map(({ to, label, icon: Icon, end }) => (
          <NavLink key={to} to={to} end={end} onClick={onNavigate}
            className={({ isActive }) => cn("flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-colors",
              isActive ? "bg-night-2 font-medium text-white" : "text-night-text hover:bg-night-2/60 hover:text-white")}>
            <Icon className="h-[18px] w-[18px]" strokeWidth={1.7} aria-hidden />
            {label}
          </NavLink>
        ))}
      </nav>
      <div className="mt-auto flex flex-col gap-3">
        <div className="flex flex-col gap-2 rounded-lg border border-night-line px-3 py-3.5">
          <span className="text-[11px] uppercase tracking-[0.08em] text-night-muted">Model in service</span>
          <span className="font-mono text-[13px] text-ground">xgb v{metrics.data?.model_version ?? "…"}</span>
          <span className="text-xs text-night-muted">τ {metrics.data ? metrics.data.threshold.toFixed(3) : "…"} · AUC {metrics.data ? metrics.data.metrics.auc.toFixed(3) : "…"}</span>
          <span className="flex items-center gap-2 text-xs text-night-muted">
            <Activity className={cn("h-3.5 w-3.5", lastRun?.status === "succeeded" ? "text-[#7FA2FF]" : "text-amber")} aria-hidden />
            {lastRun ? `Scored ${fmtRelative(lastRun.finished_at)}` : "No scoring run yet"}
          </span>
        </div>
        <NavLink to="/settings" onClick={onNavigate}
          className={({ isActive }) => cn("flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm", isActive ? "bg-night-2 text-white" : "text-night-text hover:text-white")}>
          <Settings className="h-[18px] w-[18px]" strokeWidth={1.7} aria-hidden />
          Settings
        </NavLink>
      </div>
    </div>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const location = useLocation();

  return (
    <div className="flex min-h-screen bg-ground text-ink">
      <div className="hidden w-[232px] shrink-0 bg-night lg:block">
        <aside className="sticky top-0 h-screen">
          <SidebarContent />
        </aside>
      </div>

      {open ? (
        <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal="true" aria-label="Navigation">
          <button type="button" aria-label="Close navigation" className="absolute inset-0 bg-black/40" onClick={() => setOpen(false)} />
          <aside className="relative h-full w-[260px] bg-night">
            <button type="button" aria-label="Close navigation" onClick={() => setOpen(false)} className="absolute right-3 top-3 flex h-11 w-11 items-center justify-center text-night-text">
              <X className="h-5 w-5" aria-hidden />
            </button>
            <SidebarContent onNavigate={() => setOpen(false)} />
          </aside>
        </div>
      ) : null}

      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center gap-3 border-b border-line bg-surface px-4 py-2 lg:hidden">
          <button type="button" aria-label="Open navigation" onClick={() => setOpen(true)} className="flex h-11 w-11 items-center justify-center rounded-lg text-ink">
            <Menu className="h-5 w-5" aria-hidden />
          </button>
          <span className="font-display text-lg font-semibold">GridSentinel</span>
        </div>
        <main key={location.pathname} className="flex w-full max-w-[1480px] animate-fade-up flex-col gap-6 px-4 py-6 sm:px-8 lg:px-10 lg:py-8">
          {children}
        </main>
      </div>
    </div>
  );
}
