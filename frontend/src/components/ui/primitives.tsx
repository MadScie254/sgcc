import type { ReactNode } from "react";
import { clsx } from "clsx";

export function cn(...inputs: Array<string | false | null | undefined>) {
  return clsx(inputs);
}

export function Panel({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={cn("rounded-2xl border border-gp-border bg-gp-panel/90 shadow-[0_18px_48px_rgba(0,0,0,0.35)] backdrop-blur", className)}>{children}</div>;
}

export function Button({ children, className = "", ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...props}
      className={cn(
        "inline-flex items-center justify-center rounded-xl border border-gp-border bg-gp-panel-alt px-4 py-2 text-sm font-semibold text-gp-text transition hover:border-gp-signal hover:bg-gp-signal-dim disabled:cursor-not-allowed disabled:opacity-60",
        className,
      )}
    >
      {children}
    </button>
  );
}

export function Input(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={cn("w-full rounded-xl border border-gp-border bg-gp-panel-alt px-4 py-3 text-sm text-gp-text outline-none placeholder:text-gp-text-dim focus:border-gp-signal", props.className)} />;
}

export function TextArea(props: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea {...props} className={cn("w-full rounded-xl border border-gp-border bg-gp-panel-alt px-4 py-3 text-sm text-gp-text outline-none placeholder:text-gp-text-dim focus:border-gp-signal", props.className)} />;
}

export function Badge({ children, tone = "signal" }: { children: ReactNode; tone?: "signal" | "alert" | "muted" }) {
  const toneClasses = {
    signal: "bg-gp-signal-dim text-gp-signal border-gp-signal/40",
    alert: "bg-gp-alert-dim text-gp-alert border-gp-alert/40",
    muted: "bg-gp-panel-alt text-gp-text-muted border-gp-border",
  };
  return <span className={cn("inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em]", toneClasses[tone])}>{children}</span>;
}