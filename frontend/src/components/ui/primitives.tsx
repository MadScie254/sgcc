import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode, TextareaHTMLAttributes } from "react";
import clsx from "clsx";

export function cn(...inputs: Array<string | false | null | undefined>) {
  return clsx(inputs);
}

export function Panel({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={cn("rounded-lg border border-border bg-surface", className)}>{children}</div>;
}

export function Button({ children, className = "", ...props }: ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...props}
      className={cn(
        "inline-flex items-center justify-center rounded-md border border-border bg-surface px-4 py-2 text-sm font-semibold text-primary transition hover:border-accent hover:bg-accent-bg disabled:cursor-not-allowed disabled:opacity-60",
        className,
      )}
    >
      {children}
    </button>
  );
}

export function Input(props: InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={cn("w-full rounded-md border border-border bg-surface px-4 py-3 text-sm text-primary outline-none placeholder:text-muted focus:ring-1 focus:ring-accent", props.className)} />;
}

export function TextArea(props: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea {...props} className={cn("w-full rounded-md border border-border bg-surface px-4 py-3 text-sm text-primary outline-none placeholder:text-muted focus:ring-1 focus:ring-accent", props.className)} />;
}

export function Badge({ children, tone = "signal" }: { children: ReactNode; tone?: "signal" | "alert" | "muted" }) {
  const toneClasses = {
    signal: "bg-success-bg text-success border-success/30",
    alert: "bg-danger-bg text-danger border-danger/30",
    muted: "bg-surface-alt text-secondary border-border",
  };
  return <span className={cn("inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em]", toneClasses[tone])}>{children}</span>;
}