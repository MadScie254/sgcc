import { useEffect, useState, type ButtonHTMLAttributes, type ReactNode } from "react";
import { AlertTriangle, Loader2 } from "lucide-react";
import { cn } from "@/lib/cn";
import { apiErrorMessage, type CaseStatus } from "@/lib/api";
import { STATUS_META } from "@/lib/status";

export function Card({ children, className, label }: { children: ReactNode; className?: string; label?: string }) {
  return (
    <section aria-label={label} className={cn("rounded-xl border border-line bg-surface", className)}>
      {children}
    </section>
  );
}

export function CardHeader({ title, subtitle, action }: { title: ReactNode; subtitle?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4">
      <div className="flex flex-col gap-1">
        <h2 className="m-0 font-display text-[22px] font-medium leading-tight">{title}</h2>
        {subtitle ? <p className="m-0 text-[13px] text-ink-2">{subtitle}</p> : null}
      </div>
      {action}
    </div>
  );
}

export function PageHeader({ eyebrow, title, description, actions }: { eyebrow?: ReactNode; title: string; description?: ReactNode; actions?: ReactNode }) {
  return (
    <header className="flex flex-wrap items-end justify-between gap-6">
      <div className="flex max-w-3xl flex-col gap-1.5">
        {eyebrow ? <span className="text-xs uppercase tracking-[0.08em] text-ink-3">{eyebrow}</span> : null}
        <h1 className="m-0 font-display text-[40px] font-medium leading-[1.1] tracking-[-0.01em]">{title}</h1>
        {description ? <p className="m-0 text-[15px] leading-relaxed text-ink-2">{description}</p> : null}
      </div>
      {actions ? <div className="flex flex-wrap items-center gap-2.5">{actions}</div> : null}
    </header>
  );
}

export function Stat({ label, value, hint, tone = "ink" }: { label: string; value: ReactNode; hint?: ReactNode; tone?: "ink" | "risk" | "cobalt" }) {
  return (
    <div className="flex flex-col gap-2.5 rounded-xl border border-line bg-surface px-[22px] py-5">
      <span className="text-[13px] text-ink-2">{label}</span>
      <span className={cn("font-display text-[40px] font-medium leading-none tabular", tone === "risk" && "text-risk", tone === "cobalt" && "text-cobalt")}>{value}</span>
      {hint ? <span className="text-xs text-ink-3">{hint}</span> : null}
    </div>
  );
}

type ButtonVariant = "primary" | "secondary" | "ghost";

export function Button({ variant = "secondary", busy, className, children, ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: ButtonVariant; busy?: boolean }) {
  return (
    <button
      type="button"
      {...props}
      disabled={props.disabled || busy}
      className={cn(
        "inline-flex h-11 shrink-0 items-center justify-center gap-2 whitespace-nowrap rounded-lg px-4 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-55",
        variant === "primary" && "bg-ink text-white hover:bg-[#2A2E36]",
        variant === "secondary" && "border border-line bg-surface text-ink hover:border-ink-3",
        variant === "ghost" && "text-ink-2 hover:bg-tint hover:text-ink",
        className,
      )}
    >
      {busy ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : null}
      {children}
    </button>
  );
}

export function Segmented<T extends string>({ value, options, onChange, label }: { value: T; options: Array<{ value: T; label: ReactNode }>; onChange: (value: T) => void; label: string }) {
  return (
    <div role="radiogroup" aria-label={label} className="flex gap-1 rounded-xl bg-[#E9E7E1] p-1">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          role="radio"
          aria-checked={value === option.value}
          onClick={() => onChange(option.value)}
          className={cn(
            "h-9 rounded-[9px] px-4 text-[13px] font-medium transition-colors",
            value === option.value ? "bg-surface text-ink shadow-[0_1px_2px_rgba(20,22,27,0.12)]" : "text-ink-2 hover:text-ink",
          )}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

export function Pill({ active, children, onClick }: { active?: boolean; children: ReactNode; onClick?: () => void }) {
  return (
    <button
      type="button"
      aria-pressed={active}
      onClick={onClick}
      className={cn(
        "h-9 rounded-full border px-3.5 text-[13px] transition-colors",
        active ? "border-ink bg-ink text-white" : "border-line bg-surface text-ink hover:border-ink-3",
      )}
    >
      {children}
    </button>
  );
}

export function TierBadge({ tier, solid }: { tier: string; solid?: boolean }) {
  const label = tier === "high" ? "High" : tier === "medium" ? "Medium" : "Low";
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold",
        tier === "high" && (solid ? "bg-risk text-white" : "bg-risk-bg text-risk-text"),
        tier === "medium" && "bg-amber-bg text-amber-ink",
        tier === "low" && "bg-tint text-ink-2",
      )}
    >
      {solid ? `${label} risk` : label}
    </span>
  );
}

export function StatusBadge({ status }: { status: CaseStatus }) {
  const meta = STATUS_META[status] ?? STATUS_META.new;
  return <span className={cn("inline-flex items-center whitespace-nowrap rounded-full px-2.5 py-1 text-xs font-medium", meta.className)}>{meta.label}</span>;
}

export function Skeleton({ className }: { className?: string }) {
  return <div aria-hidden className={cn("animate-pulse rounded-lg bg-tint", className)} />;
}

export function ErrorState({ title = "Could not load this view", message }: { title?: string; message: string }) {
  return (
    <div role="alert" className="flex items-start gap-3 rounded-xl border border-risk/30 bg-risk-bg px-5 py-4 text-risk-ink">
      <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" aria-hidden />
      <div className="flex flex-col gap-1">
        <span className="font-medium">{title}</span>
        <span className="text-sm">{message}</span>
      </div>
    </div>
  );
}

/** An error from an API call, showing the server's own explanation when it gives one. */
export function ApiError({ error, title }: { error: unknown; title?: string }) {
  const [message, setMessage] = useState("");
  useEffect(() => {
    let live = true;
    void apiErrorMessage(error).then((text) => live && setMessage(text));
    return () => {
      live = false;
    };
  }, [error]);
  return <ErrorState title={title} message={message || "…"} />;
}

export function Empty({ title, message }: { title: string; message?: string }) {
  return (
    <div className="flex flex-col items-center gap-1 rounded-xl border border-dashed border-line px-6 py-10 text-center">
      <span className="font-medium">{title}</span>
      {message ? <span className="text-sm text-ink-3">{message}</span> : null}
    </div>
  );
}
