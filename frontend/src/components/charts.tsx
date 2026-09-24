import { useMemo, useState } from "react";
import type { OperatingPoint, PredictionReason, ScoreDistribution, TimeSeriesPoint } from "@/lib/api";
import { featureLabel, featureValue } from "@/lib/features";
import { fmtInt, fmtNum, fmtPct } from "@/lib/format";
import { cn } from "@/lib/cn";

const RISK = "#B93C15";
const COBALT = "#2346C8";
const AMBER = "#E0873A";
const NEUTRAL = "#C4C9D2";

// ---------------------------------------------------------------------------
// Risk histogram (log scale, stacked by dataset label)
// ---------------------------------------------------------------------------

export function RiskHistogram({ dist, height = 150 }: { dist: ScoreDistribution; height?: number }) {
  const [hover, setHover] = useState<number | null>(null);
  const totals = dist.honest.map((h, i) => h + dist.theft[i]);
  const maxLog = Math.log10(Math.max(...totals, 1) + 1);
  const scale = (n: number) => (n <= 0 ? 0 : (height - 10) * (Math.log10(n + 1) / maxLog));
  const active = hover ?? null;

  return (
    <div className="flex flex-col gap-2">
      <div className="relative flex items-end gap-1 border-b border-[#D9D6CE]" style={{ height }} onMouseLeave={() => setHover(null)}>
        {totals.map((total, i) => {
          const barHeight = scale(total);
          const theftHeight = total ? (barHeight * dist.theft[i]) / total : 0;
          return (
            <button
              key={i}
              type="button"
              aria-label={`Scores ${dist.edges[i].toFixed(2)} to ${dist.edges[i + 1].toFixed(2)}: ${total} customers, ${dist.theft[i]} labelled theft`}
              onMouseEnter={() => setHover(i)}
              onFocus={() => setHover(i)}
              className={cn("flex h-full flex-1 flex-col justify-end transition-opacity", active !== null && active !== i && "opacity-45")}
            >
              <span className="block rounded-t-[2px]" style={{ height: barHeight - theftHeight, background: NEUTRAL }} />
              <span className="block" style={{ height: theftHeight, background: RISK }} />
            </button>
          );
        })}
        <div className="pointer-events-none absolute bottom-0 top-[-6px] border-l-[1.5px] border-dashed border-ink" style={{ left: `${dist.threshold * 100}%` }} />
        <span className="pointer-events-none absolute top-[-6px] font-mono text-[11px]" style={{ left: `calc(${dist.threshold * 100}% + 6px)` }}>
          τ {dist.threshold.toFixed(3)}
        </span>
      </div>
      <div className="flex justify-between font-mono text-[11px] text-ink-3">
        <span>0.0</span>
        <span className="text-ink-2">
          {active !== null
            ? `${dist.edges[active].toFixed(2)}–${dist.edges[active + 1].toFixed(2)} · ${fmtInt(totals[active])} customers · ${fmtInt(dist.theft[active])} theft`
            : "hover a bar"}
        </span>
        <span>1.0</span>
      </div>
    </div>
  );
}

export function Legend({ items }: { items: Array<{ color: string; label: string; line?: boolean }> }) {
  return (
    <div className="flex flex-wrap gap-4 text-xs text-ink-2">
      {items.map((item) => (
        <span key={item.label} className="flex items-center gap-1.5">
          <span className={item.line ? "h-[3px] w-3.5" : "h-2.5 w-2.5 rounded-[2px]"} style={{ background: item.color }} />
          {item.label}
        </span>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Consumption history
// ---------------------------------------------------------------------------

type Month = { key: string; label: string; value: number | null; observed: number; days: number };

function monthly(points: TimeSeriesPoint[]): Month[] {
  const groups = new Map<string, Month & { sum: number }>();
  points.forEach((point) => {
    const key = (point.date ?? "").slice(0, 7) || String(Math.floor(point.day_index / 30));
    const entry = groups.get(key) ?? { key, label: key, value: null, observed: 0, days: 0, sum: 0 };
    entry.days += 1;
    if (point.consumption_kwh !== null && point.consumption_kwh !== undefined) {
      entry.observed += 1;
      entry.sum += point.consumption_kwh;
    }
    groups.set(key, entry);
  });
  return [...groups.values()].map((m) => ({
    key: m.key,
    label: new Date(`${m.key}-01T00:00:00Z`).toLocaleDateString("en-GB", { month: "short", year: "numeric", timeZone: "UTC" }),
    value: m.observed ? m.sum / m.observed : null,
    observed: m.observed,
    days: m.days,
  }));
}

export function ConsumptionChart({ points, height = 240 }: { points: TimeSeriesPoint[]; height?: number }) {
  const [mode, setMode] = useState<"monthly" | "daily">("monthly");
  const [hover, setHover] = useState<number | null>(null);
  const months = useMemo(() => monthly(points), [points]);
  const daily = useMemo(() => points.map((p) => p.consumption_kwh ?? null), [points]);
  const values = mode === "monthly" ? months.map((m) => m.value) : daily;
  const observed = values.filter((v): v is number => v !== null);
  const sorted = [...observed].sort((a, b) => a - b);
  // Clip the axis at the 99th percentile so one spike does not flatten the chart.
  const top = Math.max(sorted[Math.floor(sorted.length * 0.99)] ?? 1, 0.1) * 1.1;
  const width = 1000;
  const y = (v: number) => height - Math.min(v, top) / top * (height - 12);

  // Missing stretches, as [start, end) index ranges.
  const gaps: Array<[number, number]> = [];
  let start: number | null = null;
  values.forEach((v, i) => {
    if (v === null && start === null) start = i;
    if (v !== null && start !== null) {
      if (i - start >= (mode === "monthly" ? 1 : 7)) gaps.push([start, i]);
      start = null;
    }
  });
  if (start !== null) gaps.push([start, values.length]);

  const step = width / values.length;
  const dailyPath = useMemo(() => {
    if (mode !== "daily") return "";
    let d = "";
    let pen = false;
    daily.forEach((v, i) => {
      if (v === null) {
        pen = false;
        return;
      }
      d += `${pen ? "L" : "M"}${(i + 0.5) * step},${y(v).toFixed(1)}`;
      pen = true;
    });
    return d;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, daily, step, top]);

  const hovered = hover !== null
    ? mode === "monthly"
      ? `${months[hover].label} · ${months[hover].value === null ? "no readings" : `${months[hover].value!.toFixed(2)} kWh/day`} · ${months[hover].observed}/${months[hover].days} days read`
      : `${points[hover].date ?? `day ${hover}`} · ${daily[hover] === null ? "no reading" : `${daily[hover]!.toFixed(2)} kWh`}`
    : null;

  const first = points[0]?.date?.slice(0, 7);
  const last = points[points.length - 1]?.date?.slice(0, 7);

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between gap-4">
        <span className="min-h-5 font-mono text-xs text-ink-2">{hovered ?? `${mode === "monthly" ? "Monthly mean, kWh/day" : "Daily reading, kWh"} · peak axis ${top.toFixed(1)}`}</span>
        <div role="radiogroup" aria-label="Resolution" className="flex gap-1 rounded-lg bg-tint p-0.5">
          {(["monthly", "daily"] as const).map((m) => (
            <button key={m} type="button" role="radio" aria-checked={mode === m} onClick={() => { setMode(m); setHover(null); }}
              className={cn("h-8 rounded-md px-3 text-xs font-medium capitalize", mode === m ? "bg-surface text-ink shadow-sm" : "text-ink-2")}>
              {m}
            </button>
          ))}
        </div>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full" style={{ height }} preserveAspectRatio="none" role="img"
        aria-label={`Consumption history, ${mode}`} onMouseLeave={() => setHover(null)}
        onMouseMove={(event) => {
          const rect = event.currentTarget.getBoundingClientRect();
          const index = Math.floor(((event.clientX - rect.left) / rect.width) * values.length);
          setHover(Math.max(0, Math.min(values.length - 1, index)));
        }}>
        {gaps.map(([a, b]) => (
          <rect key={a} x={a * step} y={0} width={(b - a) * step} height={height} fill="#FBE7DF" />
        ))}
        <line x1={0} y1={height - 0.5} x2={width} y2={height - 0.5} stroke="#D9D6CE" />
        {mode === "monthly"
          ? months.map((m, i) => m.value === null ? null : (
            <rect key={m.key} x={i * step + step * 0.14} width={step * 0.72} y={y(m.value)} height={height - y(m.value)}
              fill={hover === i ? "#1A3496" : COBALT} rx={2} />
          ))
          : <path d={dailyPath} fill="none" stroke={COBALT} strokeWidth={1.2} vectorEffect="non-scaling-stroke" />}
        {hover !== null ? <line x1={(hover + 0.5) * step} x2={(hover + 0.5) * step} y1={0} y2={height} stroke="#14161B" strokeDasharray="3 3" vectorEffect="non-scaling-stroke" /> : null}
      </svg>
      <div className="flex justify-between font-mono text-[11px] text-ink-3">
        <span>{first}</span>
        {gaps.length ? <span className="text-risk-text">shaded: no meter readings</span> : null}
        <span>{last}</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// SHAP waterfall (log-odds)
// ---------------------------------------------------------------------------

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export function ShapWaterfall({ baseValue, probability, reasons, featureCount }: { baseValue: number; probability: number; reasons: PredictionReason[]; featureCount: number }) {
  const output = Math.log(probability / (1 - probability));
  const shown = reasons.reduce((sum, r) => sum + r.shap_value, 0);
  const rows = [
    ...reasons.map((r) => ({ label: featureLabel(r.feature), value: featureValue(r.feature, r.value), shap: r.shap_value, key: r.feature })),
    { label: `${featureCount - reasons.length} other features`, value: "combined", shap: output - baseValue - shown, key: "__other" },
  ];
  let cursor = baseValue;
  const steps = rows.map((row) => {
    const from = cursor;
    cursor += row.shap;
    return { ...row, from, to: cursor };
  });
  const lo = Math.min(baseValue, ...steps.map((s) => Math.min(s.from, s.to)));
  const hi = Math.max(baseValue, ...steps.map((s) => Math.max(s.from, s.to)));
  const span = hi - lo || 1;
  const pos = (v: number) => ((v - lo) / span) * 100;

  return (
    <div className="grid grid-cols-[minmax(0,230px)_minmax(0,1fr)_64px] items-center gap-x-3.5 gap-y-2.5 text-[13px]">
      <span className="text-ink-2">Typical customer (base rate)</span>
      <span className="relative h-5"><span className="absolute top-1/2 h-4 w-[2px] -translate-y-1/2 bg-ink-3" style={{ left: `${pos(baseValue)}%` }} /></span>
      <span className="text-right font-mono tabular">{fmtPct(sigmoid(baseValue), 1)}</span>
      {steps.map((s) => {
        const left = Math.min(pos(s.from), pos(s.to));
        const width = Math.max(Math.abs(pos(s.to) - pos(s.from)), 0.6);
        const up = s.shap >= 0;
        return (
          <div key={s.key} className="contents">
            <span className="flex flex-col gap-0.5">
              <span className="font-medium">{s.label}</span>
              <span className="text-xs text-ink-3">{s.value}</span>
            </span>
            <span className="relative h-5" title={`${s.shap >= 0 ? "+" : ""}${s.shap.toFixed(3)} log-odds`}>
              <span className="absolute top-1/2 h-3.5 -translate-y-1/2 rounded-[3px]" style={{ left: `${left}%`, width: `${width}%`, background: up ? RISK : COBALT }} />
            </span>
            <span className={cn("text-right font-mono tabular", up ? "text-risk-text" : "text-cobalt-ink")}>
              {up ? "+" : "−"}{Math.abs(s.shap).toFixed(2)}
            </span>
          </div>
        );
      })}
      <span className="border-t border-line-soft pt-2.5 font-semibold">Model output</span>
      <span className="border-t border-line-soft pt-2.5 text-xs text-ink-3">log-odds {baseValue.toFixed(2)} → {output >= 0 ? "+" : ""}{output.toFixed(2)}</span>
      <span className="border-t border-line-soft pt-2.5 text-right font-mono font-semibold tabular">{fmtPct(probability, 1)}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Precision / recall trade-off
// ---------------------------------------------------------------------------

export function TradeoffChart({ points, index, onPick, trainedThreshold }: { points: OperatingPoint[]; index: number; onPick: (i: number) => void; trainedThreshold?: number | null }) {
  const W = 640, H = 300, L = 40, R = 620, T = 20, B = 260;
  const x = (i: number) => L + (i / (points.length - 1)) * (R - L);
  const y = (v: number) => B - v * (B - T);
  const line = (key: "precision" | "recall") => points.map((p, i) => `${x(i).toFixed(1)},${y(p[key]).toFixed(1)}`).join(" ");
  const p = points[index];
  const trainedIndex = trainedThreshold == null ? -1 : points.reduce((best, pt, i) => Math.abs(pt.threshold - trainedThreshold) < Math.abs(points[best].threshold - trainedThreshold) ? i : best, 0);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full cursor-crosshair" role="img" aria-label="Precision and recall by threshold"
      onClick={(event) => {
        const rect = event.currentTarget.getBoundingClientRect();
        const svgX = ((event.clientX - rect.left) / rect.width) * W;
        onPick(Math.max(0, Math.min(points.length - 1, Math.round(((svgX - L) / (R - L)) * (points.length - 1)))));
      }}>
      {[1, 0.5].map((v) => <line key={v} x1={L} x2={R} y1={y(v)} y2={y(v)} stroke="#F1EFEA" />)}
      <line x1={L} x2={R} y1={B} y2={B} stroke="#D9D6CE" />
      {[["1.0", 1], ["0.5", 0.5], ["0", 0]].map(([t, v]) => (
        <text key={t} x={L - 8} y={y(v as number) + 4} textAnchor="end" fontSize="11" fill="#6B7280" fontFamily="IBM Plex Mono, monospace">{t}</text>
      ))}
      <text x={L} y={H - 16} fontSize="11" fill="#6B7280" fontFamily="IBM Plex Mono, monospace">τ {points[0].threshold.toFixed(2)}</text>
      <text x={R} y={H - 16} textAnchor="end" fontSize="11" fill="#6B7280" fontFamily="IBM Plex Mono, monospace">{points[points.length - 1].threshold.toFixed(2)}</text>
      {trainedIndex >= 0 ? (
        <g>
          <line x1={x(trainedIndex)} x2={x(trainedIndex)} y1={T} y2={B} stroke="#9AA0AA" strokeDasharray="2 4" />
          <text x={x(trainedIndex) + 4} y={B - 6} fontSize="10" fill="#6B7280" fontFamily="IBM Plex Sans, sans-serif">trained</text>
        </g>
      ) : null}
      <polyline points={line("precision")} fill="none" stroke={COBALT} strokeWidth="2.5" strokeLinejoin="round" />
      <polyline points={line("recall")} fill="none" stroke={AMBER} strokeWidth="2.5" strokeLinejoin="round" />
      <line x1={x(index)} x2={x(index)} y1={T - 6} y2={B} stroke="#14161B" strokeWidth="1.5" strokeDasharray="4 4" />
      <circle cx={x(index)} cy={y(p.precision)} r="6" fill={COBALT} stroke="#fff" strokeWidth="2" />
      <circle cx={x(index)} cy={y(p.recall)} r="6" fill={AMBER} stroke="#fff" strokeWidth="2" />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Probability gauge
// ---------------------------------------------------------------------------

export function Gauge({ value, threshold }: { value: number; threshold: number }) {
  const point = (p: number, r: number) => [120 - r * Math.cos(Math.PI * p), 120 - r * Math.sin(Math.PI * p)];
  const [ex, ey] = point(Math.min(Math.max(value, 0.001), 0.999), 100);
  const [t1x, t1y] = point(threshold, 86);
  const [t2x, t2y] = point(threshold, 114);
  const color = value >= threshold ? RISK : COBALT;
  return (
    <div className="flex flex-col items-center">
      <svg width="240" height="136" viewBox="0 0 240 136" aria-hidden>
        <path d="M20 120 A100 100 0 0 1 220 120" fill="none" stroke="#F1EFEA" strokeWidth="16" strokeLinecap="round" />
        <path d={`M20 120 A100 100 0 0 1 ${ex.toFixed(1)} ${ey.toFixed(1)}`} fill="none" stroke={color} strokeWidth="16" strokeLinecap="round" />
        <line x1={t1x} y1={t1y} x2={t2x} y2={t2y} stroke="#14161B" strokeWidth="2" />
      </svg>
      <span className="-mt-16 font-display text-5xl font-medium leading-none tabular">{fmtNum(value)}</span>
    </div>
  );
}

export function ConfusionGrid({ tp, fp, fn, tn }: { tp: number; fp: number; fn: number; tn: number }) {
  const cell = "flex flex-col gap-0.5 rounded-lg p-4";
  return (
    <div className="grid grid-cols-[88px_1fr_1fr] gap-2 text-[13px]">
      <span />
      <span className="text-center text-ink-3">Actually theft</span>
      <span className="text-center text-ink-3">Actually honest</span>
      <span className="self-center text-ink-3">Flagged</span>
      <div className={cn(cell, "bg-risk-bg text-risk-ink")}><span className="font-display text-3xl tabular">{fmtInt(tp)}</span><span className="text-xs">caught</span></div>
      <div className={cn(cell, "bg-tint")}><span className="font-display text-3xl tabular">{fmtInt(fp)}</span><span className="text-xs text-ink-2">wasted visits</span></div>
      <span className="self-center text-ink-3">Cleared</span>
      <div className={cn(cell, "bg-tint")}><span className="font-display text-3xl tabular">{fmtInt(fn)}</span><span className="text-xs text-ink-2">missed</span></div>
      <div className={cn(cell, "bg-cobalt-bg text-cobalt-ink")}><span className="font-display text-3xl tabular">{fmtInt(tn)}</span><span className="text-xs">correctly cleared</span></div>
    </div>
  );
}
