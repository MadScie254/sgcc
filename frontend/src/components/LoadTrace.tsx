type LoadTraceProps = {
  data: number[];
  anomalyIndex?: number;
};

export function LoadTrace({ data, anomalyIndex }: LoadTraceProps) {
  const width = 960;
  const height = 220;
  const padding = 20;

  if (data.length < 2) {
    return (
      <div className="flex h-[220px] items-center justify-center rounded-3xl border border-dashed border-gp-border bg-gp-panel-alt/50 text-sm text-gp-text-muted">
        Load trace waiting for live data.
      </div>
    );
  }

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const stepX = (width - padding * 2) / (data.length - 1);
  const points = data
    .map((value, index) => {
      const x = padding + index * stepX;
      const y = padding + (1 - (value - min) / range) * (height - padding * 2);
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-[220px] w-full overflow-visible rounded-3xl border border-gp-border bg-[linear-gradient(180deg,rgba(19,26,34,0.8),rgba(8,16,24,0.95))] p-3">
      <defs>
        <linearGradient id="traceFill" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="rgba(41, 211, 155, 0.34)" />
          <stop offset="100%" stopColor="rgba(41, 211, 155, 0.02)" />
        </linearGradient>
      </defs>
      <polyline fill="none" stroke="#29d39b" strokeWidth="3" strokeLinejoin="round" strokeLinecap="round" points={points} />
      <polygon fill="url(#traceFill)" points={`${padding},${height - padding} ${points} ${width - padding},${height - padding}`} opacity="0.35" />
      {typeof anomalyIndex === "number" && anomalyIndex >= 0 && anomalyIndex < data.length ? (
        <circle
          cx={padding + anomalyIndex * stepX}
          cy={padding + (1 - (data[anomalyIndex] - min) / range) * (height - padding * 2)}
          r="7"
          fill="#ff7354"
          stroke="rgba(255,115,84,0.28)"
          strokeWidth="10"
        />
      ) : null}
    </svg>
  );
}