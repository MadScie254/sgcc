import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Cell, CartesianGrid, Line, LineChart, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { getModelMetrics } from "@/lib/api";
import { CenteredState } from "@/components/CenteredState";
import { Panel } from "@/components/ui/primitives";

// TODO: replace this mock series with a real logging table once prediction events are persisted.
const mockDailyVolume = Array.from({ length: 14 }).map((_, index) => ({
  day: `D-${13 - index}`,
  predictions: 180 + index * 9 + (index % 3) * 12,
  average_probability: 0.16 + index * 0.004,
}));

export function MonitorPage() {
  const metricsQuery = useQuery({ queryKey: ["model-metrics"], queryFn: getModelMetrics, staleTime: 30_000 });

  const tierData = useMemo(() => {
    const distribution = metricsQuery.data?.risk_tier_distribution ?? { high: 0, medium: 0, low: 0 };
    return [
      { name: "High", value: distribution.high ?? 0, fill: "#C0392B" },
      { name: "Medium", value: distribution.medium ?? 0, fill: "#B7791F" },
      { name: "Low", value: distribution.low ?? 0, fill: "#1E7A5F" },
    ];
  }, [metricsQuery.data]);

  if (metricsQuery.isError) {
    return <CenteredState title="Monitoring unavailable" description="The monitoring snapshot could not be loaded from the backend." />;
  }

  const baseRate = metricsQuery.data?.base_rate ?? 0;
  const currentMean = metricsQuery.data?.current_mean_probability ?? 0;
  const delta = currentMean - baseRate;

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-primary">Monitor</h1>
        <p className="mt-1 text-sm text-secondary">A lightweight operational view of prediction volume, risk mix, and drift.</p>
      </div>

      <div className="grid gap-3 md:grid-cols-3">
        <MetricCard label="Current mean probability" value={currentMean.toFixed(3)} detail="Average predicted probability across the monitored dataset" />
        <MetricCard label="Training base rate" value={baseRate.toFixed(3)} detail="Positive class share in metrics.json" />
        <MetricCard label="Drift delta" value={(delta >= 0 ? "+" : "") + delta.toFixed(3)} detail="Difference between live mean and training base rate" tone={Math.abs(delta) > 0.05 ? "warning" : "success"} />
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.2fr_0.8fr]">
        <Panel className="rounded-lg border border-border bg-surface p-4">
          <div className="text-xs uppercase tracking-[0.18em] text-secondary">Prediction volume over time</div>
          <div className="mt-4 h-[320px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={mockDailyVolume}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E5EA" />
                <XAxis dataKey="day" stroke="#5B6472" />
                <YAxis stroke="#5B6472" />
                <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #E2E5EA" }} />
                <Line type="monotone" dataKey="predictions" stroke="#2B5FAD" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel className="rounded-lg border border-border bg-surface p-4">
          <div className="text-xs uppercase tracking-[0.18em] text-secondary">Risk tier distribution</div>
          <div className="mt-4 h-[320px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #E2E5EA" }} />
                <Pie data={tierData} dataKey="value" nameKey="name" innerRadius={72} outerRadius={110} paddingAngle={4}>
                  {tierData.map((entry) => (
                    <Cell key={entry.name} fill={entry.fill} />
                  ))}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="grid grid-cols-3 gap-2 text-xs">
            {tierData.map((item) => (
              <div key={item.name} className="rounded-md border border-border bg-surface-alt px-3 py-2 text-center">
                <div className="text-muted">{item.name}</div>
                <div className="mt-1 font-mono text-sm text-primary">{item.value}</div>
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </div>
  );
}

function MetricCard({
  label,
  value,
  detail,
  tone = "success",
}: {
  label: string;
  value: string;
  detail: string;
  tone?: "success" | "warning";
}) {
  const valueClass = tone === "warning" ? "text-warning" : "text-success";
  return (
    <Panel className="rounded-lg border border-border bg-surface p-4">
      <div className="text-xs uppercase tracking-[0.18em] text-secondary">{label}</div>
      <div className={`mt-3 text-2xl font-medium ${valueClass}`}>{value}</div>
      <p className="mt-2 text-sm text-secondary">{detail}</p>
    </Panel>
  );
}
