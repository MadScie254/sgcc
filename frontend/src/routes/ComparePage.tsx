import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Badge, Panel } from "@/components/ui/primitives";
import { apiClient } from "@/lib/api-client";
import type { CompareResponse } from "@/lib/api-types";

type ComparisonRow = {
  model: string;
  recall: number;
  precision: number;
  f1: number;
  auc: number;
};

const MODEL_COLORS: Record<string, string> = {
  xgboost: "#29d39b",
  logistic_regression: "#ffb020",
  random_forest: "#8a6cbe",
  svm: "#ff7354",
};

function metricValue(metrics: Record<string, unknown> | undefined, keys: string[]): number {
  if (!metrics) return 0;
  for (const key of keys) {
    const value = metrics[key];
    if (typeof value === "number") return value;
  }
  return 0;
}

export function ComparePage() {
  const [compare, setCompare] = useState<CompareResponse | null>(null);

  const rows = useMemo<ComparisonRow[]>(() => {
    if (!compare) return [];

    const baselineRows = Object.entries(compare.baselines).map(([model, metrics]) => ({
      model,
      recall: metricValue(metrics as Record<string, unknown>, ["recall"]),
      precision: metricValue(metrics as Record<string, unknown>, ["precision"]),
      f1: metricValue(metrics as Record<string, unknown>, ["f1"]),
      auc: metricValue(metrics as Record<string, unknown>, ["roc_auc", "auc"]),
    }));

    const xgboostRow = {
      model: "xgboost",
      recall: metricValue(compare.xgboost, ["recall"]),
      precision: metricValue(compare.xgboost, ["precision"]),
      f1: metricValue(compare.xgboost, ["f1"]),
      auc: metricValue(compare.xgboost, ["auc", "roc_auc"]),
    };

    return [xgboostRow, ...baselineRows].filter((row) => row.recall > 0 || row.precision > 0 || row.f1 > 0 || row.auc > 0);
  }, [compare]);

  useEffect(() => {
    void apiClient.compareBaselines().then(setCompare).catch(() => setCompare(null));
  }, []);

  const bestRecall = rows.slice().sort((left, right) => right.recall - left.recall)[0];
  const bestF1 = rows.slice().sort((left, right) => right.f1 - left.f1)[0];

  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-3">
        <MetricCard label="Best recall" value={bestRecall?.model ?? "--"} detail={bestRecall ? bestRecall.recall.toFixed(3) : "No data"} tone="signal" />
        <MetricCard label="Best F1" value={bestF1?.model ?? "--"} detail={bestF1 ? bestF1.f1.toFixed(3) : "No data"} tone="signal" />
        <MetricCard label="Models compared" value={String(rows.length || 0)} detail="XGBoost + baselines" tone="muted" />
      </div>

      <Panel className="p-5">
        <Badge tone="signal">Model comparison</Badge>
        <h3 className="mt-3 font-display text-2xl font-semibold">Recall, precision, F1, and AUC</h3>
        <div className="mt-5 h-[420px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={rows} margin={{ top: 20, right: 24, bottom: 12, left: 12 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#233246" />
              <XAxis dataKey="model" stroke="#8a98a8" />
              <YAxis stroke="#8a98a8" domain={[0, 1]} />
              <Tooltip contentStyle={{ background: "#0f1722", border: "1px solid #233246", borderRadius: 16 }} />
              <Bar dataKey="recall" radius={[8, 8, 0, 0]}>
                {rows.map((row) => <Cell key={`${row.model}-recall`} fill={MODEL_COLORS[row.model] ?? "#29d39b"} />)}
              </Bar>
              <Bar dataKey="precision" radius={[8, 8, 0, 0]} fill="#ffb020" />
              <Bar dataKey="f1" radius={[8, 8, 0, 0]} fill="#8a6cbe" />
              <Bar dataKey="auc" radius={[8, 8, 0, 0]} fill="#4a8fd6" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Panel>

      <div className="grid gap-4 xl:grid-cols-[1fr_0.8fr]">
        <Panel className="overflow-hidden p-5">
          <h3 className="font-display text-2xl font-semibold">Per-model scores</h3>
          <div className="mt-4 overflow-auto rounded-2xl border border-gp-border">
            <table className="min-w-full text-left text-sm">
              <thead className="bg-gp-panel-alt text-gp-text-muted">
                <tr>
                  <th className="px-4 py-3">Model</th>
                  <th className="px-4 py-3">Recall</th>
                  <th className="px-4 py-3">Precision</th>
                  <th className="px-4 py-3">F1</th>
                  <th className="px-4 py-3">AUC</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.model} className="border-t border-gp-border">
                    <td className="px-4 py-3 font-medium text-gp-text">{row.model}</td>
                    <td className="px-4 py-3 text-gp-text-muted">{row.recall.toFixed(3)}</td>
                    <td className="px-4 py-3 text-gp-text-muted">{row.precision.toFixed(3)}</td>
                    <td className="px-4 py-3 text-gp-text-muted">{row.f1.toFixed(3)}</td>
                    <td className="px-4 py-3 text-gp-text-muted">{row.auc.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>

        <Panel className="p-5">
          <h3 className="font-display text-2xl font-semibold">Recommendation</h3>
          <div className="mt-4 space-y-3 text-sm text-gp-text-muted">
            <p>Leading recall model: <span className="text-gp-text">{bestRecall?.model ?? "--"}</span></p>
            <p>Leading F1 model: <span className="text-gp-text">{bestF1?.model ?? "--"}</span></p>
            <p>The chart is now driven by live API metrics rather than static JSON output.</p>
          </div>
        </Panel>
      </div>
    </div>
  );
}

function MetricCard({ label, value, detail, tone }: { label: string; value: string; detail: string; tone: "signal" | "muted" }) {
  return (
    <Panel className="p-5">
      <div className="text-xs uppercase tracking-[0.22em] text-gp-text-dim">{label}</div>
      <div className={`mt-3 font-display text-3xl font-semibold ${tone === "signal" ? "text-gp-signal" : "text-gp-text"}`}>{value}</div>
      <div className="mt-2 text-sm text-gp-text-muted">{detail}</div>
    </Panel>
  );
}