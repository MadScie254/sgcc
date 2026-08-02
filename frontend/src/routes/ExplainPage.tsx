import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Badge, Button, Input, Panel } from "@/components/ui/primitives";
import { apiClient } from "@/lib/api-client";
import type { FeatureImportanceItem, GlobalShapResponse, LocalShapResponse } from "@/lib/api-types";

type BeePoint = {
  x: number;
  y: number;
  feature: string;
  value: number;
  shap: number;
};

const COLOR_SCALE = ["#ff7354", "#d95f63", "#8a6cbe", "#4a8fd6", "#29d39b"];

function toHexColor(value: number, min: number, max: number): string {
  const ratio = max === min ? 0.5 : (value - min) / (max - min);
  const index = Math.min(COLOR_SCALE.length - 1, Math.max(0, Math.floor(ratio * (COLOR_SCALE.length - 1))));
  return COLOR_SCALE[index];
}

function buildBeeSwarm(globalShap: GlobalShapResponse | null): BeePoint[] {
  if (!globalShap) return [];

  const shapMatrix = globalShap.shap_values;
  const featureMatrix = globalShap.feature_values;
  const featureCount = globalShap.feature_names.length;
  const sampleCount = globalShap.sample_count;

  if (!featureCount || !sampleCount) return [];

  const meanAbs = globalShap.feature_names.map((feature, index) => {
    const values = shapMatrix.map((row) => Math.abs(row[index] ?? 0));
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    return { feature, index, mean };
  });

  const topFeatures = meanAbs.sort((left, right) => right.mean - left.mean).slice(0, 10);
  const width = 1;
  const radius = 0.03;
  const points: BeePoint[] = [];

  topFeatures.forEach((entry, rowIndex) => {
    const rowPoints: Array<{ x: number; y: number; feature: string; value: number; shap: number }> = [];
    const values = shapMatrix.map((row, sampleIndex) => ({
      shap: row[entry.index] ?? 0,
      value: featureMatrix[sampleIndex]?.[entry.index] ?? 0,
      feature: entry.feature,
    }));

    values.sort((left, right) => left.shap - right.shap).forEach((point) => {
      const x = point.shap;
      let y = rowIndex;

      for (let step = 0; step < 40; step += 1) {
        const direction = step % 2 === 0 ? 1 : -1;
        const distance = Math.ceil(step / 2) * radius * 2;
        const candidateY = rowIndex + direction * distance;
        const collides = rowPoints.some((other) => Math.hypot(other.x - x, other.y - candidateY) < radius * 2.2);
        if (!collides) {
          y = candidateY;
          break;
        }
      }

      rowPoints.push({ x, y, feature: point.feature, value: point.value, shap: point.shap });
    });

    points.push(...rowPoints);
  });

  return points;
}

function buildWaterfall(localShap: LocalShapResponse | null) {
  if (!localShap) return [];
  return localShap.top_reasons
    .slice()
    .sort((left, right) => Math.abs(right.shap_value) - Math.abs(left.shap_value))
    .map((reason) => ({
      feature: reason.feature,
      impact: reason.shap_value,
    }));
}

export function ExplainPage() {
  const [featureImportance, setFeatureImportance] = useState<FeatureImportanceItem[]>([]);
  const [globalShap, setGlobalShap] = useState<GlobalShapResponse | null>(null);
  const [localShap, setLocalShap] = useState<LocalShapResponse | null>(null);
  const [customerId, setCustomerId] = useState("");
  const [loadingLocal, setLoadingLocal] = useState(false);

  useEffect(() => {
    void apiClient.featureImportance().then(setFeatureImportance).catch(() => setFeatureImportance([]));
    void apiClient.globalShap().then(setGlobalShap).catch(() => setGlobalShap(null));
  }, []);

  async function loadLocalShap() {
    if (!customerId.trim()) return;
    setLoadingLocal(true);
    try {
      const response = await apiClient.localShap(customerId.trim());
      setLocalShap(response);
    } catch {
      setLocalShap(null);
    } finally {
      setLoadingLocal(false);
    }
  }

  const swarmPoints = useMemo(() => buildBeeSwarm(globalShap), [globalShap]);
  const waterfallData = useMemo(() => buildWaterfall(localShap), [localShap]);
  const localShapExtent = useMemo(() => {
    const values = localShap?.shap_values ?? [];
    if (!values.length) return { min: -1, max: 1 };
    return {
      min: Math.min(...values),
      max: Math.max(...values),
    };
  }, [localShap]);

  return (
    <div className="space-y-4">
      <div className="grid gap-4 xl:grid-cols-[0.95fr_1.05fr]">
        <Panel className="p-5">
          <Badge tone="signal">Feature importance</Badge>
          <h3 className="mt-3 font-display text-2xl font-semibold">Model-wide ranking</h3>
          <div className="mt-5 h-[420px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={featureImportance.slice().sort((left, right) => left.importance - right.importance)} layout="vertical" margin={{ top: 8, right: 20, bottom: 8, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#233246" />
                <XAxis type="number" stroke="#8a98a8" tickFormatter={(value) => Number(value).toFixed(3)} />
                <YAxis type="category" dataKey="feature" stroke="#8a98a8" width={160} />
                <Tooltip
                  contentStyle={{ background: "#0f1722", border: "1px solid #233246", borderRadius: 16 }}
                  formatter={(value: number) => Number(value).toFixed(4)}
                />
                <Bar dataKey="importance" radius={[0, 10, 10, 0]} fill="#29d39b">
                  <LabelList dataKey="importance" position="right" formatter={(value: number) => Number(value).toFixed(3)} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel className="p-5">
          <Badge tone="signal">Global SHAP</Badge>
          <h3 className="mt-3 font-display text-2xl font-semibold">Beeswarm overview</h3>
          <p className="mt-2 text-sm text-gp-text-muted">Each dot is a sample-feature SHAP contribution. The cloud is laid out with a custom collision pass so the chart stays readable without the broken package dependency.</p>
          <div className="mt-5 overflow-hidden rounded-3xl border border-gp-border bg-gp-panel-alt/60 p-4">
            <svg viewBox="0 0 1200 520" className="h-[420px] w-full">
              <defs>
                <linearGradient id="swarmGradient" x1="0" x2="1" y1="0" y2="0">
                  <stop offset="0%" stopColor="#ff7354" />
                  <stop offset="50%" stopColor="#8a6cbe" />
                  <stop offset="100%" stopColor="#29d39b" />
                </linearGradient>
              </defs>
              {[...Array(10)].map((_, index) => {
                const y = 40 + index * 42;
                return <line key={index} x1="220" x2="1140" y1={y} y2={y} stroke="#233246" strokeDasharray="6 6" />;
              })}
              <line x1="620" x2="620" y1="24" y2="500" stroke="#8a98a8" strokeWidth="2" strokeDasharray="8 8" />
              {swarmPoints.map((point, index) => {
                const x = 220 + ((point.x + 1.0) / 2.0) * 920;
                const y = 44 + point.y * 42;
                return <circle key={`${point.feature}-${index}`} cx={x} cy={y} r="6" fill={toHexColor(point.value, -1, 1)} fillOpacity="0.92" stroke="rgba(8,16,24,0.9)" strokeWidth="1.5" />;
              })}
              {[...Array(10)].map((_, index) => (
                <text key={index} x="18" y={46 + index * 42} fill="#8a98a8" fontSize="12" fontFamily="IBM Plex Mono, monospace">
                  {globalShap?.feature_names[index] ?? ""}
                </text>
              ))}
              <text x="220" y="22" fill="#536274" fontSize="12">low SHAP</text>
              <text x="1080" y="22" fill="#536274" fontSize="12">high SHAP</text>
            </svg>
          </div>
          <div className="mt-4 grid grid-cols-3 gap-3 text-xs text-gp-text-muted">
            <MetricChip label="Samples" value={String(globalShap?.sample_count ?? 0)} />
            <MetricChip label="Points" value={String(swarmPoints.length)} />
            <MetricChip label="Features" value={String(globalShap?.feature_names.length ?? 0)} />
          </div>
        </Panel>
      </div>

      <div className="grid gap-4 xl:grid-cols-[0.75fr_1.25fr]">
        <Panel className="p-5">
          <Badge tone="signal">Local SHAP</Badge>
          <h3 className="mt-3 font-display text-2xl font-semibold">Customer-level explanation</h3>
          <div className="mt-4 space-y-4">
            <Input value={customerId} onChange={(event) => setCustomerId(event.target.value)} placeholder="customer_id" />
            <Button type="button" onClick={() => void loadLocalShap()} disabled={loadingLocal}>
              {loadingLocal ? "Loading..." : "Load local SHAP"}
            </Button>
            <div className="rounded-2xl border border-gp-border bg-gp-signal-dim p-4">
              <div className="text-xs uppercase tracking-[0.18em] text-gp-text-dim">Prediction probability</div>
              <div className="mt-2 font-data text-3xl font-semibold text-gp-signal">{localShap ? localShap.probability.toFixed(3) : "--"}</div>
            </div>
          </div>
        </Panel>

        <Panel className="p-5">
          <Badge tone="alert">Waterfall</Badge>
          <h3 className="mt-3 font-display text-2xl font-semibold">Top local drivers</h3>
          <div className="mt-5 h-[360px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={waterfallData} margin={{ top: 8, right: 20, bottom: 8, left: 24 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#233246" />
                <XAxis dataKey="feature" stroke="#8a98a8" interval={0} angle={-20} textAnchor="end" height={90} />
                <YAxis stroke="#8a98a8" domain={[localShapExtent.min * 1.2, localShapExtent.max * 1.2]} />
                <Tooltip
                  contentStyle={{ background: "#0f1722", border: "1px solid #233246", borderRadius: 16 }}
                  formatter={(value: number) => Number(value).toFixed(4)}
                />
                <Bar dataKey="impact" radius={[8, 8, 0, 0]}>
                  {waterfallData.map((entry, index) => (
                    <Cell key={entry.feature} fill={entry.impact >= 0 ? "#29d39b" : "#ff7354"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-4 grid gap-3 md:grid-cols-2">
            {(localShap?.top_reasons ?? []).map((reason) => (
              <div key={reason.feature} className="rounded-2xl border border-gp-border bg-gp-panel-alt px-4 py-3 text-sm">
                <div className="font-medium text-gp-text">{reason.feature}</div>
                <div className="mt-1 font-data text-gp-text-muted">{reason.shap_value.toFixed(4)}</div>
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </div>
  );
}

function MetricChip({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-gp-border bg-gp-panel-alt px-4 py-3">
      <div className="text-[10px] uppercase tracking-[0.18em] text-gp-text-dim">{label}</div>
      <div className="mt-1 font-data text-lg text-gp-text">{value}</div>
    </div>
  );
}