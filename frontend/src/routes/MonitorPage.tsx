import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Badge, Panel } from "@/components/ui/primitives";
import { apiClient } from "@/lib/api-client";
import type { MonitorResponse } from "@/lib/api-types";

type DriftRow = {
  feature: string;
  value: number;
};

const ALERT_TONES: Record<string, string> = {
  high: "border-gp-alert/40 bg-gp-alert-dim text-gp-alert",
  medium: "border-amber-400/40 bg-amber-950/40 text-amber-300",
  low: "border-gp-border bg-gp-panel-alt text-gp-text-muted",
};

export function MonitorPage() {
  const [monitor, setMonitor] = useState<MonitorResponse | null>(null);

  useEffect(() => {
    void apiClient.monitorDrift().then(setMonitor).catch(() => setMonitor(null));
  }, []);

  const ksRows = useMemo<DriftRow[]>(() => {
    const items = monitor?.data_drift?.ks_test?.top_5_drifted as Array<{ feature: string; p_value: number; ks_statistic: number }> | undefined;
    return (items ?? []).map((item) => ({ feature: item.feature, value: item.ks_statistic }));
  }, [monitor]);

  const psiRows = useMemo<DriftRow[]>(() => {
    const items = monitor?.data_drift?.psi?.top_5_psi as Array<{ feature: string; psi: number }> | undefined;
    return (items ?? []).map((item) => ({ feature: item.feature, value: item.psi }));
  }, [monitor]);

  const driftPct = Number(monitor?.data_drift?.ks_test?.drift_pct ?? 0);
  const significantPsi = Number(monitor?.data_drift?.psi?.n_significant ?? 0);
  const recallDrop = Number(monitor?.concept_drift?.recall_drop ?? 0);

  const alerts = (monitor?.alerts ?? []) as Array<{ severity?: string; type?: string; message?: string }>;

  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-3">
        <MetricCard label="KS drift rate" value={`${driftPct.toFixed(1)}%`} detail="Percent of features failing the KS test" tone={driftPct >= 30 ? "alert" : driftPct >= 15 ? "medium" : "muted"} />
        <MetricCard label="Significant PSI" value={String(significantPsi)} detail="Features with PSI >= 0.25" tone={significantPsi > 0 ? "alert" : "muted"} />
        <MetricCard label="Recall drop" value={recallDrop.toFixed(3)} detail="Concept drift signal from the reference set" tone={recallDrop > 0.1 ? "alert" : "muted"} />
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel className="p-5">
          <Badge tone="signal">Data drift</Badge>
          <h3 className="mt-3 font-display text-2xl font-semibold">KS test leaders</h3>
          <div className="mt-5 h-[360px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={ksRows} layout="vertical" margin={{ top: 12, right: 24, bottom: 12, left: 12 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#233246" />
                <XAxis type="number" stroke="#8a98a8" />
                <YAxis type="category" dataKey="feature" stroke="#8a98a8" width={150} />
                <Tooltip contentStyle={{ background: "#0f1722", border: "1px solid #233246", borderRadius: 16 }} />
                <Bar dataKey="value" radius={[0, 10, 10, 0]} fill="#ff7354">
                  {ksRows.map((row) => <Cell key={row.feature} fill="#ff7354" />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-3 text-xs text-gp-text-muted">Top features by KS statistic from the backend drift report.</div>
        </Panel>

        <Panel className="p-5">
          <Badge tone="signal">PSI</Badge>
          <h3 className="mt-3 font-display text-2xl font-semibold">Population stability index</h3>
          <div className="mt-5 h-[360px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={psiRows} layout="vertical" margin={{ top: 12, right: 24, bottom: 12, left: 12 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#233246" />
                <XAxis type="number" stroke="#8a98a8" />
                <YAxis type="category" dataKey="feature" stroke="#8a98a8" width={150} />
                <Tooltip contentStyle={{ background: "#0f1722", border: "1px solid #233246", borderRadius: 16 }} />
                <Bar dataKey="value" radius={[0, 10, 10, 0]} fill="#29d39b">
                  {psiRows.map((row) => <Cell key={row.feature} fill="#29d39b" />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-3 text-xs text-gp-text-muted">Top features by PSI score from the backend drift report.</div>
        </Panel>
      </div>

      <div className="grid gap-4 xl:grid-cols-[0.8fr_1.2fr]">
        <Panel className="p-5">
          <h3 className="font-display text-2xl font-semibold">Alerts</h3>
          <div className="mt-4 space-y-3">
            {alerts.length > 0 ? alerts.map((alert, index) => (
              <div key={`${alert.type ?? "alert"}-${index}`} className={`rounded-2xl border px-4 py-3 text-sm ${alertTone(alert.severity)}`}>
                <div className="font-medium">{alert.type ?? "Alert"}</div>
                <div className="mt-1 text-current/80">{alert.message ?? "No message"}</div>
              </div>
            )) : <p className="text-sm text-gp-text-muted">No active alerts in the latest monitoring snapshot.</p>}
          </div>
        </Panel>

        <Panel className="p-5">
          <h3 className="font-display text-2xl font-semibold">Recommendations</h3>
          <div className="mt-4 space-y-3 text-sm text-gp-text-muted">
            {(monitor?.recommendations ?? []).map((item, index) => (
              <div key={`${index}-${item}`} className="rounded-2xl border border-gp-border bg-gp-panel-alt px-4 py-3 text-gp-text-muted">{item}</div>
            ))}
            {(monitor?.recommendations ?? []).length === 0 && <p>No recommendations returned by the backend.</p>}
          </div>
        </Panel>
      </div>
    </div>
  );
}

function alertTone(severity?: string): string {
  const key = (severity ?? "low").toLowerCase();
  if (key.includes("high")) return ALERT_TONES.high;
  if (key.includes("medium")) return ALERT_TONES.medium;
  return ALERT_TONES.low;
}

function MetricCard({ label, value, detail, tone }: { label: string; value: string; detail: string; tone: "alert" | "medium" | "muted" }) {
  return (
    <Panel className="p-5">
      <div className="text-xs uppercase tracking-[0.22em] text-gp-text-dim">{label}</div>
      <div className={`mt-3 font-display text-3xl font-semibold ${tone === "alert" ? "text-gp-alert" : tone === "medium" ? "text-amber-300" : "text-gp-signal"}`}>{value}</div>
      <div className="mt-2 text-sm text-gp-text-muted">{detail}</div>
    </Panel>
  );
}