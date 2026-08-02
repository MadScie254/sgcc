import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis, LineChart, Line as ReLine } from "recharts";
import { useQuery } from "@tanstack/react-query";
import { getCustomerShap, getCustomerTimeseries, getFeatureImportance, getModelConfig } from "@/lib/api";
import { CenteredState } from "@/components/CenteredState";
import { CustomerPicker } from "@/components/CustomerPicker";
import { Panel } from "@/components/ui/primitives";
import { ShapDrivers } from "@/components/ShapDrivers";

export function ExplainPage() {
  const [customerId, setCustomerId] = useState<string | null>(null);
  const importanceQuery = useQuery({ queryKey: ["feature-importance"], queryFn: () => getFeatureImportance(15), staleTime: 30_000 });
  const configQuery = useQuery({ queryKey: ["model-config"], queryFn: getModelConfig, staleTime: 30_000 });
  const shapQuery = useQuery({ queryKey: ["explain-shap", customerId], queryFn: () => getCustomerShap(customerId ?? ""), enabled: Boolean(customerId), staleTime: 30_000 });
  const seriesQuery = useQuery({ queryKey: ["customer-timeseries", customerId], queryFn: () => getCustomerTimeseries(customerId ?? ""), enabled: Boolean(customerId), staleTime: 30_000 });

  const sparklineData = useMemo(() => (seriesQuery.data?.points ?? []).map((point) => ({ day_index: point.day_index, consumption_kwh: point.consumption_kwh ?? 0 })), [seriesQuery.data]);

  if (importanceQuery.isError || configQuery.isError) {
    return <CenteredState title="Explainability unavailable" description="The explainability view could not load model artifacts or config data." />;
  }

  const topImportance = [...(importanceQuery.data ?? [])].slice().sort((left, right) => right.importance - left.importance);

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-primary">Explain</h1>
        <p className="mt-1 text-sm text-secondary">Global feature importance, local SHAP, and the model configuration behind the current build.</p>
      </div>

      <Panel className="rounded-lg border border-border bg-surface p-4">
        <div className="text-xs uppercase tracking-[0.18em] text-secondary">Global feature importance</div>
        <div className="mt-4 h-[360px]">
          {importanceQuery.isLoading ? (
            <div className="grid h-full place-items-center rounded-md border border-border bg-surface-alt text-sm text-secondary">Loading chart...</div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={topImportance} layout="vertical" margin={{ top: 8, right: 24, bottom: 8, left: 16 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E5EA" />
                <XAxis type="number" stroke="#5B6472" tickFormatter={(value) => Number(value).toFixed(3)} />
                <YAxis type="category" dataKey="feature" stroke="#5B6472" width={180} />
                <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #E2E5EA" }} formatter={(value: number) => value.toFixed(4)} />
                <Bar dataKey="importance" fill="#2B5FAD" radius={[0, 8, 8, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </Panel>

      <div className="grid gap-4 xl:grid-cols-[0.85fr_1.15fr]">
        <Panel className="rounded-lg border border-border bg-surface p-4">
          <div className="text-xs uppercase tracking-[0.18em] text-secondary">Customer picker</div>
          <div className="mt-4">
            <CustomerPicker selectedCustomerId={customerId} onSelect={setCustomerId} />
          </div>

          <div className="mt-4 rounded-md border border-border bg-surface-alt px-4 py-3 text-sm text-secondary">
            {customerId ? <span className="font-mono text-primary">{customerId}</span> : "Select a customer to inspect local SHAP and a consumption sparkline."}
          </div>

          <div className="mt-4">
            <ShapDrivers items={shapQuery.data?.top_reasons ?? []} emptyLabel="Pick a customer to load local drivers." />
          </div>
        </Panel>

        <Panel className="rounded-lg border border-border bg-surface p-4">
          <div className="text-xs uppercase tracking-[0.18em] text-secondary">Consumption sparkline</div>
          <div className="mt-4 h-48 rounded-md border border-border bg-surface p-3">
            {sparklineData.length > 1 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={sparklineData}>
                  <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #E2E5EA" }} />
                  <ReLine type="monotone" dataKey="consumption_kwh" stroke="#2B5FAD" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <CenteredState title="No series yet" description="Select a customer to render the time-series sparkline." />
            )}
          </div>

          <div className="mt-4 rounded-md border border-border bg-surface p-4">
            <div className="text-xs uppercase tracking-[0.18em] text-secondary">Model info</div>
            <dl className="mt-4 space-y-3 text-sm">
              <InfoRow label="Architecture" value="XGBoost" />
              <InfoRow label="Composite score" value="0.6R + 0.25P + 0.15F1" />
              {Object.entries({ ...(configQuery.data?.model ?? {}), ...(configQuery.data?.preprocessing ?? {}) })
                .slice(0, 10)
                .map(([key, value]) => (
                  <InfoRow key={key} label={key} value={formatValue(value)} />
                ))}
            </dl>
          </div>
        </Panel>
      </div>

      {shapQuery.isError || seriesQuery.isError ? (
        <CenteredState title="Customer details unavailable" description="The selected customer could not be loaded. Check the ID or verify that the model artifacts are present." />
      ) : null}
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-border pb-2 last:border-b-0 last:pb-0">
      <dt className="text-secondary">{label}</dt>
      <dd className="max-w-[60%] text-right font-medium text-primary">{value}</dd>
    </div>
  );
}

function formatValue(value: unknown): string {
  if (Array.isArray(value)) {
    return value.map((entry) => String(entry)).join(", ");
  }

  if (typeof value === "object" && value !== null) {
    return Object.entries(value as Record<string, unknown>)
      .slice(0, 3)
      .map(([key, entry]) => `${key}: ${String(entry)}`)
      .join("; ");
  }

  return String(value);
}
