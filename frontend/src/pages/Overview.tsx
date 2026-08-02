import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { getCustomerShap, getModelMetrics, listCustomers } from "@/lib/api";
import { formatMaybeNumber, riskTierClasses, riskTierLabel } from "@/lib/dashboard";
import { CenteredState } from "@/components/CenteredState";
import { Panel } from "@/components/ui/primitives";
import { ShapDrivers } from "@/components/ShapDrivers";

export function OverviewPage() {
  const navigate = useNavigate();
  const metricsQuery = useQuery({ queryKey: ["model-metrics"], queryFn: getModelMetrics, staleTime: 30_000 });
  const customersQuery = useQuery({
    queryKey: ["overview-customers"],
    queryFn: () => listCustomers({ page: 1, page_size: 8, sort_by: "risk_score", sort_dir: "desc" }),
    staleTime: 30_000,
  });
  const [selectedCustomerId, setSelectedCustomerId] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedCustomerId && customersQuery.data?.items[0]?.customer_id) {
      setSelectedCustomerId(customersQuery.data.items[0].customer_id);
    }
  }, [customersQuery.data, selectedCustomerId]);

  const shapQuery = useQuery({
    queryKey: ["overview-shap", selectedCustomerId],
    queryFn: () => getCustomerShap(selectedCustomerId ?? ""),
    enabled: Boolean(selectedCustomerId),
    staleTime: 30_000,
  });

  const kpis = [
    { label: "Recall", value: metricsQuery.data ? formatMaybeNumber(metricsQuery.data.metrics.recall, 3) : "—" },
    { label: "Precision", value: metricsQuery.data ? formatMaybeNumber(metricsQuery.data.metrics.precision, 3) : "—" },
    { label: "Flagged today", value: metricsQuery.data ? String(metricsQuery.data.flagged_today) : "—" },
    { label: "Customers monitored", value: metricsQuery.data ? String(metricsQuery.data.customers_monitored) : "—" },
  ];

  if (metricsQuery.isError || customersQuery.isError) {
    return <CenteredState title="Model not ready" description="The dashboard could not load model data. Check that the backend is running and the trained artifacts are present." />;
  }

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-primary">Overview</h1>
        <p className="mt-1 text-sm text-secondary">A concise risk command center for the current model snapshot.</p>
      </div>

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        {kpis.map((item) => (
          <Panel key={item.label} className="rounded-lg border border-border bg-surface p-4">
            <div className="text-xs uppercase tracking-[0.18em] text-secondary">{item.label}</div>
            <div className="mt-3 text-2xl font-medium text-primary">{item.value}</div>
          </Panel>
        ))}
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.3fr_1fr]">
        <Panel className="rounded-lg border border-border bg-surface p-0">
          <div className="flex items-center justify-between border-b border-border px-4 py-3">
            <div>
              <div className="text-xs uppercase tracking-[0.18em] text-secondary">High-risk customers</div>
              <div className="mt-1 text-sm text-secondary">Sorted by predicted risk score</div>
            </div>
            <button type="button" onClick={() => navigate("/customers")} className="text-sm font-medium text-accent hover:underline">
              View all
            </button>
          </div>

          <div className="overflow-hidden">
            <table className="min-w-full border-collapse text-sm">
              <thead className="bg-surface-alt text-xs uppercase tracking-[0.18em] text-secondary">
                <tr>
                  <th className="px-4 py-3 text-left font-medium">Customer</th>
                  <th className="px-4 py-3 text-left font-medium">Tier</th>
                  <th className="px-4 py-3 text-right font-medium">Score</th>
                </tr>
              </thead>
              <tbody>
                {customersQuery.isLoading
                  ? Array.from({ length: 8 }).map((_, index) => (
                      <tr key={index} className="border-t border-border">
                        <td colSpan={3} className="px-4 py-4">
                          <div className="h-4 w-full animate-pulse rounded-md bg-surface-alt" />
                        </td>
                      </tr>
                    ))
                  : customersQuery.data?.items.map((item) => (
                      <tr
                        key={item.customer_id}
                        onClick={() => navigate(`/predict?customer=${encodeURIComponent(item.customer_id)}`)}
                        className="cursor-pointer border-t border-border transition hover:bg-surface-alt"
                      >
                        <td className="px-4 py-3 font-mono text-xs text-primary">{item.customer_id}</td>
                        <td className="px-4 py-3">
                          <span className={`inline-flex rounded-full border px-2.5 py-1 text-xs font-semibold ${riskTierClasses(item.risk_tier)}`}>
                            {riskTierLabel(item.risk_tier)}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-right font-mono text-sm text-primary">{item.risk_score.toFixed(4)}</td>
                      </tr>
                    ))}
              </tbody>
            </table>
          </div>
        </Panel>

        <Panel className="rounded-lg border border-border bg-surface p-4">
          <div className="flex items-start justify-between gap-4">
            <div>
              <div className="text-xs uppercase tracking-[0.18em] text-secondary">Top SHAP drivers</div>
              <div className="mt-1 text-sm text-secondary">Selected customer {selectedCustomerId ? <span className="font-mono text-primary">{selectedCustomerId}</span> : "loading..."}</div>
            </div>
          </div>

          <div className="mt-4">
            {shapQuery.isLoading ? (
              <div className="space-y-3">
                {Array.from({ length: 4 }).map((_, index) => (
                  <div key={index} className="space-y-2 rounded-md border border-border bg-surface p-3">
                    <div className="h-4 w-1/2 animate-pulse rounded-md bg-surface-alt" />
                    <div className="h-1.5 w-full animate-pulse rounded-full bg-surface-alt" />
                    <div className="h-3 w-1/3 animate-pulse rounded-md bg-surface-alt" />
                  </div>
                ))}
              </div>
            ) : (
              <ShapDrivers items={shapQuery.data?.top_reasons ?? []} emptyLabel="Pick a customer from the table to inspect local SHAP values." />
            )}
          </div>
        </Panel>
      </div>
    </div>
  );
}
