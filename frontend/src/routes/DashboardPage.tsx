import { useEffect, useState } from "react";
import { LoadTrace } from "@/components/LoadTrace";
import { Badge, Panel } from "@/components/ui/primitives";
import { apiClient } from "@/lib/api-client";
import type { DatasetSummary, ThresholdPreviewResponse } from "@/lib/api-types";

export function DashboardPage() {
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [preview, setPreview] = useState<ThresholdPreviewResponse | null>(null);

  useEffect(() => {
    void apiClient.datasetSummary().then(setSummary).catch(() => setSummary(null));
    void apiClient.thresholdPreview(0.5).then(setPreview).catch(() => setPreview(null));
  }, []);

  const trace = [46, 49, 51, 48, 45, 42, 44, 47, 53, 50, 61, 58, 55, 62, 69, 66, 63, 71, 68, 64];

  return (
    <div className="space-y-6">
      <div className="grid gap-4 xl:grid-cols-[1.4fr_0.9fr]">
        <Panel className="p-5 lg:p-6">
          <div className="flex items-start justify-between gap-4">
            <div>
              <Badge tone="signal">Live trace</Badge>
              <h3 className="mt-3 font-display text-3xl font-semibold tracking-tight">Consumption trace with anomaly emphasis</h3>
              <p className="mt-2 max-w-2xl text-sm leading-6 text-gp-text-muted">
                This dashboard is wired to the FastAPI backend and is ready to surface real customer traces, threshold changes, and model health.
              </p>
            </div>
            <div className="rounded-2xl border border-gp-signal/20 bg-gp-signal-dim px-4 py-3 text-right">
              <div className="font-data text-2xl font-semibold text-gp-signal">{summary?.total_customers ?? "--"}</div>
              <div className="mt-1 text-xs uppercase tracking-[0.2em] text-gp-text-dim">Customers</div>
            </div>
          </div>
          <div className="mt-6">
            <LoadTrace data={trace} anomalyIndex={15} />
          </div>
        </Panel>

        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-1">
          <Panel className="p-5">
            <p className="text-xs uppercase tracking-[0.22em] text-gp-text-dim">Dataset</p>
            <div className="mt-3 font-data text-2xl font-semibold">{summary?.total_rows ?? "--"}</div>
            <p className="mt-2 text-sm text-gp-text-muted">Total consumption rows in the current training set.</p>
          </Panel>
          <Panel className="p-5">
            <p className="text-xs uppercase tracking-[0.22em] text-gp-text-dim">Threshold preview</p>
            <div className="mt-3 font-data text-2xl font-semibold text-gp-alert">{preview?.metrics.recall?.toFixed(3) ?? "--"}</div>
            <p className="mt-2 text-sm text-gp-text-muted">Recall at threshold 0.5 from the backend preview endpoint.</p>
          </Panel>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        {[
          ["Detection", "Prediction, thresholding, SHAP reasons"],
          ["Explainability", "Global and local model explanations"],
          ["Operations", "Drift, alerts, and retraining loops"],
        ].map(([title, body]) => (
          <Panel key={title} className="p-5">
            <p className="font-display text-xl font-semibold">{title}</p>
            <p className="mt-2 text-sm leading-6 text-gp-text-muted">{body}</p>
          </Panel>
        ))}
      </div>
    </div>
  );
}