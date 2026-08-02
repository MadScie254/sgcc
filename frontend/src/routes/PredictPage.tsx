import { useState } from "react";
import { Badge, Button, Input, Panel, TextArea } from "@/components/ui/primitives";
import { apiClient } from "@/lib/api-client";
import type { SinglePredictionResponse, ThresholdPreviewResponse } from "@/lib/api-types";
import { useAppStore } from "@/store/useAppStore";

export function PredictPage() {
  const [customerId, setCustomerId] = useState("");
  const [featureJson, setFeatureJson] = useState('{"mean": 120.4, "std": 18.2}');
  const [result, setResult] = useState<SinglePredictionResponse | null>(null);
  const [preview, setPreview] = useState<ThresholdPreviewResponse | null>(null);
  const threshold = useAppStore((state) => state.threshold);
  const setThreshold = useAppStore((state) => state.setThreshold);

  async function handlePredict() {
    const parsed = featureJson.trim() ? JSON.parse(featureJson) : null;
    const payload = customerId.trim()
      ? { customer_id: customerId.trim(), threshold }
      : { features: parsed, threshold };
    const response = await apiClient.predictSingle(payload);
    setResult(response);
    setPreview(await apiClient.thresholdPreview(threshold));
  }

  return (
    <div className="grid gap-4 xl:grid-cols-[1fr_0.9fr]">
      <Panel className="p-5">
        <Badge tone="signal">Single prediction</Badge>
        <h3 className="mt-3 font-display text-3xl font-semibold">Risk scoring and explanation</h3>
        <div className="mt-5 space-y-4">
          <Input value={customerId} onChange={(event) => setCustomerId(event.target.value)} placeholder="customer_id (optional)" />
          <TextArea value={featureJson} onChange={(event) => setFeatureJson(event.target.value)} rows={7} placeholder="Feature JSON" />
          <label className="block text-sm text-gp-text-muted">
            Threshold: {threshold.toFixed(2)}
            <input
              className="mt-2 w-full accent-[color:var(--color-gp-signal)]"
              type="range"
              min={0.1}
              max={0.9}
              step={0.05}
              value={threshold}
              onChange={(event) => setThreshold(Number(event.target.value))}
            />
          </label>
          <Button type="button" onClick={() => void handlePredict()}>Predict</Button>
        </div>
      </Panel>

      <div className="grid gap-4">
        <Panel className="p-5">
          <p className="text-xs uppercase tracking-[0.22em] text-gp-text-dim">Result</p>
          <div className="mt-3 font-data text-4xl font-semibold text-gp-signal">{result ? result.probability.toFixed(3) : "--"}</div>
          <p className="mt-2 text-sm text-gp-text-muted">Probability of theft at the selected threshold.</p>
          <p className="mt-4 text-sm text-gp-text-muted">Prediction: {result ? (result.prediction ? "Theft" : "Honest") : "--"}</p>
        </Panel>
        <Panel className="p-5">
          <p className="text-xs uppercase tracking-[0.22em] text-gp-text-dim">Top reasons</p>
          <div className="mt-4 space-y-3 text-sm">
            {(result?.top_reasons ?? []).map((reason) => (
              <div key={reason.feature} className="flex items-center justify-between rounded-2xl border border-gp-border bg-gp-panel-alt px-4 py-3">
                <span>{reason.feature}</span>
                <span className="font-data text-gp-text-muted">{reason.shap_value.toFixed(3)}</span>
              </div>
            ))}
            {(!result || result.top_reasons.length === 0) && <p className="text-gp-text-muted">Run a prediction to see SHAP reasons.</p>}
          </div>
        </Panel>
        <Panel className="p-5">
          <p className="text-xs uppercase tracking-[0.22em] text-gp-text-dim">Threshold preview</p>
          <pre className="mt-4 overflow-auto rounded-2xl bg-gp-panel-alt p-4 text-xs text-gp-text-muted">{JSON.stringify(preview?.metrics ?? {}, null, 2)}</pre>
        </Panel>
      </div>
    </div>
  );
}