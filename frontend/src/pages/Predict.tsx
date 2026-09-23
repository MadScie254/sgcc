import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { getModelConfig, getModelMetrics, predictSingle } from "@/lib/api";
import { CenteredState } from "@/components/CenteredState";
import { CustomerPicker } from "@/components/CustomerPicker";
import { Panel, Button, Input } from "@/components/ui/primitives";
import { ShapDrivers } from "@/components/ShapDrivers";
import { HIGH_RISK_PROBABILITY, riskTierFromProbability, riskTierLabel, riskTierClasses } from "@/lib/dashboard";

type EntryMode = "lookup" | "manual";

export function PredictPage() {
  const [searchParams] = useSearchParams();
  const [mode, setMode] = useState<EntryMode>("lookup");
  const [selectedCustomerId, setSelectedCustomerId] = useState<string | null>(searchParams.get("customer"));
  const [threshold, setThreshold] = useState<number | null>(null);
  const [prediction, setPrediction] = useState<Awaited<ReturnType<typeof predictSingle>> | null>(null);
  const [manualValues, setManualValues] = useState<Record<string, string>>({});

  const configQuery = useQuery({ queryKey: ["model-config"], queryFn: getModelConfig, staleTime: 30_000 });
  const metricsQuery = useQuery({ queryKey: ["model-metrics"], queryFn: getModelMetrics, staleTime: 30_000 });
  // Start from the model's tuned decision threshold until the user moves the slider.
  const effectiveThreshold = threshold ?? metricsQuery.data?.threshold ?? 0.5;
  const featureGroups = useMemo(() => Object.entries(configQuery.data?.feature_groups ?? {}), [configQuery.data]);

  useEffect(() => {
    if (!featureGroups.length || Object.keys(manualValues).length > 0) return;
    const initialValues: Record<string, string> = {};
    featureGroups.forEach(([, features]) => {
      features.forEach((feature) => {
        initialValues[feature] = "";
      });
    });
    setManualValues(initialValues);
  }, [featureGroups, manualValues]);

  async function runPrediction() {
    if (mode === "lookup") {
      if (!selectedCustomerId) return;
      const response = await predictSingle({ customer_id: selectedCustomerId, threshold: effectiveThreshold });
      setPrediction(response);
      return;
    }

    // Blank fields are omitted and treated as missing by the model, as in training.
    const payload: Record<string, number> = {};
    Object.entries(manualValues).forEach(([feature, value]) => {
      const numeric = Number.parseFloat(value);
      if (Number.isFinite(numeric)) {
        payload[feature] = numeric;
      }
    });
    const response = await predictSingle({ features: payload, threshold: effectiveThreshold });
    setPrediction(response);
  }

  const probability = prediction?.probability ?? null;
  const highCutoff = Math.max(HIGH_RISK_PROBABILITY, effectiveThreshold);
  const derivedLabel = probability === null ? null : probability >= effectiveThreshold ? 1 : 0;
  const derivedTier = probability === null ? null : riskTierFromProbability(probability, effectiveThreshold);

  if (configQuery.isError) {
    return <CenteredState title="Model unavailable" description="The prediction form could not load the model configuration. Verify the API and trained artifacts." />;
  }

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-primary">Predict</h1>
        <p className="mt-1 text-sm text-secondary">Look up a customer or enter features manually, then reuse the cached probability as the threshold changes.</p>
      </div>

      <div className="flex flex-wrap gap-2 rounded-md border border-border bg-surface p-1">
        {([
          ["lookup", "Lookup customer"],
          ["manual", "Manual input"],
        ] as const).map(([value, label]) => (
          <button
            key={value}
            type="button"
            onClick={() => setMode(value)}
            className={`rounded-md px-3 py-2 text-sm font-medium transition ${mode === value ? "bg-accent-bg text-accent" : "text-secondary hover:bg-surface-alt"}`}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.05fr_0.95fr]">
        <Panel className="rounded-lg border border-border bg-surface p-4">
          <div className="text-xs uppercase tracking-[0.18em] text-secondary">Input</div>

          {mode === "lookup" ? (
            <div className="mt-4 space-y-4">
              <CustomerPicker selectedCustomerId={selectedCustomerId} onSelect={setSelectedCustomerId} />

              <div className="rounded-md border border-border bg-surface-alt px-4 py-3 text-sm text-secondary">
                {selectedCustomerId ? (
                  <>
                    Selected customer <span className="font-mono text-primary">{selectedCustomerId}</span>
                  </>
                ) : (
                  "Search and select a customer to score."
                )}
              </div>
            </div>
          ) : (
            <div className="mt-4 space-y-4">
              {featureGroups.map(([groupName, features]) => (
                <details key={groupName} className="rounded-md border border-border bg-surface" open={groupName === "statistical"}>
                  <summary className="cursor-pointer list-none border-b border-border px-4 py-3 text-sm font-medium capitalize text-primary">
                    {groupName}
                  </summary>
                  <div className="grid gap-3 p-4 md:grid-cols-2">
                    {features.map((feature) => (
                      <label key={feature} className="space-y-1 text-sm">
                        <span className="block text-xs uppercase tracking-[0.16em] text-secondary">{feature}</span>
                        <Input
                          inputMode="decimal"
                          value={manualValues[feature] ?? ""}
                          onChange={(event) => setManualValues((current) => ({ ...current, [feature]: event.target.value }))}
                          placeholder="missing"
                        />
                      </label>
                    ))}
                  </div>
                </details>
              ))}
            </div>
          )}

          <div className="mt-5 flex items-center gap-3">
            <div className="min-w-40 space-y-2">
              <div className="flex items-center justify-between text-xs uppercase tracking-[0.18em] text-secondary">
                <span>Threshold</span>
                <span className="font-mono text-primary">{effectiveThreshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.05"
                max="0.95"
                step="0.01"
                value={effectiveThreshold}
                onChange={(event) => setThreshold(Number(event.target.value))}
                className="h-2 w-full cursor-pointer appearance-none rounded-full bg-surface-alt accent-accent"
              />
            </div>
            <Button type="button" onClick={() => void runPrediction()} className="rounded-md border-accent bg-accent-bg text-accent">
              Run prediction
            </Button>
          </div>
        </Panel>

        <Panel className="rounded-lg border border-border bg-surface p-4">
          <div className="text-xs uppercase tracking-[0.18em] text-secondary">Result</div>
          {prediction ? (
            <div className="mt-4 space-y-5">
              <div className="rounded-md border border-border bg-surface px-4 py-4">
                <div className="flex items-center justify-between text-sm text-secondary">
                  <span>Risk probability</span>
                  <span className="font-mono text-primary">{(probability ?? 0).toFixed(3)}</span>
                </div>
                <div className="mt-4 rounded-full border border-border bg-surface-alt p-1">
                  <div className="relative h-3 overflow-hidden rounded-full">
                    <div className="absolute inset-y-0 left-0 bg-success-bg" style={{ width: `${effectiveThreshold * 100}%` }} />
                    <div className="absolute inset-y-0 bg-warning-bg" style={{ left: `${effectiveThreshold * 100}%`, width: `${Math.max(highCutoff - effectiveThreshold, 0) * 100}%` }} />
                    <div className="absolute inset-y-0 right-0 bg-danger-bg" style={{ width: `${(1 - highCutoff) * 100}%` }} />
                    <div
                      className="absolute top-[-4px] h-5 w-[2px] bg-primary"
                      style={{ left: `${(probability ?? 0) * 100}%`, transform: "translateX(-1px)" }}
                    />
                  </div>
                </div>
                <div className="mt-3 flex flex-wrap items-center gap-2 text-sm">
                  <span className={`inline-flex rounded-full border px-2.5 py-1 text-xs font-semibold ${riskTierClasses(derivedTier ?? "low")}`}>{riskTierLabel(derivedTier ?? "low")}</span>
                  <span className={`inline-flex rounded-full border px-2.5 py-1 text-xs font-semibold ${derivedLabel ? "border-danger bg-danger-bg text-danger" : "border-success bg-success-bg text-success"}`}>
                    {derivedLabel ? "Theft" : "Normal"}
                  </span>
                </div>
              </div>

              <ShapDrivers items={prediction.top_reasons} />
            </div>
          ) : (
            <CenteredState title="No prediction yet" description="Run the model once to see the probability gauge and the top SHAP drivers." />
          )}
        </Panel>
      </div>

      {mode === "lookup" && selectedCustomerId ? <div className="text-xs text-secondary">Selected customer <span className="font-mono text-primary">{selectedCustomerId}</span>.</div> : null}
    </div>
  );
}
