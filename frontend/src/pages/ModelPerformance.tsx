import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiErrorMessage, getCompareBaselines, getGlobalShap, getModelMetrics, getTrainingSummary, type BaselineMetrics } from "@/lib/api";
import { featureLabel } from "@/lib/features";
import { fmtDateTime, fmtInt, fmtNum } from "@/lib/format";
import { Card, CardHeader, ErrorState, PageHeader, Skeleton, Stat } from "@/components/ui";
import { Legend } from "@/components/charts";

const MODELS: Array<{ key: string; label: string; color: string }> = [
  { key: "xgboost", label: "XGBoost (in service)", color: "#14161B" },
  { key: "random_forest", label: "Random forest", color: "#8A919C" },
  { key: "logistic_regression", label: "Logistic regression", color: "#C4C9D2" },
];

const METRICS: Array<{ key: keyof BaselineMetrics; label: string }> = [
  { key: "auc", label: "ROC-AUC" },
  { key: "pr_auc", label: "PR-AUC" },
  { key: "f1", label: "F1" },
  { key: "precision", label: "Precision" },
  { key: "recall", label: "Recall" },
];

export function ModelPerformancePage() {
  const metrics = useQuery({ queryKey: ["model-metrics"], queryFn: getModelMetrics });
  const compare = useQuery({ queryKey: ["compare"], queryFn: getCompareBaselines });
  const training = useQuery({ queryKey: ["training-summary"], queryFn: getTrainingSummary });
  const shap = useQuery({ queryKey: ["global-shap", 400], queryFn: () => getGlobalShap(400), staleTime: Infinity });

  const effects = useMemo(() => {
    if (!shap.data) return [];
    const { feature_names, shap_values, feature_values } = shap.data;
    return feature_names.map((name, j) => {
      const col = shap_values.map((row) => row[j]);
      const vals = feature_values.map((row) => row[j]);
      const meanAbs = col.reduce((s, v) => s + Math.abs(v), 0) / col.length;
      // Direction: does a higher feature value push towards theft? (rank-free sign of covariance)
      const pairs = col.map((s, i) => [vals[i], s] as const).filter(([v]) => v !== null) as Array<[number, number]>;
      const mx = pairs.reduce((s, [v]) => s + v, 0) / Math.max(pairs.length, 1);
      const ms = pairs.reduce((s, [, v]) => s + v, 0) / Math.max(pairs.length, 1);
      const cov = pairs.reduce((s, [v, sv]) => s + (v - mx) * (sv - ms), 0);
      return { name, meanAbs, direction: cov >= 0 ? "higher → more risk" : "lower → more risk" };
    }).sort((a, b) => b.meanAbs - a.meanAbs).slice(0, 14);
  }, [shap.data]);

  if (metrics.isError) return <ErrorState message={apiErrorMessage(metrics.error)} />;
  const m = metrics.data?.metrics;
  const models: Record<string, Partial<BaselineMetrics>> = {
    xgboost: (compare.data?.xgboost ?? {}) as Partial<BaselineMetrics>,
    ...((compare.data?.baselines ?? {}) as Record<string, Partial<BaselineMetrics>>),
  };
  const maxEffect = effects[0]?.meanAbs ?? 1;

  return (
    <>
      <PageHeader
        eyebrow={training.data ? `Hold-out: ${fmtInt(training.data.test_customers)} customers never seen in tuning or fitting · trained ${fmtDateTime(training.data.trained_at)}` : "Loading…"}
        title="Model performance"
        description="How well the model separates theft from honest customers, how it compares with simpler models, and what it pays attention to."
      />

      <section aria-label="Hold-out metrics" className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {m ? (
          <>
            <Stat label="ROC-AUC" value={fmtNum(m.auc)} hint="0.5 = random, 1.0 = perfect ranking" />
            <Stat label="PR-AUC" value={fmtNum(m.pr_auc)} hint={`vs ${((metrics.data?.base_rate ?? 0) * 100).toFixed(1)}% for random guessing`} />
            <Stat label="F1 at trained τ" value={fmtNum(m.f1)} hint={`precision ${fmtNum(m.precision)} · recall ${fmtNum(m.recall)}`} />
            <Stat label="MCC" value={fmtNum(m.mcc)} hint="balanced accuracy for rare classes" />
          </>
        ) : Array.from({ length: 4 }, (_, i) => <Skeleton key={i} className="h-[132px]" />)}
      </section>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1fr_1fr]">
        <Card label="Comparison with baselines" className="flex flex-col gap-5 p-[22px]">
          <CardHeader title="Against simpler models" subtitle="Same customers, same split. Baselines use a 0.5 threshold; XGBoost its tuned one." />
          <Legend items={MODELS.map((x) => ({ color: x.color, label: x.label }))} />
          {compare.data ? (
            <div className="flex flex-col gap-4">
              {METRICS.map((metric) => (
                <div key={metric.key} className="grid grid-cols-[88px_1fr] items-center gap-3">
                  <span className="text-[13px] text-ink-2">{metric.label}</span>
                  <div className="flex flex-col gap-1">
                    {MODELS.map((model) => {
                      const value = models[model.key]?.[metric.key] as number | undefined;
                      return (
                        <div key={model.key} className="flex items-center gap-2" title={`${model.label}: ${fmtNum(value)}`}>
                          <span className="h-3 rounded-[3px]" style={{ width: `${(value ?? 0) * 100}%`, background: model.color }} />
                          <span className="font-mono text-[11px] text-ink-2 tabular">{fmtNum(value)}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          ) : <Skeleton className="h-72" />}
        </Card>

        <Card label="What the model pays attention to" className="flex flex-col gap-4 p-[22px]">
          <CardHeader title="What drives the score" subtitle={`Mean |SHAP| over ${shap.data?.sample_count ?? "…"} served customers`} />
          {effects.length ? (
            <ol className="m-0 flex list-none flex-col gap-2 p-0">
              {effects.map((effect) => (
                <li key={effect.name} className="grid grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)_52px] items-center gap-3 text-[13px]">
                  <span className="flex flex-col"><span className="font-medium">{featureLabel(effect.name)}</span><span className="text-[11px] text-ink-3">{effect.direction}</span></span>
                  <span className="h-2.5 rounded-[3px] bg-cobalt" style={{ width: `${(effect.meanAbs / maxEffect) * 100}%` }} />
                  <span className="text-right font-mono text-xs text-ink-2 tabular">{effect.meanAbs.toFixed(3)}</span>
                </li>
              ))}
            </ol>
          ) : <Skeleton className="h-80" />}
        </Card>
      </div>

      <Card label="Training configuration" className="flex flex-col gap-4 p-[22px]">
        <CardHeader title="How it was trained" subtitle={training.data ? `${training.data.n_trials} Optuna trials · best cross-validated ${training.data.cv_metric} ${fmtNum(training.data.cv_best_score)} · ${training.data.n_features} features` : "…"} />
        <dl className="m-0 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
          {Object.entries(training.data?.best_params ?? {}).map(([key, value]) => (
            <div key={key} className="flex flex-col gap-1 rounded-lg bg-surface-alt px-3.5 py-3">
              <dt className="font-mono text-[11px] text-ink-3">{key}</dt>
              <dd className="m-0 font-mono text-sm">{typeof value === "number" ? +value.toPrecision(4) : String(value)}</dd>
            </div>
          ))}
        </dl>
      </Card>
    </>
  );
}
