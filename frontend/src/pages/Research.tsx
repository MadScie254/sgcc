import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { FlaskConical } from "lucide-react";
import {
  getCalibration, getModelComparison, getModelDrivers, getResamplingEffect, getResearchCurve, getResearchDistribution,
  getResearchEvaluation, getTrainingSummary,
} from "@/lib/api";
import { fmtDateTime, fmtInt, fmtNum } from "@/lib/format";
import { cn } from "@/lib/cn";
import { ApiError, Card, CardHeader, PageHeader, Skeleton, Stat } from "@/components/ui";
import { ConfusionGrid, Legend, ReliabilityChart, RiskHistogram, TradeoffChart } from "@/components/charts";

// Validated categorical slots in fixed order; identity only, every value is also printed.
const PIPELINE_COLORS: Record<string, string> = {
  proposed: "#2a78d6",
  xgboost: "#eb6834",
  xgboost_default: "#1baf7a",
  random_forest_smote: "#eda100",
  logistic_regression_smote: "#e87ba4",
};

const SCORE_COLUMNS = [
  { key: "pr_auc", label: "PR-AUC" },
  { key: "recall", label: "Recall" },
  { key: "precision", label: "Precision" },
  { key: "f1", label: "F1" },
  { key: "gmean", label: "G-Mean" },
  { key: "mcc", label: "MCC" },
] as const;

const DIRECTION = { higher: "higher → more risk", lower: "lower → more risk", unclear: "no consistent direction" } as const;

function pValue(p: number | null, isReference: boolean) {
  if (isReference) return <span className="text-ink-3">reference</span>;
  if (p === null) return <span className="text-ink-3">—</span>;
  return <span className={cn(p < 0.05 && "font-semibold text-ink")}>{p < 0.001 ? "<0.001" : p.toFixed(3)}{p < 0.05 ? " *" : ""}</span>;
}

const interval = (ci: [number, number] | null) => (ci ? `${ci[0].toFixed(3)}–${ci[1].toFixed(3)}` : "—");

export function ResearchPage() {
  const evaluation = useQuery({ queryKey: ["research-evaluation"], queryFn: getResearchEvaluation });
  const comparison = useQuery({ queryKey: ["model-comparison"], queryFn: getModelComparison });
  const curve = useQuery({ queryKey: ["research-curve"], queryFn: getResearchCurve });
  const dist = useQuery({ queryKey: ["research-distribution"], queryFn: getResearchDistribution });
  const calibration = useQuery({ queryKey: ["calibration"], queryFn: getCalibration });
  const training = useQuery({ queryKey: ["training-summary"], queryFn: getTrainingSummary });
  const resampling = useQuery({ queryKey: ["resampling-effect"], queryFn: getResamplingEffect });
  const drivers = useQuery({ queryKey: ["model-drivers"], queryFn: getModelDrivers, staleTime: Infinity });
  const [index, setIndex] = useState<number | null>(null);

  useEffect(() => {
    if (index === null && curve.data) {
      const { points, threshold } = curve.data;
      setIndex(points.reduce((b, p, i) => (Math.abs(p.threshold - threshold) < Math.abs(points[b].threshold - threshold) ? i : b), 0));
    }
  }, [index, curve.data]);

  if (evaluation.isError) return <ApiError error={evaluation.error} />;
  const e = evaluation.data;
  const m = e?.metrics;
  const served = comparison.data?.find((row) => row.served);
  const point = curve.data && index !== null ? curve.data.points[index] : null;
  const folds = training.data?.cv_fold_scores ?? [];
  const foldSd = folds.length > 1 ? Math.sqrt(folds.reduce((s, v) => s + (v - (training.data?.cv_best_score ?? 0)) ** 2, 0) / (folds.length - 1)) : null;
  const commit = training.data?.provenance.code?.commit;
  const effects = drivers.data?.drivers ?? [];
  const maxEffect = effects[0]?.mean_abs_shap || 1;
  const rows = comparison.data ?? [];
  const best = Object.fromEntries(SCORE_COLUMNS.map(({ key }) => [key, Math.max(...rows.map((r) => r[key]))]));
  const hasTests = rows.some((r) => r.p_value_pr_auc !== null);
  const r = resampling.data;
  const stages = r ? [
    { name: "Original", counts: r.smote_enn.counts.before, stats: r.before },
    { name: "After SMOTE", counts: r.smote.counts.after, stats: r.smote.diagnostics },
    { name: "After SMOTE+ENN", counts: r.smote_enn.counts.after, stats: r.smote_enn.diagnostics },
  ] : [];

  return (
    <>
      <PageHeader
        eyebrow={e ? `Research record · ${e.population.description}` : "Loading…"}
        title="Research evaluation"
        description={`How the pipeline in service${e ? ` (${e.pipeline_label})` : ""} and the other pipelines of the study did on the test customers, how well calibrated their probabilities are, what SMOTE+ENN does to the training data, and how the model was trained. Nothing here is used to run operations.`}
      />
      {e ? (
        <p role="note" className="m-0 flex items-start gap-2.5 rounded-lg bg-amber-bg px-3.5 py-3 text-[13px] text-amber-ink">
          <FlaskConical className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
          <span>Population: {fmtInt(e.population.customers)} test customers ({fmtInt(e.population.theft)} thieves), scored once after every choice was made on training and validation customers. {e.population.limitation}</span>
        </p>
      ) : null}

      <section aria-label="Test-set metrics" className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {m && e ? (
          <>
            <Stat label="ROC-AUC" value={fmtNum(m.auc)} hint="0.5 = random, 1.0 = perfect ranking" />
            <Stat label="PR-AUC" value={fmtNum(m.pr_auc)} hint={`95% CI ${interval(served?.pr_auc_ci ?? null)} · random ${(e.population.theft / e.population.customers).toFixed(3)}`} />
            <Stat label={`F1 at trained τ ${e.threshold.toFixed(3)}`} value={fmtNum(m.f1)} hint={`precision ${fmtNum(m.precision)} · recall ${fmtNum(m.recall)}`} />
            <Stat label="MCC" value={fmtNum(m.mcc)} hint="balanced accuracy for rare classes" />
          </>
        ) : Array.from({ length: 4 }, (_, i) => <Skeleton key={i} className="h-[132px]" />)}
      </section>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1.3fr_1fr]">
        <Card label="Precision and recall on the test customers" className="flex flex-col gap-3 p-[22px]">
          <CardHeader title="Precision and recall by threshold" subtitle="Test customers; click to read another threshold. The threshold in use was chosen on validation."
            action={<Legend items={[{ color: "#2346C8", label: "Precision", line: true }, { color: "#E0873A", label: "Recall", line: true }]} />} />
          {curve.data && index !== null ? (
            <TradeoffChart points={curve.data.points} index={index} onPick={setIndex} trainedThreshold={curve.data.threshold} />
          ) : <Skeleton className="h-64" />}
        </Card>
        <Card label="Test-set outcome" className="flex flex-col gap-4 p-[22px]">
          <CardHeader title={point ? `Test customers at τ ${point.threshold.toFixed(2)}` : "Test customers"} subtitle={point ? `precision ${fmtNum(point.precision)} · recall ${fmtNum(point.recall)}` : undefined} />
          {point ? <ConfusionGrid tp={point.tp} fp={point.fp} fn={point.fn} tn={point.tn} /> : <Skeleton className="h-40" />}
          <CardHeader title="Calibrated probabilities by true class" action={<span className="text-xs text-ink-3">log scale</span>} />
          {dist.data ? (
            <RiskHistogram edges={dist.data.edges} total={dist.data.honest.map((h, i) => h + dist.data.theft[i])} theft={dist.data.theft} threshold={dist.data.threshold} height={120} />
          ) : <Skeleton className="h-[140px]" />}
          <Legend items={[{ color: "#C4C9D2", label: "Honest" }, { color: "#B93C15", label: "Theft" }]} />
        </Card>
      </div>

      <Card label="Pipelines compared" className="flex flex-col gap-4 p-[22px]">
        <CardHeader title="Pipelines compared"
          subtitle="Same test customers for every pipeline, each at the threshold that maximised F1 on the validation customers. Best score per column in bold; 95% intervals from a paired stratified bootstrap." />
        {comparison.data ? (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[1080px] border-collapse text-left text-[13px]">
              <thead>
                <tr className="border-b border-line-soft text-[11px] uppercase tracking-[0.08em] text-ink-3">
                  <th className="pb-2 font-normal">Pipeline</th>
                  {SCORE_COLUMNS.map((c) => <th key={c.key} className="pb-2 text-right font-normal">{c.label}</th>)}
                  <th className="pb-2 text-right font-normal">PR-AUC 95% CI</th>
                  <th className="pb-2 text-right font-normal" title="Paired stratified bootstrap of the PR-AUC difference from the proposed pipeline, Holm-adjusted">p (PR-AUC)</th>
                  <th className="pb-2 text-right font-normal" title="McNemar's exact test on the flag decisions, against the proposed pipeline, Holm-adjusted">p (McNemar)</th>
                  <th className="pb-2 text-right font-normal">Train s</th>
                  <th className="pb-2 text-right font-normal">ms / customer</th>
                  <th className="pb-2 text-right font-normal">MB</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.model} className="h-11 border-b border-[#F1EFEA]">
                    <td>
                      <span className="flex items-center gap-2">
                        <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: PIPELINE_COLORS[row.model] }} aria-hidden />
                        <span className="font-medium">{row.label}</span>
                        {row.served ? <span className="rounded-full bg-ink px-2 py-0.5 text-[11px] font-medium text-white">in service</span> : null}
                      </span>
                    </td>
                    {SCORE_COLUMNS.map((c) => (
                      <td key={c.key} className={cn("text-right font-mono tabular", row[c.key] === best[c.key] && "font-semibold")}>{fmtNum(row[c.key])}</td>
                    ))}
                    <td className="whitespace-nowrap text-right font-mono text-xs tabular text-ink-2">{interval(row.pr_auc_ci)}</td>
                    <td className="text-right font-mono text-xs tabular text-ink-2">{pValue(row.p_value_pr_auc, row.model === "proposed")}</td>
                    <td className="text-right font-mono text-xs tabular text-ink-2">{pValue(row.p_value_mcnemar, row.model === "proposed")}</td>
                    <td className="text-right font-mono tabular text-ink-2">{row.training_time.toFixed(1)}</td>
                    <td className="text-right font-mono tabular text-ink-2">{row.inference_ms_per_customer.toFixed(3)}</td>
                    <td className="text-right font-mono tabular text-ink-2">{row.model_size_mb.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="m-0 mt-3 text-xs text-ink-3">
              {hasTests
                ? "* significant at α = 0.05, Holm-adjusted across the four comparisons: a paired stratified bootstrap (10,000 resamples of the test customers) for PR-AUC and McNemar's exact test for the flag decisions."
                : "Run scripts/significance.py to add the paired significance tests."}
              {" "}Times on a 4-core CPU; training time includes resampling.
            </p>
          </div>
        ) : comparison.isError ? <ApiError error={comparison.error} /> : <Skeleton className="h-56" />}
      </Card>

      <Card label="Calibration" className="flex flex-col gap-4 p-[22px]">
        <CardHeader title="Are the probabilities honest?"
          subtitle="Platt scaling fitted on the validation customers (served); isotonic regression shown for comparison. Lower is better; test customers." />
        {calibration.data ? (
          <div className="grid grid-cols-1 gap-6 xl:grid-cols-[380px_1fr]">
            <div className="flex flex-col gap-2">
              <ReliabilityChart raw={calibration.data.reliability.raw ?? []} calibrated={calibration.data.reliability.platt ?? []} />
              <Legend items={[{ color: "#E0873A", label: "Raw score", line: true }, { color: "#2346C8", label: "Calibrated (Platt)", line: true }, { color: "#D9D6CE", label: "Perfect calibration", line: true }]} />
              <span className="text-xs text-ink-3">Observed theft rate against predicted probability, pipeline in service.</span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[560px] border-collapse text-left text-[13px]">
                <thead>
                  <tr className="border-b border-line-soft text-[11px] uppercase tracking-[0.08em] text-ink-3">
                    <th className="pb-2 font-normal">Pipeline</th>
                    <th className="pb-2 text-right font-normal">Brier raw</th>
                    <th className="pb-2 text-right font-normal">Brier Platt</th>
                    <th className="pb-2 text-right font-normal">ECE raw</th>
                    <th className="pb-2 text-right font-normal">ECE Platt</th>
                    <th className="pb-2 text-right font-normal">ECE isotonic</th>
                    <th className="pb-2 text-right font-normal">Log-loss Platt</th>
                  </tr>
                </thead>
                <tbody>
                  {calibration.data.pipelines.map((row) => (
                    <tr key={row.model} className="h-10 border-b border-[#F1EFEA]">
                      <td className="font-medium">{row.label}</td>
                      <td className="text-right font-mono tabular text-ink-2">{row.test_raw.brier.toFixed(4)}</td>
                      <td className="text-right font-mono tabular">{row.test_platt.brier.toFixed(4)}</td>
                      <td className="text-right font-mono tabular text-ink-2">{row.test_raw.ece.toFixed(4)}</td>
                      <td className="text-right font-mono tabular">{row.test_platt.ece.toFixed(4)}</td>
                      <td className="text-right font-mono tabular text-ink-2">{row.test_isotonic.ece.toFixed(4)}</td>
                      <td className="text-right font-mono tabular">{row.test_platt.log_loss.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : calibration.isError ? <ApiError error={calibration.error} /> : <Skeleton className="h-64" />}
      </Card>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1fr_1fr]">
        <Card label="Effect of SMOTE+ENN on the training data" className="flex flex-col gap-4 p-[22px]">
          <CardHeader title="What SMOTE+ENN does to the training data"
            subtitle={r ? `Objective 1. SMOTE to theft : honest = ${r.config.sampling_strategy} with k = ${r.config.smote_k_neighbors}, then ENN with k = ${r.config.enn_n_neighbors}; training customers only.` : undefined} />
          {r ? (
            <>
              <div className="overflow-x-auto">
              <table className="w-full min-w-[480px] border-collapse text-left text-[13px]">
                <thead>
                  <tr className="border-b border-line-soft text-[11px] uppercase tracking-[0.08em] text-ink-3">
                    <th className="pb-2 font-normal">Training data</th>
                    <th className="pb-2 text-right font-normal">Honest</th>
                    <th className="pb-2 text-right font-normal">Theft</th>
                    <th className="pb-2 text-right font-normal" title="Mean silhouette of the two classes">Silhouette</th>
                    <th className="pb-2 text-right font-normal" title="Mean Fisher discriminant ratio over features">Fisher</th>
                    <th className="pb-2 text-right font-normal" title="Share of theft rows whose 5 nearest neighbours are mostly honest">Noisy theft</th>
                  </tr>
                </thead>
                <tbody>
                  {stages.map((s) => (
                    <tr key={s.name} className="h-10 border-b border-[#F1EFEA]">
                      <td className="font-medium">{s.name}</td>
                      <td className="text-right font-mono tabular">{fmtInt(s.counts.honest)}</td>
                      <td className="text-right font-mono tabular">{fmtInt(s.counts.theft)}</td>
                      <td className="text-right font-mono tabular">{s.stats.silhouette.toFixed(3)}</td>
                      <td className="text-right font-mono tabular">{s.stats.fisher_ratio_mean.toFixed(3)}</td>
                      <td className="text-right font-mono tabular">{(s.stats.boundary_noise_theft * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              </div>
              <p className="m-0 text-[13px] text-ink-2">
                SMOTE created {fmtInt(r.smote_enn.counts.synthetic_created)} synthetic theft customers; ENN then removed {fmtInt(r.smote_enn.counts.honest_removed_by_enn)} honest
                and {fmtInt((r.smote_enn.counts.theft_removed_by_enn ?? 0) + (r.smote_enn.counts.synthetic_removed_by_enn ?? 0))} theft rows
                ({fmtInt(r.smote_enn.counts.synthetic_removed_by_enn)} of them synthetic) whose neighbours disagreed with their label.
              </p>
            </>
          ) : resampling.isError ? <ApiError error={resampling.error} /> : <Skeleton className="h-48" />}
        </Card>

        <Card label="What the model pays attention to" className="flex flex-col gap-4 p-[22px]">
          <CardHeader title="What drives the score" subtitle={`Mean |SHAP| on the raw score over ${drivers.data ? fmtInt(drivers.data.sample_size) : "…"} customers of the population in service`} />
          {effects.length ? (
            <ol className="m-0 flex list-none flex-col gap-2 p-0">
              {effects.map((effect) => (
                <li key={effect.feature} className="grid grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)_52px] items-center gap-3 text-[13px]">
                  <span className="flex flex-col"><span className="font-medium">{effect.label}</span><span className="text-[11px] text-ink-3">{DIRECTION[effect.risk_when]}</span></span>
                  <span className="h-2.5 rounded-[3px] bg-cobalt" style={{ width: `${(effect.mean_abs_shap / maxEffect) * 100}%` }} />
                  <span className="text-right font-mono text-xs text-ink-2 tabular">{effect.mean_abs_shap.toFixed(3)}</span>
                </li>
              ))}
            </ol>
          ) : drivers.isError ? <ApiError error={drivers.error} /> : <Skeleton className="h-80" />}
        </Card>
      </div>

      <Card label="Training configuration" className="flex flex-col gap-4 p-[22px]">
        <CardHeader title="How it was trained"
          subtitle={training.data
            ? `${fmtInt(training.data.train_customers)} training / ${fmtInt(training.data.validation_customers)} validation / ${fmtInt(training.data.test_customers)} test customers · ${training.data.n_trials} Optuna trials · best ${training.data.cv_metric} ${fmtNum(training.data.cv_best_score)}${foldSd !== null ? ` ± ${fmtNum(foldSd)}` : ""} (mean ± SD over ${folds.length} folds) · ${training.data.n_features} features · trained ${fmtDateTime(training.data.trained_at)} on ${training.data.device ?? "cpu"}`
            : "…"} />
        <dl className="m-0 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
          {Object.entries(training.data?.best_params ?? {}).map(([key, value]) => (
            <div key={key} className="flex flex-col gap-1 rounded-lg bg-surface-alt px-3.5 py-3">
              <dt className="font-mono text-[11px] text-ink-3">{key}</dt>
              <dd className="m-0 font-mono text-sm">{+value.toPrecision(4)}</dd>
            </div>
          ))}
        </dl>
        {training.data ? (
          <p className="m-0 font-mono text-xs text-ink-3">
            code {commit ? commit.slice(0, 10) : "unknown"}{training.data.provenance.code?.dirty ? " (uncommitted changes)" : ""} · data sha256{" "}
            {training.data.provenance.data_sha256?.slice(0, 12) ?? "unknown"} · xgboost {training.data.provenance.libraries?.xgboost ?? "?"} ·
            scikit-learn {training.data.provenance.libraries?.["scikit-learn"] ?? "?"} · published {fmtDateTime(training.data.manifest.published_at ?? null)}
          </p>
        ) : null}
      </Card>
    </>
  );
}
