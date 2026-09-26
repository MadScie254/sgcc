import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowRight, CheckCircle2, CircleDashed, Loader2, Play, XCircle } from "lucide-react";
import { getPipelineConfig, getPipelineRuns, getTrainingSummary, startPipelineRun } from "@/lib/api";
import { fmtDateTime, fmtInt, fmtNum, fmtSeconds } from "@/lib/format";
import { cn } from "@/lib/cn";
import { ApiError, Button, Card, CardHeader, Empty, PageHeader, Segmented, Skeleton } from "@/components/ui";

const SCORING_STAGES: Array<{ key: string; name: string; about: string; input: string }> = [
  { key: "ingest", name: "Ingest reads", about: "Loads the population in service (the unlabelled SGCC sample, or readings a supervisor promoted), parses the date header and sorts days chronologically. Missing reads stay missing: they are signal, not noise.", input: "Population in service" },
  { key: "features", name: "Build features", about: "Computes 87 behavioural features per customer with the settings frozen at training: gaps and zero runs, day-over-day drops, trends, year-over-year ratios, change points and a 34-month usage profile.", input: "Daily series" },
  { key: "score", name: "Score", about: "The served XGBoost model scores every customer; Platt scaling, fitted on validation customers, turns the score into a calibrated theft probability. Customers at or above the operating threshold become cases.", input: "Feature matrix" },
  { key: "explain", name: "Explain", about: "TreeSHAP breaks each flagged score into per-feature contributions, so every case arrives with the reasons behind it.", input: "Model + features" },
  { key: "route", name: "Route cases", about: "Opens a case for each newly flagged customer, keeping the status and notes of existing cases.", input: "Scores + explanations" },
];

const TRAINING_ABOUT: Record<string, string> = {
  Load: "Reads the full SGCC dataset: 42,372 customers, 1,034 days, 8.5% labelled theft.",
  Clean: "Proposal section 3.7: caps readings above 10,000 kWh, replaces 3-SD outliers with the customer's median, interpolates gaps up to 3 days and fills longer ones. Missingness is kept as signal.",
  Features: "The same features as scoring, on raw and on cleaned readings, grouped as statistical, temporal, trend and anomaly.",
  Split: "Customers split 70 / 15 / 15 into training, validation and test, stratified by label.",
  Resample: "Objective 1: measures class counts, separability and boundary noise before and after SMOTE and SMOTE+ENN on the training customers.",
  Tune: "Optuna searches the hyperparameters of both XGBoost pipelines on the mean PR-AUC of 5 cross-validation folds, each scored separately; SMOTE+ENN is redone inside every fold.",
  Validate: "Fits all five pipelines (tuned XGBoost with early stopping), calibrates each on validation (Platt scaling), picks each one's F1-maximising threshold on the calibrated validation probabilities, and keeps the two tuned XGBoost pipelines as candidates for service.",
  Sequence: "With PyTorch installed: trains the Wide & Deep CNN on the daily readings (early stopping on validation PR-AUC), calibrates it, and blends it with standard XGBoost; the blend weight, its calibration and its threshold are chosen on validation customers. The hybrid is served when its validation PR-AUC is highest; the CNN is exported to ONNX.",
  Evaluate: "Scores every pipeline once on the untouched test customers: effectiveness, calibration, training time, inference time and model size. The predictions are saved for the significance tests.",
  Publish: "Stages the model, its frozen pipeline spec, all results and an unlabelled population sample, moves them into place, and writes a SHA-256 manifest last; the API refuses to score if any file changes.",
};

type StageState = "done" | "running" | "failed" | "idle";

type Mode = "scoring" | "training";

function StageCard({ index, name, seconds, share, selected, state, onClick }: {
  index: number; name: string; seconds?: number; share: number; selected: boolean; state: StageState; onClick: () => void;
}) {
  const running = state === "running";
  return (
    <button type="button" onClick={onClick} aria-pressed={selected}
      className={cn("flex min-h-[124px] w-full flex-col items-start gap-2 rounded-xl p-3.5 text-left transition-colors",
        selected ? "border-[1.5px] border-cobalt bg-cobalt-soft" : "border border-line bg-[#FAFAF8] hover:border-ink-3")}>
      <span className="flex w-full items-center justify-between">
        <span className="font-mono text-[11px] text-ink-3">{String(index + 1).padStart(2, "0")}</span>
        {running ? <Loader2 className="h-4 w-4 animate-spin text-cobalt" aria-label="running" />
          : state === "failed" ? <XCircle className="h-4 w-4 text-risk" aria-label="failed" />
            : state === "done" ? <CheckCircle2 className="h-4 w-4 text-cobalt" aria-label="done" />
              : <CircleDashed className="h-4 w-4 text-ink-3" aria-label="not run" />}
      </span>
      <span className="text-sm font-semibold">{name}</span>
      <span className="font-mono text-xs text-ink-2">{running ? "running…" : fmtSeconds(seconds)}</span>
      <span className="block h-1 w-full overflow-hidden rounded-full bg-tint">
        <span className={cn("block h-full bg-cobalt", running && "animate-pulse")} style={{ width: `${Math.max(3, share * 100)}%` }} />
      </span>
    </button>
  );
}

export function PipelinePage() {
  const queryClient = useQueryClient();
  const [mode, setMode] = useState<Mode>("scoring");
  const [picked, setPicked] = useState<Record<Mode, number>>({ scoring: 2, training: 2 });

  const runs = useQuery({ queryKey: ["pipeline-runs", "history"], queryFn: () => getPipelineRuns(15) });
  const training = useQuery({ queryKey: ["training-summary"], queryFn: getTrainingSummary });
  const config = useQuery({ queryKey: ["pipeline-config"], queryFn: getPipelineConfig });
  // A run reloads the model and data, so every view is refreshed afterwards.
  const score = useMutation({ mutationFn: startPipelineRun, onSettled: () => queryClient.invalidateQueries() });

  const latest = runs.data?.[0];
  const firstMissing = SCORING_STAGES.findIndex((s) => !latest?.stages.some((r) => r.key === s.key));
  const scoringStages = SCORING_STAGES.map((s, i) => {
    const recorded = latest?.stages.find((r) => r.key === s.key);
    const state: StageState = score.isPending ? "running"
      : recorded ? "done"
        : latest?.status === "failed" && i === firstMissing ? "failed" : "idle";
    return { ...s, seconds: recorded?.seconds, detail: recorded?.detail, state };
  });
  const trainingStages = (training.data?.stages ?? []).map((s) => ({ ...s, state: "done" as StageState }));
  const stages = mode === "scoring" ? scoringStages : trainingStages;
  const maxSeconds = Math.max(...stages.map((s) => s.seconds ?? 0), 0.001);
  const selectedIndex = Math.min(picked[mode], Math.max(stages.length - 1, 0));
  const selected = stages[selectedIndex];
  const scoringDetail = mode === "scoring" ? scoringStages[selectedIndex] : undefined;
  const trainingTotal = trainingStages.reduce((sum, s) => sum + s.seconds, 0);

  return (
    <>
      <PageHeader
        eyebrow="Automated workflow"
        title="Pipeline"
        actions={
          <>
            <Segmented label="Pipeline" value={mode} onChange={setMode} options={[{ value: "scoring", label: "Daily scoring" }, { value: "training", label: "Training" }]} />
            {mode === "scoring" ? (
              <Button variant="primary" busy={score.isPending} onClick={() => score.mutate()}>
                {!score.isPending ? <Play className="h-4 w-4" aria-hidden /> : null}{score.isPending ? "Scoring…" : "Run scoring now"}
              </Button>
            ) : null}
          </>
        }
      />
      {score.isError ? <ApiError title="Scoring run failed" error={score.error} /> : null}
      {latest?.status === "failed" && !score.isPending ? (
        <div role="status" className="rounded-xl border border-risk/30 bg-risk-bg px-5 py-3.5 text-sm text-risk-ink">
          The last run ({latest.trigger}, {fmtDateTime(latest.finished_at)}) failed: {latest.error ?? "no error recorded"}
        </div>
      ) : null}
      {runs.isError ? <ApiError error={runs.error} /> : null}

      <Card label="Stages" className="flex flex-col gap-5 px-6 py-6">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <CardHeader
            title={mode === "scoring" ? "Daily scoring" : "Training"}
            subtitle={mode === "scoring" ? "Meter reads in, ranked and explained cases out" : `Full dataset to a tuned, evaluated model · trained ${fmtDateTime(training.data?.trained_at)}`}
          />
          <span className="flex items-center gap-2 text-[13px] font-medium text-cobalt-ink">
            <span className={cn("h-2 w-2 rounded-full", mode === "scoring" && latest?.status === "failed" ? "bg-risk" : "bg-cobalt")} />
            {mode === "scoring"
              ? latest ? `Last run ${latest.status} · ${fmtSeconds(latest.seconds)} end to end` : "No run yet"
              : trainingStages.length ? `${fmtSeconds(trainingTotal)} end to end` : "Stage timings not recorded"}
          </span>
        </div>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:flex lg:items-center">
          {stages.map((stage, i) => (
            <div key={`${mode}-${stage.name}`} className="flex flex-1 items-center gap-2">
              <StageCard index={i} name={stage.name} seconds={stage.seconds} share={Math.sqrt((stage.seconds ?? 0) / maxSeconds)}
                selected={i === selectedIndex} state={stage.state} onClick={() => setPicked((p) => ({ ...p, [mode]: i }))} />
              {i < stages.length - 1 ? <ArrowRight className="hidden h-3.5 w-3.5 shrink-0 text-[#9AA0AA] lg:block" aria-hidden /> : null}
            </div>
          ))}
          {stages.length === 0 ? (training.isLoading ? <Skeleton className="h-[124px] w-full" /> : <Empty title="No training record" message="artifacts/metrics.json has no stage timings." />) : null}
        </div>
      </Card>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1.25fr_1fr]">
        <Card label="Stage detail" className="flex flex-col gap-4 p-[22px]">
          <CardHeader title={selected?.name ?? "—"} action={<span className="font-mono text-[13px] text-ink-2">{fmtSeconds(selected?.seconds)}</span>} />
          <p className="m-0 text-sm leading-relaxed text-ink-2">
            {scoringDetail ? scoringDetail.about : TRAINING_ABOUT[selected?.name ?? ""]}
          </p>
          {scoringDetail ? (
            <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
              <div className="flex flex-col gap-1 rounded-lg bg-surface-alt px-3.5 py-3"><span className="text-[11px] uppercase tracking-[0.08em] text-ink-3">Input</span><span className="text-[13px]">{scoringDetail.input}</span></div>
              <div className="flex flex-col gap-1 rounded-lg bg-surface-alt px-3.5 py-3"><span className="text-[11px] uppercase tracking-[0.08em] text-ink-3">Result of last run</span><span className="text-[13px]">{scoringDetail.detail ?? "—"}</span></div>
            </div>
          ) : (
            <>
              <div className="rounded-lg bg-night px-4 py-3.5 font-mono text-xs leading-relaxed text-night-text">
                {training.data ? (
                  <>
                    <div>model v{training.data.model_version ?? "—"} · {training.data.n_trials ?? "—"} trials · CV {training.data.cv_metric ?? "—"} {fmtNum(training.data.cv_best_score)} (mean of {training.data.cv_fold_scores?.length ?? "—"} folds)</div>
                    <div>serving: {training.data.pipeline_label ?? "—"} · trained on {training.data.device ?? "cpu"}</div>
                    <div>train {fmtInt(training.data.train_customers)} · validation {fmtInt(training.data.validation_customers)} · test {fmtInt(training.data.test_customers)} customers · {training.data.n_features ?? "—"} features</div>
                    <div>code {training.data.provenance.code?.commit?.slice(0, 10) ?? "unknown"} · data sha256 {training.data.provenance.data_sha256?.slice(0, 12) ?? "unknown"}</div>
                    <div className="mt-2 text-night-muted">best params: {Object.entries(training.data.best_params).map(([k, v]) => `${k}=${+v.toPrecision(3)}`).join(", ") || "—"}</div>
                  </>
                ) : training.isError ? "Training record unavailable." : "…"}
              </div>
              <div className="flex flex-col gap-1.5 text-[13px] text-ink-2">
                <span>Training rewrites the served model, so it runs from the command line rather than the dashboard:</span>
                <pre className="m-0 overflow-x-auto rounded-lg bg-surface-alt px-3.5 py-2.5 font-mono text-xs text-ink">{"python -m src.train --quick         # smoke run into artifacts/quick/\npython -m src.train                 # full run: 40 trials, 5-fold CV\npython -m src.train --device cuda   # XGBoost on an NVIDIA GPU"}</pre>
                <span>Then press <strong className="font-medium text-ink">Run scoring now</strong> on the Daily scoring tab to load the new model.</span>
              </div>
            </>
          )}
        </Card>

        <div className="flex flex-col gap-4">
          <Card label="Automation" className="flex flex-col gap-3 p-[22px]">
            <CardHeader title="Automation" subtitle={config.data ? `Server environment: ${config.data.environment}` : undefined} />
            {[
              ["Score on server start", config.data ? (config.data.run_on_startup ? "On" : "Off") : "…", "Warms caches and records a run"],
              ["Scheduled scoring", config.data ? (config.data.scoring_interval_minutes > 0 ? `Every ${config.data.scoring_interval_minutes} min` : "Off") : "…", "SCORING_INTERVAL_MINUTES"],
              ["Training", "Command line", "python -m src.train"],
            ].map(([label, value, hint]) => (
              <div key={label} className="flex min-h-11 items-center justify-between gap-3 border-t border-line-soft pt-3 first-of-type:border-0 first-of-type:pt-0">
                <span className="flex flex-col"><span className="text-sm font-medium">{label}</span><span className="font-mono text-[11px] text-ink-3">{hint}</span></span>
                <span className="text-sm text-ink-2">{value}</span>
              </div>
            ))}
          </Card>

          <Card label="Run history" className="flex flex-1 flex-col gap-3 p-[22px]">
            <CardHeader title="Scoring runs" />
            {runs.data?.length ? (
              <table className="w-full border-collapse text-left text-[13px]">
                <thead>
                  <tr className="border-b border-line-soft text-[11px] uppercase tracking-[0.08em] text-ink-3">
                    <th className="pb-2 font-normal">Finished</th><th className="pb-2 font-normal">Trigger</th><th className="pb-2 font-normal">Duration</th><th className="pb-2 text-right font-normal">Flagged</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.data.slice(0, 8).map((run) => (
                    <tr key={run.run_id} className="h-10 border-b border-[#F1EFEA]">
                      <td className="flex h-10 items-center gap-2">
                        {run.status === "succeeded" ? <CheckCircle2 className="h-4 w-4 text-cobalt" aria-label="succeeded" /> : <XCircle className="h-4 w-4 text-risk" aria-label={run.status} />}
                        {fmtDateTime(run.finished_at)}
                      </td>
                      <td className="text-ink-2"><span className="capitalize">{run.trigger}</span>{run.actor && run.actor !== "system" ? <span className="block text-[11px] text-ink-3">{run.actor}</span> : null}</td>
                      <td className="font-mono">{fmtSeconds(run.seconds)}</td>
                      <td className="text-right font-mono">{run.summary ? fmtInt(run.summary.flagged) : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : runs.isLoading ? <Skeleton className="h-32" /> : <Empty title="No runs yet" message="Run scoring to record one." />}
          </Card>
        </div>
      </div>
    </>
  );
}
