import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowRight, CheckCircle2, CircleDashed, Loader2, Play, XCircle } from "lucide-react";
import {
  apiErrorMessage,
  createTrainingJob,
  getPipelineConfig,
  getPipelineRuns,
  getTrainingJob,
  getTrainingSummary,
  startPipelineRun,
} from "@/lib/api";
import { fmtDateTime, fmtInt, fmtNum, fmtSeconds } from "@/lib/format";
import { cn } from "@/lib/cn";
import { Button, Card, CardHeader, Empty, ErrorState, PageHeader, Segmented, Skeleton } from "@/components/ui";

const SCORING_ABOUT: Record<string, { about: string; input: string; output: string }> = {
  ingest: { about: "Loads each customer's daily kWh series, parses the date header and sorts days chronologically. Missing reads stay missing: they are signal, not noise.", input: "data/sgcc_demo.csv.gz", output: "Customer × day matrix" },
  features: { about: "Computes 85 behavioural features per customer: gaps and zero runs, day-over-day drops, trends, year-over-year ratios, change points and a 34-month usage profile.", input: "Daily series", output: "Feature matrix" },
  score: { about: "The tuned XGBoost model assigns every customer a theft probability. Customers at or above the operating threshold become cases.", input: "Feature matrix", output: "Theft probabilities" },
  explain: { about: "TreeSHAP breaks each flagged score into per-feature contributions, so every case arrives with the reasons behind it.", input: "Model + features", output: "SHAP drivers per case" },
  route: { about: "Opens a case for each newly flagged customer, keeping the status and notes of existing cases.", input: "Scores + explanations", output: "Investigation queue" },
};

const TRAINING_ABOUT: Record<string, string> = {
  Load: "Reads the full SGCC dataset (42,372 customers) and drops duplicated customers.",
  Features: "The same 85 features as scoring, then a stratified 80/20 split by customer.",
  Tune: "Optuna searches ten XGBoost hyperparameters, each trial scored by cross-validated PR-AUC on training customers only.",
  Threshold: "Out-of-fold predictions choose the threshold that maximises F1, never touching the test customers.",
  Evaluate: "Fits the final model and measures it once on the held-out customers.",
  Baselines: "Logistic regression and random forest on the same split, as reference points.",
  Publish: "Writes the model, metrics and the held-out sample the dashboard serves.",
};

type Mode = "scoring" | "training";

function StageCard({ index, name, seconds, share, selected, running, failed, onClick }: {
  index: number; name: string; seconds?: number; share: number; selected: boolean; running?: boolean; failed?: boolean; onClick: () => void;
}) {
  return (
    <button type="button" onClick={onClick} aria-pressed={selected}
      className={cn("flex min-h-[124px] w-full flex-col items-start gap-2 rounded-xl p-3.5 text-left transition-colors",
        selected ? "border-[1.5px] border-cobalt bg-cobalt-soft" : "border border-line bg-[#FAFAF8] hover:border-ink-3")}>
      <span className="flex w-full items-center justify-between">
        <span className="font-mono text-[11px] text-ink-3">{String(index + 1).padStart(2, "0")}</span>
        {running ? <Loader2 className="h-4 w-4 animate-spin text-cobalt" aria-label="running" />
          : failed ? <XCircle className="h-4 w-4 text-risk" aria-label="failed" />
            : seconds !== undefined ? <CheckCircle2 className="h-4 w-4 text-cobalt" aria-label="done" />
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
  const [picked, setPicked] = useState<Record<Mode, number>>({ scoring: 3, training: 2 });
  const [jobId, setJobId] = useState<string | null>(null);

  const runs = useQuery({ queryKey: ["pipeline-runs", "history"], queryFn: () => getPipelineRuns(15) });
  const training = useQuery({ queryKey: ["training-summary"], queryFn: getTrainingSummary });
  const config = useQuery({ queryKey: ["pipeline-config"], queryFn: getPipelineConfig });
  const job = useQuery({
    queryKey: ["training-job", jobId],
    queryFn: () => getTrainingJob(jobId as string),
    enabled: jobId !== null,
    refetchInterval: (q) => (q.state.data && ["succeeded", "failed"].includes(q.state.data.status) ? false : 2000),
  });

  const score = useMutation({ mutationFn: startPipelineRun, onSuccess: () => queryClient.invalidateQueries() });
  const retrain = useMutation({
    mutationFn: () => createTrainingJob({ mode: "quick" }),
    onSuccess: (data) => setJobId(data.job_id),
  });

  const latest = runs.data?.[0];
  const scoringStages = (latest?.stages ?? Object.keys(SCORING_ABOUT).map((key) => ({ key, name: key, seconds: undefined as unknown as number, detail: "" })));
  const trainingStages = training.data?.stages ?? [];
  const stages = mode === "scoring"
    ? scoringStages.map((s) => ({ name: s.name, seconds: s.seconds as number | undefined }))
    : trainingStages.map((s) => ({ name: s.name, seconds: s.seconds as number | undefined }));
  const maxSeconds = Math.max(...stages.map((s) => s.seconds ?? 0), 0.001);
  const selectedIndex = Math.min(picked[mode], Math.max(stages.length - 1, 0));
  const total = stages.reduce((sum, s) => sum + (s.seconds ?? 0), 0);

  const scoringDetail = mode === "scoring" ? scoringStages[selectedIndex] : undefined;
  const trainingDetail = mode === "training" ? trainingStages[selectedIndex] : undefined;

  return (
    <>
      <PageHeader
        eyebrow="Automated workflow"
        title="Pipeline"
        actions={
          <>
            <Segmented label="Pipeline" value={mode} onChange={setMode} options={[{ value: "scoring", label: "Daily scoring" }, { value: "training", label: "Retraining" }]} />
            {mode === "scoring" ? (
              <Button variant="primary" busy={score.isPending} onClick={() => score.mutate()}>
                {!score.isPending ? <Play className="h-4 w-4" aria-hidden /> : null}{score.isPending ? "Scoring…" : "Run scoring now"}
              </Button>
            ) : (
              <Button variant="primary" busy={retrain.isPending || job.data?.status === "running" || job.data?.status === "queued"}
                disabled={config.data ? !config.data.training_api_enabled : false} onClick={() => retrain.mutate()}
                title={config.data && !config.data.training_api_enabled ? "Disabled on this server (ENABLE_TRAINING_API)" : undefined}>
                <Play className="h-4 w-4" aria-hidden />Start quick retrain
              </Button>
            )}
          </>
        }
      />
      {score.isError ? <ErrorState title="Scoring run failed" message={apiErrorMessage(score.error)} /> : null}
      {retrain.isError ? <ErrorState title="Could not start retraining" message={apiErrorMessage(retrain.error)} /> : null}
      {job.data ? (
        <div role="status" className={cn("rounded-xl border px-5 py-3.5 text-sm",
          job.data.status === "failed" ? "border-risk/30 bg-risk-bg text-risk-ink" : "border-cobalt/30 bg-cobalt-bg text-cobalt-ink")}>
          Retraining job {job.data.job_id.slice(0, 8)}: <strong className="font-medium">{job.data.status}</strong> · {job.data.message ?? job.data.current_step}
        </div>
      ) : null}

      <Card label="Stages" className="flex flex-col gap-5 px-6 py-6">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <CardHeader
            title={mode === "scoring" ? "Daily scoring" : "Retraining"}
            subtitle={mode === "scoring" ? "Meter reads in, ranked and explained cases out" : `Full dataset to a tuned, evaluated model · last run ${fmtDateTime(training.data?.trained_at)}`}
          />
          <span className="flex items-center gap-2 text-[13px] font-medium text-cobalt-ink">
            <span className={cn("h-2 w-2 rounded-full", mode === "scoring" && latest?.status === "failed" ? "bg-risk" : "bg-cobalt")} />
            {mode === "scoring"
              ? latest ? `Last run ${latest.status} · ${fmtSeconds(latest.seconds)} end to end` : "No run yet"
              : trainingStages.length ? `${fmtSeconds(total)} end to end` : "Stage timings not recorded"}
          </span>
        </div>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:flex lg:items-center">
          {stages.map((stage, i) => (
            <div key={`${mode}-${stage.name}`} className="flex flex-1 items-center gap-2">
              <StageCard index={i} name={stage.name} seconds={stage.seconds} share={Math.sqrt((stage.seconds ?? 0) / maxSeconds)}
                selected={i === selectedIndex} running={mode === "scoring" && score.isPending}
                failed={mode === "scoring" && latest?.status === "failed"}
                onClick={() => setPicked((p) => ({ ...p, [mode]: i }))} />
              {i < stages.length - 1 ? <ArrowRight className="hidden h-3.5 w-3.5 shrink-0 text-[#9AA0AA] lg:block" aria-hidden /> : null}
            </div>
          ))}
          {stages.length === 0 ? <Skeleton className="h-[124px] w-full" /> : null}
        </div>
      </Card>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1.25fr_1fr]">
        <Card label="Stage detail" className="flex flex-col gap-4 p-[22px]">
          <CardHeader title={stages[selectedIndex]?.name ?? "—"} action={<span className="font-mono text-[13px] text-ink-2">{fmtSeconds(stages[selectedIndex]?.seconds)}</span>} />
          <p className="m-0 text-sm leading-relaxed text-ink-2">
            {mode === "scoring" ? SCORING_ABOUT[scoringDetail?.key ?? ""]?.about : TRAINING_ABOUT[trainingDetail?.name ?? ""]}
          </p>
          {mode === "scoring" ? (
            <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
              <div className="flex flex-col gap-1 rounded-lg bg-surface-alt px-3.5 py-3"><span className="text-[11px] uppercase tracking-[0.08em] text-ink-3">Input</span><span className="text-[13px]">{SCORING_ABOUT[scoringDetail?.key ?? ""]?.input}</span></div>
              <div className="flex flex-col gap-1 rounded-lg bg-surface-alt px-3.5 py-3"><span className="text-[11px] uppercase tracking-[0.08em] text-ink-3">Result of last run</span><span className="text-[13px]">{scoringDetail?.detail || "—"}</span></div>
            </div>
          ) : (
            <div className="rounded-lg bg-night px-4 py-3.5 font-mono text-xs leading-relaxed text-night-text">
              {training.data ? (
                <>
                  <div>model v{training.data.model_version} · {training.data.n_trials} trials · CV {training.data.cv_metric} {fmtNum(training.data.cv_best_score)}</div>
                  <div>train {fmtInt(training.data.train_customers)} · test {fmtInt(training.data.test_customers)} customers · {training.data.n_features} features</div>
                  <div>test ROC-AUC {fmtNum(training.data.auc)} · PR-AUC {fmtNum(training.data.pr_auc)} · F1 {fmtNum(training.data.f1)} @ τ {fmtNum(training.data.threshold)}</div>
                  <div className="mt-2 text-night-muted">best params: {Object.entries(training.data.best_params).map(([k, v]) => `${k}=${typeof v === "number" ? +v.toPrecision(3) : v}`).join(", ")}</div>
                </>
              ) : "…"}
            </div>
          )}
        </Card>

        <div className="flex flex-col gap-4">
          <Card label="Automation" className="flex flex-col gap-3 p-[22px]">
            <CardHeader title="Automation" subtitle={config.data ? `Server environment: ${config.data.environment}` : undefined} />
            {[
              ["Score on server start", config.data?.run_on_startup ? "On" : "Off", "Warms caches and records a run"],
              ["Scheduled scoring", config.data ? (config.data.scoring_interval_minutes > 0 ? `Every ${config.data.scoring_interval_minutes} min` : "Off") : "…", "SCORING_INTERVAL_MINUTES"],
              ["Retraining from the dashboard", config.data ? (config.data.training_api_enabled ? "Allowed" : "Disabled") : "…", "ENABLE_TRAINING_API"],
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
                      <td className="capitalize text-ink-2">{run.trigger}</td>
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
