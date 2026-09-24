import { useRef, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Download, FileUp, Sparkles } from "lucide-react";
import {
  apiErrorMessage,
  downloadBatchPredictions,
  downloadReport,
  generateReport,
  getDatasetCatalog,
  getLatestReports,
  getModelMetrics,
  predictSingle,
  uploadDataset,
} from "@/lib/api";
import { featureLabel, featureValue } from "@/lib/features";
import { fmtDateTime, fmtInt, fmtNum, fmtPct } from "@/lib/format";
import { Button, Card, CardHeader, Empty, ErrorState, PageHeader, TierBadge } from "@/components/ui";

function FilePicker({ id, label, onPick, busy }: { id: string; label: string; onPick: (file: File) => void; busy?: boolean }) {
  const input = useRef<HTMLInputElement>(null);
  return (
    <>
      <input ref={input} id={id} type="file" accept=".csv,text/csv" className="sr-only"
        onChange={(e) => { const file = e.target.files?.[0]; if (file) onPick(file); e.target.value = ""; }} />
      <Button variant="secondary" busy={busy} onClick={() => input.current?.click()}>
        {!busy ? <FileUp className="h-4 w-4" aria-hidden /> : null}{label}
      </Button>
    </>
  );
}

export function ScoringPage() {
  const queryClient = useQueryClient();
  const [customerId, setCustomerId] = useState("");
  const metrics = useQuery({ queryKey: ["model-metrics"], queryFn: getModelMetrics });
  const reports = useQuery({ queryKey: ["reports"], queryFn: getLatestReports });
  const catalog = useQuery({ queryKey: ["dataset-catalog"], queryFn: getDatasetCatalog });

  const score = useMutation({ mutationFn: (id: string) => predictSingle({ customer_id: id }) });
  const batch = useMutation({ mutationFn: downloadBatchPredictions });
  const upload = useMutation({ mutationFn: uploadDataset, onSuccess: () => queryClient.invalidateQueries({ queryKey: ["dataset-catalog"] }) });
  const report = useMutation({
    mutationFn: (datasetId?: string) => generateReport(datasetId ? { dataset_id: datasetId } : {}),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["reports"] }),
  });
  const download = useMutation({ mutationFn: downloadReport });

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (customerId.trim()) score.mutate(customerId.trim());
  };
  const result = score.data;
  const uploaded = upload.data?.item;
  const summary = uploaded?.summary as Record<string, number | string | null | string[]> | undefined;

  return (
    <>
      <PageHeader eyebrow="Model as a service" title="Scoring & reports"
        description="Score a single customer, score a file of feature rows, or check a new dataset against the model and export the result as a PDF." />

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Card label="Score a customer" className="flex flex-col gap-4 p-[22px]">
          <CardHeader title="Score a customer" subtitle={`Uses the threshold in service (τ ${metrics.data ? metrics.data.threshold.toFixed(3) : "…"}).`} />
          <form onSubmit={submit} className="flex gap-2.5">
            <label htmlFor="score-id" className="sr-only">Customer ID</label>
            <input id="score-id" value={customerId} onChange={(e) => setCustomerId(e.target.value)} placeholder="Customer ID, e.g. 8E1EBF0A95AAF71D4CAA7032326A3474"
              className="h-11 min-w-0 flex-1 rounded-lg border border-line px-3.5 font-mono text-sm outline-none focus:border-cobalt" />
            <Button type="submit" variant="primary" busy={score.isPending}><Sparkles className="h-4 w-4" aria-hidden />Score</Button>
          </form>
          {score.isError ? <ErrorState title="Could not score" message={apiErrorMessage(score.error)} /> : null}
          {result ? (
            <div className="flex flex-col gap-3 rounded-xl bg-surface-alt p-4">
              <div className="flex flex-wrap items-center gap-3">
                <span className="font-display text-4xl tabular">{fmtNum(result.probability)}</span>
                {result.risk_tier ? <TierBadge tier={result.risk_tier} solid={result.risk_tier === "high"} /> : null}
                <span className="text-sm text-ink-2">{result.prediction ? "Flagged for inspection" : "Below threshold"}</span>
                <Link to={`/cases/${result.customer_id}`} className="ml-auto text-[13px] font-medium text-cobalt">Open case file</Link>
              </div>
              <ul className="m-0 flex list-none flex-col gap-1.5 p-0 text-[13px]">
                {result.top_reasons.map((r) => (
                  <li key={r.feature} className="flex justify-between gap-3">
                    <span>{featureLabel(r.feature)} <span className="text-ink-3">· {featureValue(r.feature, r.value)}</span></span>
                    <span className={`font-mono ${r.shap_value >= 0 ? "text-risk-text" : "text-cobalt-ink"}`}>{r.shap_value >= 0 ? "+" : "−"}{Math.abs(r.shap_value).toFixed(2)}</span>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
        </Card>

        <Card label="Batch scoring" className="flex flex-col gap-4 p-[22px]">
          <CardHeader title="Batch scoring" subtitle="Upload a CSV whose columns are model features. Missing columns count as missing readings." />
          <div className="flex flex-wrap items-center gap-2.5">
            <FilePicker id="batch-file" label="Score a CSV" busy={batch.isPending} onPick={(file) => batch.mutate(file)} />
            <span className="text-[13px] text-ink-3">Returns the rows with probability and prediction added.</span>
          </div>
          {batch.isSuccess ? <p role="status" className="m-0 text-[13px] text-cobalt-ink">predictions.csv downloaded.</p> : null}
          {batch.isError ? <ErrorState title="Batch scoring failed" message={apiErrorMessage(batch.error)} /> : null}
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1.3fr_1fr]">
        <Card label="Dataset verification" className="flex flex-col gap-4 p-[22px]">
          <CardHeader title="Verify a dataset" subtitle="Upload feature rows (optionally with a label column) to see how the model scores them."
            action={<FilePicker id="verify-file" label="Upload CSV" busy={upload.isPending} onPick={(file) => upload.mutate(file)} />} />
          {upload.isError ? <ErrorState title="Upload failed" message={apiErrorMessage(upload.error)} /> : null}
          {uploaded && summary ? (
            <div className="flex flex-col gap-3">
              <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
                {[
                  ["Rows", fmtInt(summary.rows as number)],
                  ["Flagged", fmtInt(summary.predicted_positive_rows as number)],
                  ["Mean probability", fmtNum(summary.mean_probability as number)],
                  ["Label agreement", summary.verification_accuracy == null ? "no labels" : fmtPct(summary.verification_accuracy as number, 1)],
                ].map(([k, v]) => (
                  <div key={k} className="flex flex-col gap-1 rounded-lg bg-surface-alt px-3.5 py-3"><span className="text-xs text-ink-3">{k}</span><span className="font-mono text-base">{v}</span></div>
                ))}
              </div>
              {(summary.missing_features as string[])?.length ? (
                <p className="m-0 text-[13px] text-amber-ink">{(summary.missing_features as string[]).length} model features were missing and treated as missing readings.</p>
              ) : null}
              <Button variant="primary" className="self-start" busy={report.isPending} onClick={() => report.mutate(uploaded.dataset_id)}>Generate PDF report</Button>
            </div>
          ) : (
            <div className="flex flex-col gap-2 text-[13px] text-ink-2">
              <span>Recent uploads</span>
              {catalog.data?.items.length ? catalog.data.items.slice(0, 4).map((item) => (
                <div key={item.dataset_id} className="flex items-center justify-between gap-3 border-t border-line-soft pt-2">
                  <span className="truncate">{item.original_filename} <span className="text-ink-3">· {fmtInt(item.rows)} rows</span></span>
                  <Button variant="ghost" busy={report.isPending && report.variables === item.dataset_id} onClick={() => report.mutate(item.dataset_id)}>Report</Button>
                </div>
              )) : <Empty title="No uploads yet" />}
            </div>
          )}
          {report.isError ? <ErrorState title="Report failed" message={apiErrorMessage(report.error)} /> : null}
        </Card>

        <Card label="Reports" className="flex flex-col gap-4 p-[22px]">
          <CardHeader title="Reports" action={<Button variant="secondary" busy={report.isPending && report.variables === undefined} onClick={() => report.mutate(undefined)}>Portfolio report</Button>} />
          {reports.data?.length ? (
            <ul className="m-0 flex list-none flex-col p-0">
              {reports.data.map((r) => (
                <li key={r.report_id} className="flex items-center justify-between gap-3 border-b border-line-soft py-2.5 text-[13px]">
                  <span className="flex min-w-0 flex-col"><span className="truncate font-medium">{r.dataset_label}</span><span className="text-xs text-ink-3">{fmtDateTime(r.generated_at)}</span></span>
                  <Button variant="ghost" aria-label={`Download ${r.report_id}`} busy={download.isPending && download.variables === r.report_id} onClick={() => download.mutate(r.report_id)}>
                    <Download className="h-4 w-4" aria-hidden />PDF
                  </Button>
                </li>
              ))}
            </ul>
          ) : <Empty title="No reports yet" message="Generate one from a dataset or the whole portfolio." />}
          {download.isError ? <ErrorState title="Download failed" message={apiErrorMessage(download.error)} /> : null}
        </Card>
      </div>
    </>
  );
}
