import { useRef, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Download, FileDown, FileUp, Sparkles } from "lucide-react";
import {
  createAndDownloadReport,
  downloadBatchScores,
  downloadReport,
  getDatasets,
  getHealth,
  getModelMetrics,
  getReports,
  predictCustomer,
  uploadDataset,
  type Dataset,
  type ReportKind,
} from "@/lib/api";
import { fmtDateTime, fmtInt, fmtNum, fmtPct } from "@/lib/format";
import { ApiError, Button, Card, CardHeader, Empty, ErrorState, PageHeader, Skeleton, TierBadge } from "@/components/ui";

const KIND_LABEL: Record<ReportKind, string> = { portfolio: "Portfolio", dataset: "Dataset", case: "Case file" };

function FilePicker({ id, label, onPick, busy }: { id: string; label: string; onPick: (file: File) => void; busy?: boolean }) {
  const input = useRef<HTMLInputElement>(null);
  return (
    <>
      <input ref={input} id={id} type="file" accept=".csv,text/csv" className="sr-only" tabIndex={-1} aria-hidden
        onChange={(e) => { const file = e.target.files?.[0]; if (file) onPick(file); e.target.value = ""; }} />
      <Button variant="secondary" busy={busy} onClick={() => input.current?.click()}>
        {!busy ? <FileUp className="h-4 w-4" aria-hidden /> : null}{label}
      </Button>
    </>
  );
}

function Figure({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-1 rounded-lg bg-surface-alt px-3.5 py-3">
      <span className="text-xs text-ink-3">{label}</span>
      <span className="font-mono text-base tabular">{value}</span>
    </div>
  );
}

function DatasetResult({ dataset }: { dataset: Dataset }) {
  const s = dataset.summary;
  const lm = s.label_metrics;
  return (
    <div className="flex flex-col gap-3">
      <p className="m-0 text-[13px] text-ink-2">
        <span className="font-medium text-ink">{dataset.filename}</span> · {s.format === "consumption"
          ? `SGCC meter data, ${fmtInt(s.days)} days per customer; features built by the pipeline`
          : `feature rows, ${s.features_found} of ${s.features_expected} model features present`} · scored at τ {s.threshold.toFixed(3)}
      </p>
      <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
        <Figure label="Customers" value={fmtInt(s.customers)} />
        <Figure label="Flagged" value={`${fmtInt(s.flagged)} (${fmtPct(s.customers ? s.flagged / s.customers : 0, 1)})`} />
        <Figure label="High / medium risk" value={`${fmtInt(s.tiers.high)} / ${fmtInt(s.tiers.medium)}`} />
        <Figure label="Mean probability" value={fmtNum(s.mean_probability)} />
      </div>
      {lm ? (
        <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
          <Figure label="Thefts caught" value={`${fmtInt(lm.caught)} of ${fmtInt(lm.theft)}`} />
          <Figure label="Hit rate (precision)" value={lm.precision === null ? "—" : fmtPct(lm.precision, 1)} />
          <Figure label="ROC-AUC" value={fmtNum(lm.roc_auc)} />
          <Figure label="PR-AUC" value={fmtNum(lm.pr_auc)} />
        </div>
      ) : <p className="m-0 text-[13px] text-ink-3">No FLAG column, so there is no ground truth to check the scores against.</p>}
      {s.format === "features" && s.features_found < s.features_expected ? (
        <p className="m-0 text-[13px] text-amber-ink">{s.features_expected - s.features_found} model features were absent and treated as missing readings.</p>
      ) : null}
      <table className="w-full border-collapse text-left text-[13px]">
        <caption className="pb-2 text-left text-xs text-ink-3">Highest-risk customers in the file</caption>
        <thead>
          <tr className="border-b border-line-soft text-[11px] uppercase tracking-[0.08em] text-ink-3">
            <th className="pb-2 font-normal">Customer</th><th className="pb-2 font-normal">Tier</th><th className="pb-2 text-right font-normal">Probability</th>{s.labelled ? <th className="pb-2 text-right font-normal">Label</th> : null}
          </tr>
        </thead>
        <tbody>
          {s.top.slice(0, 5).map((row) => (
            <tr key={row.customer_id} className="h-9 border-b border-[#F1EFEA]">
              <td className="max-w-0 truncate font-mono text-xs" title={row.customer_id}>{row.customer_id}</td>
              <td><TierBadge tier={row.risk_tier} /></td>
              <td className="text-right font-mono tabular">{fmtNum(row.probability)}</td>
              {s.labelled ? <td className={`text-right ${row.label === 1 ? "text-risk-text" : "text-ink-2"}`}>{row.label === 1 ? "theft" : "honest"}</td> : null}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ReportsPage() {
  const queryClient = useQueryClient();
  const [customerId, setCustomerId] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const health = useQuery({ queryKey: ["health"], queryFn: getHealth, retry: false });
  const metrics = useQuery({ queryKey: ["model-metrics"], queryFn: getModelMetrics });
  const reports = useQuery({ queryKey: ["reports"], queryFn: getReports });
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: getDatasets });

  const refreshReports = () => queryClient.invalidateQueries({ queryKey: ["reports"] });
  const portfolio = useMutation({ mutationFn: () => createAndDownloadReport({ kind: "portfolio" }), onSettled: refreshReports });
  const datasetReport = useMutation({ mutationFn: (id: string) => createAndDownloadReport({ kind: "dataset", dataset_id: id }), onSettled: refreshReports });
  const download = useMutation({ mutationFn: downloadReport });
  const upload = useMutation({
    mutationFn: uploadDataset,
    onSuccess: (data) => {
      setSelectedId(data.dataset_id);
      void queryClient.invalidateQueries({ queryKey: ["datasets"] });
    },
  });
  const batch = useMutation({ mutationFn: downloadBatchScores });
  const score = useMutation({ mutationFn: predictCustomer });

  const reportsOff = health.data && !health.data.reports.available ? health.data.reports.detail : null;
  const selected = datasets.data?.find((d) => d.dataset_id === selectedId) ?? (selectedId === upload.data?.dataset_id ? upload.data : undefined);

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (customerId.trim()) score.mutate(customerId.trim());
  };

  return (
    <>
      <PageHeader eyebrow="Model as a service" title="Reports & scoring"
        description="Export PDF reports for the portfolio, an uploaded dataset or a single case; check a new file of meter data against the model; score customers on demand." />

      {reportsOff ? <ErrorState title="PDF reports are unavailable on this server" message={reportsOff} /> : null}

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1fr_1.2fr]">
        <Card label="Portfolio report" className="flex flex-col gap-4 p-[22px]">
          <CardHeader title="Portfolio report" subtitle="The served population at the threshold in service: model quality, risk tiers, the case workflow and the top flagged customers with their reasons." />
          <Button variant="primary" className="self-start" disabled={Boolean(reportsOff)} busy={portfolio.isPending} onClick={() => portfolio.mutate()}>
            {!portfolio.isPending ? <FileDown className="h-4 w-4" aria-hidden /> : null}Generate and download PDF
          </Button>
          {portfolio.isSuccess ? <p role="status" className="m-0 text-[13px] text-cobalt-ink">{portfolio.data.report_id}.pdf downloaded.</p> : null}
          {portfolio.isError ? <ApiError title="Could not produce the portfolio report" error={portfolio.error} /> : null}
          <p className="m-0 mt-auto text-[13px] text-ink-3">Case-file reports are on each customer's case page.</p>
        </Card>

        <Card label="Report archive" className="flex flex-col gap-3 p-[22px]">
          <CardHeader title="Report archive" subtitle="Every report generated on this server, newest first." />
          {reports.data?.length ? (
            <ul className="m-0 flex max-h-[300px] list-none flex-col overflow-y-auto p-0">
              {reports.data.map((r) => (
                <li key={r.report_id} className="flex items-center justify-between gap-3 border-b border-line-soft py-2.5 text-[13px]">
                  <span className="flex min-w-0 flex-col">
                    <span className="truncate font-medium">{KIND_LABEL[r.kind]} · <span className="font-normal text-ink-2">{r.subject}</span></span>
                    <span className="text-xs text-ink-3">{fmtDateTime(r.created_at)} · {fmtInt(Math.ceil(r.bytes / 1024))} KB</span>
                  </span>
                  <Button variant="ghost" aria-label={`Download ${r.report_id}`} busy={download.isPending && download.variables === r.report_id} onClick={() => download.mutate(r.report_id)}>
                    <Download className="h-4 w-4" aria-hidden />PDF
                  </Button>
                </li>
              ))}
            </ul>
          ) : reports.isLoading ? <Skeleton className="h-32" /> : reports.isError ? <ApiError error={reports.error} /> : <Empty title="No reports yet" message="Generate one here, or from a case file." />}
          {download.isError ? <ApiError title="Download failed" error={download.error} /> : null}
        </Card>
      </div>

      <Card label="Check a dataset" className="flex flex-col gap-4 p-[22px]">
        <CardHeader title="Check a dataset"
          subtitle="Upload SGCC meter data (CONS_NO, optional FLAG, one column per day) or rows of model features. The model scores every customer; with a FLAG column you also see how many thefts it catches."
          action={<FilePicker id="dataset-file" label="Upload CSV" busy={upload.isPending} onPick={(file) => upload.mutate(file)} />} />
        {upload.isError ? <ApiError title="Upload rejected" error={upload.error} /> : null}
        <div className="grid grid-cols-1 gap-5 xl:grid-cols-[1.6fr_1fr]">
          <div className="flex flex-col gap-3">
            {selected ? (
              <>
                <DatasetResult dataset={selected} />
                <Button variant="primary" className="self-start" disabled={Boolean(reportsOff)} busy={datasetReport.isPending} onClick={() => datasetReport.mutate(selected.dataset_id)}>
                  {!datasetReport.isPending ? <FileDown className="h-4 w-4" aria-hidden /> : null}Dataset report (PDF)
                </Button>
                {datasetReport.isError ? <ApiError title="Could not produce the dataset report" error={datasetReport.error} /> : null}
              </>
            ) : <Empty title="No dataset selected" message="Upload a CSV, or pick an earlier upload." />}
          </div>
          <div className="flex flex-col gap-2 text-[13px]">
            <span className="text-ink-2">Earlier uploads</span>
            {datasets.data?.length ? (
              <ul className="m-0 flex max-h-[360px] list-none flex-col overflow-y-auto p-0">
                {datasets.data.map((d) => (
                  <li key={d.dataset_id}>
                    <button type="button" aria-pressed={d.dataset_id === selectedId} onClick={() => setSelectedId(d.dataset_id)}
                      className={`flex w-full flex-col gap-0.5 border-t border-line-soft px-2 py-2 text-left hover:bg-tint ${d.dataset_id === selectedId ? "bg-cobalt-soft" : ""}`}>
                      <span className="truncate font-medium">{d.filename}</span>
                      <span className="text-xs text-ink-3">{fmtDateTime(d.uploaded_at)} · {fmtInt(d.summary.customers)} customers · {fmtInt(d.summary.flagged)} flagged</span>
                    </button>
                  </li>
                ))}
              </ul>
            ) : datasets.isLoading ? <Skeleton className="h-24" /> : datasets.isError ? <ApiError error={datasets.error} /> : <span className="text-ink-3">None yet.</span>}
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Card label="Score a customer" className="flex flex-col gap-4 p-[22px]">
          <CardHeader title="Score a customer" subtitle={`A served customer, at the threshold in service (τ ${metrics.data ? metrics.data.threshold.toFixed(3) : "…"}).`} />
          <form onSubmit={submit} className="flex gap-2.5">
            <label htmlFor="score-id" className="sr-only">Customer ID</label>
            <input id="score-id" value={customerId} onChange={(e) => setCustomerId(e.target.value)} placeholder="Customer ID" maxLength={64}
              className="h-11 min-w-0 flex-1 rounded-lg border border-line px-3.5 font-mono text-sm outline-none focus:border-cobalt" />
            <Button type="submit" variant="primary" disabled={!customerId.trim()} busy={score.isPending}><Sparkles className="h-4 w-4" aria-hidden />Score</Button>
          </form>
          {score.isError ? <ApiError title="Could not score" error={score.error} /> : null}
          {score.data ? (
            <div className="flex flex-col gap-3 rounded-xl bg-surface-alt p-4">
              <div className="flex flex-wrap items-center gap-3">
                <span className="font-display text-4xl tabular">{fmtNum(score.data.probability)}</span>
                <TierBadge tier={score.data.risk_tier} solid={score.data.risk_tier === "high"} />
                <span className="text-sm text-ink-2">{score.data.prediction ? "Flagged for inspection" : "Below threshold"}</span>
                {score.data.customer_id ? <Link to={`/cases/${encodeURIComponent(score.data.customer_id)}`} className="ml-auto text-[13px] font-medium text-cobalt">Open case file</Link> : null}
              </div>
              <ul className="m-0 flex list-none flex-col gap-1.5 p-0 text-[13px]">
                {score.data.reasons.map((r) => (
                  <li key={r.feature} className="flex justify-between gap-3">
                    <span>{r.label} <span className="text-ink-3">· {r.display_value}</span></span>
                    <span className={`font-mono ${r.shap_value >= 0 ? "text-risk-text" : "text-cobalt-ink"}`}>{r.shap_value >= 0 ? "+" : "−"}{Math.abs(r.shap_value).toFixed(2)}</span>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
        </Card>

        <Card label="Batch scoring" className="flex flex-col gap-4 p-[22px]">
          <CardHeader title="Batch scoring" subtitle="Score a CSV in the same formats as above and download one row per customer: probability, prediction and risk tier (plus the label when the file has one)." />
          <div className="flex flex-wrap items-center gap-2.5">
            <FilePicker id="batch-file" label="Score a CSV" busy={batch.isPending} onPick={(file) => batch.mutate(file)} />
            <span className="text-[13px] text-ink-3">Nothing is stored on the server.</span>
          </div>
          {batch.isSuccess ? <p role="status" className="m-0 text-[13px] text-cobalt-ink">Scores downloaded.</p> : null}
          {batch.isError ? <ApiError title="Batch scoring failed" error={batch.error} /> : null}
        </Card>
      </div>
    </>
  );
}
