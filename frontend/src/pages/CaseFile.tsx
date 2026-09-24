import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, FileDown } from "lucide-react";
import { createAndDownloadReport, getCase, getExplanation, getModelMetrics, getTimeseries, updateCase, type CaseStatus, type Reading } from "@/lib/api";
import { fmtDateTime, fmtInt } from "@/lib/format";
import { ApiError, Button, Card, CardHeader, Skeleton, StatusBadge, TierBadge } from "@/components/ui";
import { ConsumptionChart, Gauge, ShapWaterfall } from "@/components/charts";

function seriesFacts(points: Reading[]) {
  const observed = points.filter((p): p is { date: string; kwh: number } => p.kwh !== null);
  let longest = 0;
  let run = 0;
  points.forEach((p) => {
    run = p.kwh === null ? run + 1 : 0;
    longest = Math.max(longest, run);
  });
  const mean = observed.length ? observed.reduce((s, p) => s + p.kwh, 0) / observed.length : null;
  return {
    first: observed[0]?.date ?? null,
    last: observed[observed.length - 1]?.date ?? null,
    coverage: points.length ? observed.length / points.length : 0,
    longestGap: longest,
    zeroDays: observed.filter((p) => p.kwh === 0).length,
    mean,
  };
}

const ACTIONS: Array<{ status: CaseStatus; label: string; primary?: boolean }> = [
  { status: "reviewing", label: "Start review" },
  { status: "dispatched", label: "Dispatch field inspection", primary: true },
  { status: "confirmed", label: "Confirm theft" },
  { status: "cleared", label: "Clear customer" },
];

export function CaseFilePage() {
  const { customerId = "" } = useParams();
  const queryClient = useQueryClient();
  const caseQuery = useQuery({ queryKey: ["case", customerId], queryFn: () => getCase(customerId) });
  const metrics = useQuery({ queryKey: ["model-metrics"], queryFn: getModelMetrics });
  const threshold = metrics.data?.threshold ?? 0.5;
  const explanation = useQuery({ queryKey: ["explanation", customerId], queryFn: () => getExplanation(customerId), enabled: caseQuery.isSuccess });
  const series = useQuery({ queryKey: ["timeseries", customerId], queryFn: () => getTimeseries(customerId), enabled: caseQuery.isSuccess });
  const [note, setNote] = useState("");

  useEffect(() => { setNote(caseQuery.data?.note ?? ""); }, [caseQuery.data?.note]);

  const save = useMutation({
    mutationFn: (payload: { status?: CaseStatus; note?: string }) => updateCase(customerId, payload),
    onSuccess: (data) => {
      queryClient.setQueryData(["case", customerId], data);
      queryClient.invalidateQueries({ queryKey: ["cases"] });
    },
  });

  const report = useMutation({
    mutationFn: () => createAndDownloadReport({ kind: "case", customer_id: customerId }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["reports"] }),
  });

  const facts = useMemo(() => (series.data ? seriesFacts(series.data.points) : null), [series.data]);

  if (caseQuery.isError) {
    return (
      <>
        <Link to="/cases" className="flex items-center gap-2 text-[13px] text-cobalt"><ArrowLeft className="h-4 w-4" aria-hidden />Case files</Link>
        <ApiError title="Customer not found" error={caseQuery.error} />
      </>
    );
  }

  const c = caseQuery.data;
  const contributions = explanation.data?.contributions ?? [];
  const top = contributions[0];

  return (
    <>
      <nav aria-label="Breadcrumb" className="flex items-center gap-2 text-[13px] text-ink-3">
        <Link to="/cases" className="flex items-center gap-1.5 text-cobalt hover:text-cobalt-ink"><ArrowLeft className="h-4 w-4" aria-hidden />Case files</Link>
        <span aria-hidden>/</span>
        <span>{c ? (c.flagged ? `Rank ${c.rank} of ${fmtInt(c.population)}` : "Not flagged") : "…"}</span>
      </nav>

      <header className="flex flex-wrap items-start justify-between gap-6">
        <div className="flex min-w-0 max-w-3xl flex-col gap-2.5">
          <div className="flex flex-wrap items-center gap-2.5">
            {c ? <TierBadge tier={c.risk_tier} solid={c.risk_tier === "high"} /> : null}
            {c ? <StatusBadge status={c.status} /> : null}
          </div>
          <h1 className="m-0 break-all font-mono text-[26px] font-medium tracking-[-0.01em]">{customerId}</h1>
          <p className="m-0 text-[15px] leading-relaxed text-ink-2">
            {facts && top
              ? <>Readings cover {Math.round(facts.coverage * 100)}% of days{facts.last ? `, the last on ${facts.last}` : ""}; longest gap {fmtInt(facts.longestGap)} days. Strongest signal: <strong className="font-medium text-ink">{top.label.toLowerCase()}</strong> ({top.display_value}).</>
              : "Loading the customer's history and the model's reasons…"}
          </p>
        </div>
        <div className="flex flex-wrap gap-2.5">
          <Button variant="ghost" disabled={!c} busy={report.isPending} onClick={() => report.mutate()}>
            <FileDown className="h-4 w-4" aria-hidden />Case report (PDF)
          </Button>
          {ACTIONS.map((action) => (
            <Button key={action.status} variant={action.primary ? "primary" : "secondary"} disabled={!c || c.status === action.status}
              busy={save.isPending && save.variables?.status === action.status} onClick={() => save.mutate({ status: action.status })}>
              {action.label}
            </Button>
          ))}
        </div>
      </header>
      {save.isError ? <ApiError title="Could not update the case" error={save.error} /> : null}
      {report.isError ? <ApiError title="Could not produce the case report" error={report.error} /> : null}

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[300px_1fr]">
        <Card label="Theft probability" className="flex flex-col items-center gap-3 p-[22px]">
          <span className="self-start text-[13px] text-ink-2">Theft probability</span>
          {c ? <Gauge value={c.risk_score} threshold={threshold} /> : <Skeleton className="h-[136px] w-[240px]" />}
          <span className="text-xs text-ink-3">{c ? `${c.flagged ? `Rank ${c.rank}` : "Below threshold"} · τ ${threshold.toFixed(3)}` : "…"}</span>
          <dl className="m-0 flex w-full flex-col gap-2 border-t border-line-soft pt-3 text-[13px]">
            {[
              ["First reading", facts?.first ?? "—"],
              ["Last reading", facts?.last ?? "—"],
              ["Longest gap", facts ? `${fmtInt(facts.longestGap)} days` : "—"],
              ["Zero-use days", facts ? fmtInt(facts.zeroDays) : "—"],
              ["Use while reporting", facts?.mean != null ? `${facts.mean.toFixed(2)} kWh/day` : "—"],
              ["Dataset label", series.data ? (series.data.label === 1 ? "theft" : "honest") : "—"],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between gap-3">
                <dt className="text-ink-2">{k}</dt>
                <dd className={`m-0 font-mono ${k === "Dataset label" && v === "theft" ? "text-risk-text" : ""}`}>{v}</dd>
              </div>
            ))}
          </dl>
        </Card>

        <Card label="Consumption history" className="flex flex-col gap-3.5 p-[22px]">
          <CardHeader title="Consumption history" subtitle="Hover to read a month or day; shaded stretches had no meter readings." />
          {series.data ? <ConsumptionChart points={series.data.points} /> : <Skeleton className="h-[280px]" />}
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1.4fr_1fr]">
        <Card label="Why the model flagged this customer" className="flex flex-col gap-4 p-[22px]">
          <CardHeader title="Why it was flagged" subtitle="SHAP contributions in log-odds: red pushes towards theft, blue away from it." />
          {explanation.data ? (
            <ShapWaterfall baseValue={explanation.data.base_value} probability={explanation.data.probability} reasons={contributions.slice(0, 5)} featureCount={contributions.length} />
          ) : explanation.isError ? <ApiError error={explanation.error} /> : <Skeleton className="h-64" />}
        </Card>

        <Card label="Case activity" className="flex flex-col gap-4 p-[22px]">
          <CardHeader title="Case activity" />
          <ol className="m-0 flex list-none flex-col gap-3.5 p-0">
            {(c?.history ?? []).slice().reverse().map((event, i) => (
              <li key={`${event.at}-${i}`} className="flex gap-3">
                <span className={`mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full ${i === 0 ? "bg-risk" : "bg-cobalt"}`} />
                <span className="flex flex-col gap-0.5">
                  <span className="text-sm font-medium">{event.event}</span>
                  <span className="text-xs text-ink-3">{fmtDateTime(event.at)}</span>
                </span>
              </li>
            ))}
            {c && c.history.length === 0 ? <li className="text-sm text-ink-3">No activity yet. This customer is below the threshold and has no open case.</li> : null}
          </ol>
          <form className="mt-auto flex flex-col gap-2" onSubmit={(e) => { e.preventDefault(); save.mutate({ note }); }}>
            <label htmlFor="case-note" className="text-[13px] text-ink-2">Analyst note</label>
            <textarea id="case-note" rows={3} value={note} onChange={(e) => setNote(e.target.value)} maxLength={4000} placeholder="Add context for the field team"
              className="resize-none rounded-lg border border-line px-3 py-2.5 text-sm outline-none focus:border-cobalt" />
            <Button type="submit" variant="secondary" className="self-end" disabled={!c || note === c.note} busy={save.isPending && save.variables?.note !== undefined}>Save note</Button>
          </form>
        </Card>
      </div>
    </>
  );
}

