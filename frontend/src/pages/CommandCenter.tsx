import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowRight, CheckCircle2, Play, Search, XCircle } from "lucide-react";
import { getCases, getModelMetrics, getPipelineRuns, getPopulationDistribution, getThresholdPreview, startPipelineRun, type Tier } from "@/lib/api";
import { fmtInt, fmtNum, fmtPct, fmtRelative, fmtSeconds, shortId } from "@/lib/format";
import { ApiError, Button, Card, CardHeader, PageHeader, Pill, Skeleton, Stat, StatusBadge, TierBadge } from "@/components/ui";
import { Legend, RiskHistogram } from "@/components/charts";

export function CommandCenterPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [tier, setTier] = useState<Tier | undefined>(undefined);
  const [search, setSearch] = useState("");

  const metrics = useQuery({ queryKey: ["model-metrics"], queryFn: getModelMetrics });
  const dist = useQuery({ queryKey: ["score-distribution"], queryFn: getPopulationDistribution });
  const runs = useQuery({ queryKey: ["pipeline-runs"], queryFn: () => getPipelineRuns(1) });
  const queue = useQuery({ queryKey: ["cases", "queue", tier], queryFn: () => getCases({ tier, page_size: 9 }) });
  const threshold = metrics.data?.threshold;
  const preview = useQuery({
    queryKey: ["threshold-preview", threshold],
    queryFn: () => getThresholdPreview(threshold as number),
    enabled: threshold !== undefined,
  });

  const run = useMutation({
    mutationFn: startPipelineRun,
    onSuccess: () => queryClient.invalidateQueries(),
  });

  if (metrics.isError) {
    return <ApiError error={metrics.error} />;
  }

  const m = metrics.data;
  const lastRun = runs.data?.[0];
  const expectedHitRate = m && m.flagged ? m.expected_thefts_flagged / m.flagged : undefined;

  return (
    <>
      <PageHeader
        eyebrow={m ? `${m.population.source === "sample" ? "SGCC sample of unseen customers" : `Uploaded readings · ${m.population.filename}`} · ${fmtInt(m.customers_monitored)} customers` : "Loading…"}
        title="Command center"
        actions={
          <>
            <form role="search" onSubmit={(event) => { event.preventDefault(); if (search.trim()) navigate(`/cases/${encodeURIComponent(search.trim())}`); }}
              className="flex h-11 w-[280px] items-center gap-2 rounded-lg border border-line bg-surface px-3.5">
              <Search className="h-4 w-4 text-ink-3" aria-hidden />
              <label htmlFor="cc-search" className="sr-only">Open a customer by ID</label>
              <input id="cc-search" type="search" value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Open customer by ID"
                className="min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-ink-3" />
            </form>
            <Button variant="primary" busy={run.isPending} onClick={() => run.mutate()}>
              {!run.isPending ? <Play className="h-4 w-4" aria-hidden /> : null}
              {run.isPending ? "Scoring…" : "Run scoring"}
            </Button>
          </>
        }
      />
      {run.isError ? <ApiError title="Scoring run failed" error={run.error} /> : null}

      <section aria-label="Key figures" className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {m ? (
          <>
            <Stat label="Customers monitored" value={fmtInt(m.customers_monitored)} hint={`${m.pipeline_label} · v${m.model_version}`} />
            <Stat label="Flagged for inspection" tone="risk" value={fmtInt(m.flagged)} hint={`${fmtPct(m.flagged / m.customers_monitored, 1)} of customers at τ = ${m.threshold.toFixed(3)}`} />
            <Stat label="Expected thefts among flagged" value={fmtNum(m.expected_thefts_flagged, 0)}
              hint={`Estimate: sum of calibrated probabilities · ${fmtPct(expectedHitRate)} of visits`} />
            <Stat label="Expected thefts, all customers" value={fmtNum(m.expected_thefts_total, 0)}
              hint={`Estimate · ${fmtPct(m.expected_thefts_total / m.customers_monitored, 1)} of the population`} />
          </>
        ) : Array.from({ length: 4 }, (_, i) => <Skeleton key={i} className="h-[132px]" />)}
      </section>

      <Card label="Scoring pipeline" className="flex flex-col gap-3.5 px-[22px] py-[18px]">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <Link to="/pipeline" className="text-[13px] font-semibold hover:text-cobalt">Scoring pipeline</Link>
          <span className="text-xs text-ink-3">
            {lastRun ? `${lastRun.trigger} run ${fmtRelative(lastRun.finished_at)} · ${fmtSeconds(lastRun.seconds)}` : "No run recorded yet"}
          </span>
        </div>
        <ol className="m-0 grid flex-1 list-none grid-cols-1 gap-2.5 p-0 sm:grid-cols-5">
          {(lastRun?.stages ?? []).map((stage) => (
            <li key={stage.key} className="flex min-w-0 items-center gap-2.5 rounded-lg border border-line bg-[#FAFAF8] px-3 py-2.5">
              {lastRun?.status === "failed" ? <XCircle className="h-[18px] w-[18px] shrink-0 text-risk" aria-hidden /> : <CheckCircle2 className="h-[18px] w-[18px] shrink-0 text-cobalt" aria-hidden />}
              <div className="flex min-w-0 flex-col">
                <span className="flex items-baseline justify-between gap-2 text-[13px] font-medium">{stage.name}<span className="font-mono text-[11px] font-normal text-ink-3">{fmtSeconds(stage.seconds)}</span></span>
                <span className="truncate text-[12px] text-ink-2" title={stage.detail}>{stage.detail}</span>
              </div>
            </li>
          ))}
          {!lastRun && runs.isLoading ? Array.from({ length: 5 }, (_, i) => <Skeleton key={i} className="h-[54px]" />) : null}
        </ol>
      </Card>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1.55fr_1fr]">
        <Card label="Investigation queue" className="flex flex-col overflow-hidden">
          <div className="flex flex-wrap items-start justify-between gap-3 px-[22px] pb-3.5 pt-5">
            <CardHeader title="Investigation queue" subtitle={m ? `Ranked by theft probability · ${m.risk_tier_distribution.high} high, ${m.risk_tier_distribution.medium} medium` : "…"} />
            <div role="group" aria-label="Filter by tier" className="flex gap-1.5">
              <Pill active={!tier} onClick={() => setTier(undefined)}>All flagged</Pill>
              <Pill active={tier === "high"} onClick={() => setTier("high")}>High</Pill>
              <Pill active={tier === "medium"} onClick={() => setTier("medium")}>Medium</Pill>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[640px] border-collapse text-left">
              <thead>
                <tr className="border-y border-line-soft text-[11px] uppercase tracking-[0.08em] text-ink-3">
                  <th className="w-12 py-2.5 pl-[22px] font-normal">#</th>
                  <th className="py-2.5 font-normal">Customer</th>
                  <th className="py-2.5 font-normal">Risk</th>
                  <th className="whitespace-nowrap py-2.5 font-normal">Strongest signal</th>
                  <th className="py-2.5 pr-[22px] font-normal">Status</th>
                </tr>
              </thead>
              <tbody>
                {queue.data?.items.map((item) => (
                  <tr key={item.customer_id} className="group h-[52px] cursor-pointer border-b border-[#F1EFEA] hover:bg-surface-alt" onClick={() => navigate(`/cases/${item.customer_id}`)}>
                    <td className="pl-[22px] font-mono text-xs text-ink-3">{item.rank}</td>
                    <td className="font-mono text-[13px]">
                      <Link to={`/cases/${item.customer_id}`} className="hover:text-cobalt" onClick={(e) => e.stopPropagation()}>{shortId(item.customer_id)}</Link>
                    </td>
                    <td className="pr-4">
                      <span className="flex items-center gap-2.5">
                        <span className="hidden h-1.5 w-20 overflow-hidden rounded-full bg-tint 2xl:flex">
                          <span className="block h-full" style={{ width: `${item.risk_score * 100}%`, background: item.risk_tier === "high" ? "#B93C15" : "#E0873A" }} />
                        </span>
                        <span className="w-10 text-right font-mono text-[13px] tabular">{item.risk_score.toFixed(3)}</span>
                        <TierBadge tier={item.risk_tier} />
                      </span>
                    </td>
                    <td className="max-w-[240px] truncate whitespace-nowrap pr-3 text-[13px] text-ink-2" title={item.top_driver?.label}>{item.top_driver?.label ?? "—"}</td>
                    <td className="pr-[22px]"><StatusBadge status={item.status} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
            {queue.isLoading ? <div className="flex flex-col gap-2 p-4">{Array.from({ length: 6 }, (_, i) => <Skeleton key={i} className="h-10" />)}</div> : null}
          </div>
          <Link to="/cases" className="mt-auto flex items-center justify-end gap-1.5 px-[22px] py-3.5 text-[13px] font-medium text-cobalt hover:text-cobalt-ink">
            All {queue.data ? fmtInt(queue.data.total) : ""} cases <ArrowRight className="h-4 w-4" aria-hidden />
          </Link>
        </Card>

        <div className="flex flex-col gap-4">
          <Card label="Risk distribution" className="flex flex-col gap-3.5 px-[22px] py-5">
            <CardHeader title="Risk distribution" action={<span className="text-xs text-ink-3">log scale</span>} />
            {dist.data ? <RiskHistogram edges={dist.data.edges} total={dist.data.counts} threshold={dist.data.threshold} /> : <Skeleton className="h-[170px]" />}
            <Legend items={[{ color: "#E0873A", label: "At or above τ" }, { color: "#C4C9D2", label: "Below τ" }]} />
          </Card>
          <Card label="What the threshold means" className="flex flex-1 flex-col gap-3.5 px-[22px] py-5">
            <CardHeader title={m ? `What τ ${m.threshold.toFixed(3)} means` : "What the threshold means"}
              subtitle={preview.data ? `On the ${fmtInt(preview.data.validation.customers)} validation customers the threshold was chosen on` : undefined}
              action={<Link to="/threshold" className="text-[13px] font-medium text-cobalt hover:text-cobalt-ink">Tune threshold</Link>} />
            {preview.data ? (
              <dl className="grid grid-cols-2 gap-3 text-[13px]">
                {[
                  ["Precision", fmtPct(preview.data.validation.precision), "of flagged customers were thieves"],
                  ["Recall", fmtPct(preview.data.validation.recall), "of thieves were flagged"],
                ].map(([label, value, hint]) => (
                  <div key={label} className="flex flex-col gap-0.5 rounded-lg bg-tint p-4">
                    <dt className="text-ink-3">{label}</dt>
                    <dd className="font-display text-3xl tabular">{value}</dd>
                    <dd className="text-xs text-ink-2">{hint}</dd>
                  </div>
                ))}
              </dl>
            ) : <Skeleton className="h-40" />}
            <p className="text-xs text-ink-3">A flag is a reason to inspect, not evidence of theft. <Link to="/research" className="text-cobalt">Test-set evaluation</Link></p>
          </Card>
        </div>
      </div>
    </>
  );
}
