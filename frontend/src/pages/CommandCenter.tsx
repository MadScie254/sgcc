import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowRight, CheckCircle2, Play, Search, XCircle } from "lucide-react";
import { getCases, getModelMetrics, getPipelineRuns, getScoreDistribution, startPipelineRun, type Tier } from "@/lib/api";
import { fmtInt, fmtNum, fmtPct, fmtRelative, fmtSeconds, shortId } from "@/lib/format";
import { ApiError, Button, Card, CardHeader, PageHeader, Pill, Skeleton, Stat, StatusBadge, TierBadge } from "@/components/ui";
import { ConfusionGrid, Legend, RiskHistogram } from "@/components/charts";

export function CommandCenterPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [tier, setTier] = useState<Tier | undefined>(undefined);
  const [search, setSearch] = useState("");

  const metrics = useQuery({ queryKey: ["model-metrics"], queryFn: getModelMetrics });
  const dist = useQuery({ queryKey: ["score-distribution"], queryFn: getScoreDistribution });
  const runs = useQuery({ queryKey: ["pipeline-runs"], queryFn: () => getPipelineRuns(1) });
  const queue = useQuery({ queryKey: ["cases", "queue", tier], queryFn: () => getCases({ tier, page_size: 9 }) });

  const run = useMutation({
    mutationFn: startPipelineRun,
    onSuccess: () => queryClient.invalidateQueries(),
  });

  if (metrics.isError) {
    return <ApiError error={metrics.error} />;
  }

  const m = metrics.data;
  const cm = m?.confusion_matrix;
  const hitRate = cm ? cm.tp / Math.max(cm.tp + cm.fp, 1) : undefined;
  const lastRun = runs.data?.[0];

  return (
    <>
      <PageHeader
        eyebrow={m ? `SGCC network · ${fmtInt(m.customers_monitored)} held-out customers · Jan 2014 – Oct 2016` : "Loading…"}
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
            <Stat label="Customers monitored" value={fmtInt(m.customers_monitored)} hint="1,034 days of meter reads each" />
            <Stat label="Flagged for inspection" tone="risk" value={fmtInt(m.flagged)} hint={`${fmtPct(m.flagged / m.customers_monitored, 1)} of customers at τ = ${m.threshold.toFixed(3)}`} />
            <Stat label="Hit rate of flags" value={fmtPct(hitRate)} hint={`vs ${fmtPct(m.base_rate, 1)} base rate · ${hitRate ? (hitRate / m.base_rate).toFixed(1) : "—"}× random`} />
            <Stat label="Model quality (hold-out)" value={fmtNum(m.metrics.auc)} hint={`ROC-AUC · PR-AUC ${fmtNum(m.metrics.pr_auc)}`} />
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
            {dist.data ? <RiskHistogram dist={dist.data} /> : <Skeleton className="h-[170px]" />}
            <Legend items={[{ color: "#C4C9D2", label: "Honest (label)" }, { color: "#B93C15", label: "Theft (label)" }]} />
          </Card>
          <Card label="Outcome at current threshold" className="flex flex-1 flex-col gap-3.5 px-[22px] py-5">
            <CardHeader title={m ? `Outcome at τ ${m.threshold.toFixed(3)}` : "Outcome"} action={<Link to="/threshold" className="text-[13px] font-medium text-cobalt hover:text-cobalt-ink">Tune threshold</Link>} />
            {cm ? <ConfusionGrid tp={cm.tp} fp={cm.fp} fn={cm.fn} tn={cm.tn} /> : <Skeleton className="h-40" />}
          </Card>
        </div>
      </div>
    </>
  );
}
