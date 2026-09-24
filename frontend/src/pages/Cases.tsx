import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, Search } from "lucide-react";
import { apiErrorMessage, getCases, type CaseStatus } from "@/lib/api";
import { featureLabel, featureValue } from "@/lib/features";
import { fmtInt, fmtRelative, shortId } from "@/lib/format";
import { cn } from "@/lib/cn";
import { Button, Card, Empty, ErrorState, PageHeader, Pill, Skeleton, StatusBadge, TierBadge } from "@/components/ui";
import { STATUS_META } from "@/lib/status";

const STATUSES: CaseStatus[] = ["new", "reviewing", "dispatched", "confirmed", "cleared"];
const PAGE_SIZE = 25;

export function CasesPage() {
  const navigate = useNavigate();
  const [status, setStatus] = useState<CaseStatus | undefined>(undefined);
  const [tier, setTier] = useState<"high" | "medium" | undefined>(undefined);
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);

  const cases = useQuery({
    queryKey: ["cases", "list", status, tier, search, page],
    queryFn: () => getCases({ status, tier, search: search.trim() || undefined, page, page_size: PAGE_SIZE }),
    placeholderData: keepPreviousData,
  });
  const counts = cases.data?.status_counts;
  const totalAll = counts ? Object.values(counts).reduce((a, b) => a + b, 0) : undefined;
  const pages = cases.data ? Math.max(1, Math.ceil(cases.data.total / PAGE_SIZE)) : 1;

  const reset = (fn: () => void) => { fn(); setPage(1); };

  return (
    <>
      <PageHeader
        eyebrow={cases.data ? `${fmtInt(totalAll)} customers at or above threshold ${cases.data.threshold.toFixed(3)}` : "Loading…"}
        title="Case files"
        description="Every flagged customer becomes a case with the model's reasons attached. Work them from new to a field outcome."
      />

      <div role="tablist" aria-label="Case status" className="flex flex-wrap gap-2 border-b border-line">
        {[undefined, ...STATUSES].map((s) => {
          const active = status === s;
          const count = s ? counts?.[s] : totalAll;
          return (
            <button key={s ?? "all"} type="button" role="tab" aria-selected={active} onClick={() => reset(() => setStatus(s))}
              className={cn("-mb-px flex h-11 items-center gap-2 border-b-2 px-3 text-sm transition-colors",
                active ? "border-ink font-medium text-ink" : "border-transparent text-ink-2 hover:text-ink")}>
              {s ? STATUS_META[s].label : "All cases"}
              <span className="rounded-full bg-tint px-2 py-0.5 font-mono text-[11px] text-ink-2">{count ?? "…"}</span>
            </button>
          );
        })}
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div role="group" aria-label="Filter by tier" className="flex gap-1.5">
          <Pill active={!tier} onClick={() => reset(() => setTier(undefined))}>All tiers</Pill>
          <Pill active={tier === "high"} onClick={() => reset(() => setTier("high"))}>High</Pill>
          <Pill active={tier === "medium"} onClick={() => reset(() => setTier("medium"))}>Medium</Pill>
        </div>
        <label className="flex h-11 w-full max-w-[320px] items-center gap-2 rounded-lg border border-line bg-surface px-3.5">
          <Search className="h-4 w-4 text-ink-3" aria-hidden />
          <span className="sr-only">Filter by customer ID</span>
          <input type="search" value={search} onChange={(e) => reset(() => setSearch(e.target.value))} placeholder="Filter by customer ID"
            className="min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-ink-3" />
        </label>
      </div>

      {cases.isError ? <ErrorState message={apiErrorMessage(cases.error)} /> : null}

      <Card label="Cases" className="overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[860px] border-collapse text-left">
            <thead>
              <tr className="border-b border-line-soft text-[11px] uppercase tracking-[0.08em] text-ink-3">
                <th className="w-14 py-3 pl-[22px] font-normal">Rank</th>
                <th className="py-3 font-normal">Customer</th>
                <th className="py-3 font-normal">Probability</th>
                <th className="py-3 font-normal">Tier</th>
                <th className="py-3 font-normal">Strongest signal</th>
                <th className="py-3 font-normal">Status</th>
                <th className="py-3 pr-[22px] font-normal">Updated</th>
              </tr>
            </thead>
            <tbody>
              {cases.data?.items.map((item) => (
                <tr key={item.customer_id} className="h-14 cursor-pointer border-b border-[#F1EFEA] hover:bg-surface-alt" onClick={() => navigate(`/cases/${item.customer_id}`)}>
                  <td className="pl-[22px] font-mono text-xs text-ink-3">{item.rank}</td>
                  <td className="font-mono text-[13px]">
                    <Link to={`/cases/${item.customer_id}`} onClick={(e) => e.stopPropagation()} className="hover:text-cobalt" title={item.customer_id}>{shortId(item.customer_id, 16)}</Link>
                  </td>
                  <td className="font-mono text-[13px] tabular">{item.risk_score.toFixed(3)}</td>
                  <td><TierBadge tier={item.risk_tier} /></td>
                  <td className="text-[13px]">
                    {item.top_driver ? (
                      <span className="flex flex-col">
                        <span>{featureLabel(item.top_driver.feature)}</span>
                        <span className="text-xs text-ink-3">{featureValue(item.top_driver.feature, item.top_driver.value)}</span>
                      </span>
                    ) : "—"}
                  </td>
                  <td><StatusBadge status={item.status} /></td>
                  <td className="pr-[22px] text-xs text-ink-3">{fmtRelative(item.updated_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {cases.isLoading ? <div className="flex flex-col gap-2 p-4">{Array.from({ length: 8 }, (_, i) => <Skeleton key={i} className="h-11" />)}</div> : null}
          {cases.data && cases.data.items.length === 0 ? <div className="p-6"><Empty title="No cases match" message="Change the status, tier, or search filter." /></div> : null}
        </div>
        <div className="flex items-center justify-between border-t border-line-soft px-[22px] py-3 text-[13px] text-ink-2">
          <span>{cases.data ? `${fmtInt(cases.data.total)} cases · page ${page} of ${pages}` : "…"}</span>
          <div className="flex gap-2">
            <Button variant="secondary" aria-label="Previous page" disabled={page <= 1} onClick={() => setPage((p) => p - 1)} className="w-11 px-0"><ChevronLeft className="h-4 w-4" aria-hidden /></Button>
            <Button variant="secondary" aria-label="Next page" disabled={page >= pages} onClick={() => setPage((p) => p + 1)} className="w-11 px-0"><ChevronRight className="h-4 w-4" aria-hidden /></Button>
          </div>
        </div>
      </Card>
    </>
  );
}
