import { useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { listCustomers } from "@/lib/api";
import { CenteredState } from "@/components/CenteredState";
import { Panel } from "@/components/ui/primitives";
import { riskTierClasses, riskTierLabel } from "@/lib/dashboard";

const PAGE_SIZE = 20;

export function CustomersPage() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [search, setSearch] = useState(searchParams.get("search") ?? "");
  const [riskTier, setRiskTier] = useState<string | null>(searchParams.get("risk_tier"));
  const [sortBy, setSortBy] = useState("risk_score");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [page, setPage] = useState(1);

  useEffect(() => {
    setSearch(searchParams.get("search") ?? "");
    setRiskTier(searchParams.get("risk_tier"));
  }, [searchParams]);

  const query = useQuery({
    queryKey: ["customers", search, riskTier, sortBy, sortDir, page],
    queryFn: () => listCustomers({ search: search || undefined, risk_tier: riskTier ?? undefined, sort_by: sortBy, sort_dir: sortDir, page, page_size: PAGE_SIZE }),
    staleTime: 30_000,
  });

  const totalPages = Math.max(Math.ceil((query.data?.total ?? 0) / PAGE_SIZE), 1);

  const columns = useMemo(
    () => [
      { key: "customer_id", label: "Customer" },
      { key: "risk_score", label: "Risk score" },
      { key: "risk_tier", label: "Risk tier" },
      { key: "predicted_label", label: "Label" },
    ],
    [],
  );

  function updateSearch(nextSearch: string) {
    setSearch(nextSearch);
    setPage(1);
    setSearchParams((current) => {
      const params = new URLSearchParams(current);
      if (nextSearch) params.set("search", nextSearch);
      else params.delete("search");
      if (riskTier) params.set("risk_tier", riskTier);
      else params.delete("risk_tier");
      return params;
    });
  }

  function updateTier(nextTier: string | null) {
    setRiskTier(nextTier);
    setPage(1);
    setSearchParams((current) => {
      const params = new URLSearchParams(current);
      if (search) params.set("search", search);
      else params.delete("search");
      if (nextTier) params.set("risk_tier", nextTier);
      else params.delete("risk_tier");
      return params;
    });
  }

  function toggleSort(column: string) {
    setPage(1);
    if (sortBy === column) {
      setSortDir((current) => (current === "asc" ? "desc" : "asc"));
      return;
    }
    setSortBy(column);
    setSortDir(column === "customer_id" ? "asc" : "desc");
  }

  if (query.isError) {
    return <CenteredState title="Customer audit unavailable" description="The customer table could not be loaded from the backend." />;
  }

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-primary">Customers</h1>
        <p className="mt-1 text-sm text-secondary">Audit-focused table with server-side search, sorting, filtering, and pagination.</p>
      </div>

      <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
        <div className="w-full max-w-xl">
          <label className="text-xs uppercase tracking-[0.18em] text-secondary">Search customer ID</label>
          <input
            value={search}
            onChange={(event) => updateSearch(event.target.value)}
            placeholder="Search customer ID"
            className="mt-2 w-full rounded-md border border-border bg-surface px-4 py-3 text-sm text-primary outline-none placeholder:text-muted focus:ring-1 focus:ring-accent"
          />
        </div>

        <div className="flex flex-wrap gap-2">
          {([null, "high", "medium", "low"] as Array<string | null>).map((tier) => (
            <button
              key={tier ?? "all"}
              type="button"
              onClick={() => updateTier(tier)}
              className={`rounded-md border px-3 py-2 text-sm font-medium ${riskTier === tier ? "border-accent bg-accent-bg text-accent" : "border-border bg-surface text-secondary hover:bg-surface-alt"}`}
            >
              {tier ? riskTierLabel(tier as "high" | "medium" | "low") : "All tiers"}
            </button>
          ))}
        </div>
      </div>

      <Panel className="overflow-hidden rounded-lg border border-border bg-surface p-0">
        <table className="min-w-full border-collapse text-sm">
          <thead className="bg-surface-alt text-xs uppercase tracking-[0.18em] text-secondary">
            <tr>
              {columns.map((column) => (
                <th key={column.key} className="px-4 py-3 text-left font-medium">
                  <button type="button" onClick={() => toggleSort(column.key)} className="flex items-center gap-2 text-left">
                    <span>{column.label}</span>
                    <span className="text-[10px] text-muted">{sortBy === column.key ? (sortDir === "asc" ? "↑" : "↓") : ""}</span>
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {query.isLoading
              ? Array.from({ length: PAGE_SIZE }).map((_, index) => (
                  <tr key={index} className="border-t border-border">
                    <td colSpan={4} className="px-4 py-4">
                      <div className="h-4 w-full animate-pulse rounded-md bg-surface-alt" />
                    </td>
                  </tr>
                ))
              : query.data?.items.map((item) => (
                  <tr key={item.customer_id} onClick={() => navigate(`/predict?customer=${encodeURIComponent(item.customer_id)}`)} className="cursor-pointer border-t border-border transition hover:bg-surface-alt">
                    <td className="px-4 py-3 font-mono text-xs text-primary">{item.customer_id}</td>
                    <td className="px-4 py-3 font-mono text-xs text-primary">{item.risk_score.toFixed(4)}</td>
                    <td className="px-4 py-3">
                      <span className={`inline-flex rounded-full border px-2.5 py-1 text-xs font-semibold ${riskTierClasses(item.risk_tier)}`}>{riskTierLabel(item.risk_tier)}</span>
                    </td>
                    <td className="px-4 py-3 text-sm text-primary">{item.predicted_label ? "Theft" : "Normal"}</td>
                  </tr>
                ))}
          </tbody>
        </table>
      </Panel>

      <div className="flex items-center justify-between rounded-md border border-border bg-surface px-4 py-3 text-sm text-secondary">
        <span>
          Page <span className="font-mono text-primary">{page}</span> of <span className="font-mono text-primary">{totalPages}</span>
        </span>
        <div className="flex items-center gap-2">
          <button type="button" onClick={() => setPage((current) => Math.max(1, current - 1))} disabled={page <= 1} className="rounded-md border border-border px-3 py-2 disabled:cursor-not-allowed disabled:opacity-50">
            Prev
          </button>
          <button type="button" onClick={() => setPage((current) => Math.min(totalPages, current + 1))} disabled={page >= totalPages} className="rounded-md border border-border px-3 py-2 disabled:cursor-not-allowed disabled:opacity-50">
            Next
          </button>
        </div>
      </div>
    </div>
  );
}
