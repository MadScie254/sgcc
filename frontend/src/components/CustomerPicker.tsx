import { useState } from "react";
import { ChevronDown, Search } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { listCustomers } from "@/lib/api";
import { cn } from "@/components/ui/primitives";

type CustomerPickerProps = {
  selectedCustomerId: string | null;
  // eslint-disable-next-line no-unused-vars
  onSelect: (customerId: string) => void;
  placeholder?: string;
};

export function CustomerPicker({ selectedCustomerId, onSelect, placeholder = "Search customer ID" }: CustomerPickerProps) {
  const [search, setSearch] = useState("");

  const query = useQuery({
    queryKey: ["customer-picker", search],
    queryFn: () => listCustomers({ search, page_size: 8, sort_by: "risk_score", sort_dir: "desc" }),
    enabled: search.trim().length >= 2,
  });

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 rounded-md border border-border bg-surface px-3 py-2">
        <Search className="h-4 w-4 text-muted" />
        <input
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder={placeholder}
          className="w-full border-0 bg-transparent text-sm text-primary outline-none placeholder:text-muted"
        />
      </div>
      <div className="rounded-md border border-border bg-surface">
        <div className="flex items-center justify-between border-b border-border px-3 py-2 text-[11px] uppercase tracking-[0.18em] text-muted">
          <span>Results</span>
          <ChevronDown className="h-4 w-4" />
        </div>
        <div className="max-h-56 overflow-auto p-1">
          {(query.data?.items ?? []).map((item) => (
            <button
              key={item.customer_id}
              type="button"
              onClick={() => onSelect(item.customer_id)}
              className={cn(
                "flex w-full items-center justify-between rounded-md px-3 py-2 text-left text-sm transition hover:bg-surface-alt",
                selectedCustomerId === item.customer_id && "bg-accent-bg text-accent",
              )}
            >
              <span className="font-mono text-xs">{item.customer_id}</span>
              <span className="text-xs text-muted">{item.risk_score.toFixed(3)}</span>
            </button>
          ))}
          {search.trim().length >= 2 && !query.isLoading && (query.data?.items.length ?? 0) === 0 ? (
            <div className="px-3 py-4 text-sm text-secondary">No matching customers.</div>
          ) : null}
          {search.trim().length < 2 ? <div className="px-3 py-4 text-sm text-secondary">Type at least 2 characters.</div> : null}
        </div>
      </div>
    </div>
  );
}
