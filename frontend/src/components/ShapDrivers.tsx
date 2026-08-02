import type { PredictionReason } from "@/lib/api";
import { cn } from "@/components/ui/primitives";

type ShapDriversProps = {
  items: PredictionReason[];
  title?: string;
  emptyLabel?: string;
};

export function ShapDrivers({ items, title = "Top SHAP drivers", emptyLabel = "Select a customer to load SHAP values." }: ShapDriversProps) {
  const maxValue = Math.max(...items.map((item) => Math.abs(item.shap_value)), 0.0001);

  return (
    <div className="space-y-3">
      <div className="text-xs uppercase tracking-[0.18em] text-muted">{title}</div>
      {items.length === 0 ? (
        <div className="rounded-md border border-border bg-surface-alt px-4 py-6 text-sm text-secondary">{emptyLabel}</div>
      ) : (
        <div className="space-y-3">
          {items.map((item) => {
            const width = Math.max(Math.abs(item.shap_value) / maxValue, 0.06) * 100;
            const isPositive = item.shap_value >= 0;
            return (
              <div key={item.feature} className="space-y-1 rounded-md border border-border bg-surface px-3 py-3">
                <div className="flex items-start justify-between gap-3 text-sm">
                  <span className="font-medium text-primary">{item.feature}</span>
                  <span className={cn("font-mono text-xs", isPositive ? "text-danger" : "text-success")}>{item.shap_value.toFixed(4)}</span>
                </div>
                <div className="h-1.5 overflow-hidden rounded-full bg-surface-alt">
                  <div className={cn("h-full rounded-full", isPositive ? "bg-danger" : "bg-success")} style={{ width: `${width}%` }} />
                </div>
                <div className="text-[11px] text-secondary">{item.value.toFixed(4)}</div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
