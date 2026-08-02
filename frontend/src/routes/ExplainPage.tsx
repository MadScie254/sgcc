import { Panel } from "@/components/ui/primitives";

export function ExplainPage() {
  return (
    <div className="grid gap-4 xl:grid-cols-2">
      <Panel className="p-5">
        <h3 className="font-display text-2xl font-semibold">Global SHAP</h3>
        <p className="mt-3 text-sm text-gp-text-muted">This route is ready for the beeswarm and waterfall implementation in the next phase.</p>
      </Panel>
      <Panel className="p-5">
        <h3 className="font-display text-2xl font-semibold">Local explanation</h3>
        <p className="mt-3 text-sm text-gp-text-muted">Hook up per-customer SHAP and the comparison cards here.</p>
      </Panel>
    </div>
  );
}