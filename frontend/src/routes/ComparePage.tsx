import { useEffect, useState } from "react";
import { Panel } from "@/components/ui/primitives";
import { apiClient } from "@/lib/api-client";
import type { CompareResponse } from "@/lib/api-types";

export function ComparePage() {
  const [compare, setCompare] = useState<CompareResponse | null>(null);

  useEffect(() => {
    void apiClient.compareBaselines().then(setCompare).catch(() => setCompare(null));
  }, []);

  return (
    <Panel className="p-5">
      <h3 className="font-display text-2xl font-semibold">Baseline comparison</h3>
      <pre className="mt-4 overflow-auto rounded-2xl bg-gp-panel-alt p-4 text-xs text-gp-text-muted">{JSON.stringify(compare, null, 2)}</pre>
    </Panel>
  );
}