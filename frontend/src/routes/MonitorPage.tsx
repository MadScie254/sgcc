import { useEffect, useState } from "react";
import { Panel } from "@/components/ui/primitives";
import { apiClient } from "@/lib/api-client";
import type { MonitorResponse } from "@/lib/api-types";

export function MonitorPage() {
  const [monitor, setMonitor] = useState<MonitorResponse | null>(null);

  useEffect(() => {
    void apiClient.monitorDrift().then(setMonitor).catch(() => setMonitor(null));
  }, []);

  return (
    <Panel className="p-5">
      <h3 className="font-display text-2xl font-semibold">Model monitoring</h3>
      <pre className="mt-4 overflow-auto rounded-2xl bg-gp-panel-alt p-4 text-xs text-gp-text-muted">{JSON.stringify(monitor, null, 2)}</pre>
    </Panel>
  );
}