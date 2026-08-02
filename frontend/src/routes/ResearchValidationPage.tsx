import { useEffect, useState } from "react";
import { Panel } from "@/components/ui/primitives";
import { apiClient } from "@/lib/api-client";

export function ResearchValidationPage() {
  const [health, setHealth] = useState<{ status: string; service: string } | null>(null);

  useEffect(() => {
    void apiClient.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  return (
    <Panel className="p-5">
      <h3 className="font-display text-2xl font-semibold">Research validation</h3>
      <p className="mt-3 text-sm text-gp-text-muted">Backend health: {health ? `${health.status} (${health.service})` : "unavailable"}</p>
    </Panel>
  );
}