import { useEffect, useState } from "react";
import { Badge, Button, Input, Panel } from "@/components/ui/primitives";
import { apiClient } from "@/lib/api-client";
import type { TrainingJobStatusResponse } from "@/lib/api-types";

export function TrainPage() {
  const [job, setJob] = useState<TrainingJobStatusResponse | null>(null);
  const [jobId, setJobId] = useState("");

  useEffect(() => {
    if (!jobId) return;
    const timer = window.setInterval(() => {
      void apiClient.trainingJobStatus(jobId).then(setJob).catch(() => undefined);
    }, 1500);
    return () => window.clearInterval(timer);
  }, [jobId]);

  async function startQuickTrain() {
    const created = await apiClient.createTrainingJob({ mode: "quick" });
    setJobId(created.job_id);
    setJob(created);
  }

  return (
    <div className="grid gap-4 xl:grid-cols-[0.85fr_1.15fr]">
      <Panel className="p-5">
        <Badge tone="signal">Training</Badge>
        <h3 className="mt-3 font-display text-3xl font-semibold">Live training job control</h3>
        <div className="mt-5 space-y-4">
          <Button type="button" onClick={() => void startQuickTrain()}>Start quick job</Button>
          <Input value={jobId} onChange={(event) => setJobId(event.target.value)} placeholder="Existing job_id for refresh" />
        </div>
      </Panel>

      <Panel className="p-5">
        <p className="text-xs uppercase tracking-[0.22em] text-gp-text-dim">Job status</p>
        <div className="mt-4 grid gap-3 md:grid-cols-2">
          <StatusTile label="Job" value={job?.job_id ?? "--"} />
          <StatusTile label="State" value={job?.status ?? "--"} />
          <StatusTile label="Step" value={job?.current_step ?? "--"} />
          <StatusTile label="Best score" value={job?.best_score?.toFixed(4) ?? "--"} />
        </div>
        <pre className="mt-4 overflow-auto rounded-2xl bg-gp-panel-alt p-4 text-xs text-gp-text-muted">{JSON.stringify(job, null, 2)}</pre>
      </Panel>
    </div>
  );
}

function StatusTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-gp-border bg-gp-panel-alt p-4">
      <div className="text-xs uppercase tracking-[0.18em] text-gp-text-dim">{label}</div>
      <div className="mt-2 break-words font-data text-sm text-gp-text">{value}</div>
    </div>
  );
}