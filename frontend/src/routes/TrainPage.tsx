import { useEffect, useRef, useState } from "react";
import { Badge, Button, Input, Panel } from "@/components/ui/primitives";
import { API_BASE_URL, apiClient } from "@/lib/api-client";
import type { TrainingJobStatusResponse } from "@/lib/api-types";

export function TrainPage() {
  const [job, setJob] = useState<TrainingJobStatusResponse | null>(null);
  const [jobId, setJobId] = useState("");
  const [mode, setMode] = useState<"quick" | "full" | "custom">("quick");
  const [events, setEvents] = useState<TrainingJobStatusResponse[]>([]);
  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    return () => {
      eventSourceRef.current?.close();
    };
  }, []);

  function attachStream(targetJobId: string) {
    eventSourceRef.current?.close();
    const source = new EventSource(`${API_BASE_URL}/api/train/jobs/${encodeURIComponent(targetJobId)}/stream`);
    eventSourceRef.current = source;

    source.onmessage = (event) => {
      const payload = JSON.parse(event.data) as TrainingJobStatusResponse;
      setJob(payload);
      setEvents((current) => [payload, ...current].slice(0, 12));
      if (payload.status === "succeeded" || payload.status === "failed") {
        source.close();
      }
    };

    source.onerror = () => {
      source.close();
    };
  }

  async function startQuickTrain() {
    const created = await apiClient.createTrainingJob({ mode });
    setJobId(created.job_id);
    setJob(created);
    setEvents([created]);
    attachStream(created.job_id);
  }

  return (
    <div className="grid gap-4 xl:grid-cols-[0.85fr_1.15fr]">
      <Panel className="p-5">
        <Badge tone="signal">Training</Badge>
        <h3 className="mt-3 font-display text-3xl font-semibold">Live training job control</h3>
        <div className="mt-5 space-y-4">
          <label className="block text-sm text-gp-text-muted">
            Mode
            <select
              className="mt-2 w-full rounded-xl border border-gp-border bg-gp-panel-alt px-4 py-3 text-sm text-gp-text outline-none"
              value={mode}
              onChange={(event) => setMode(event.target.value as "quick" | "full" | "custom")}
            >
              <option value="quick">Quick</option>
              <option value="full">Full</option>
              <option value="custom">Custom</option>
            </select>
          </label>
          <Button type="button" onClick={() => void startQuickTrain()}>Start job</Button>
          <Input value={jobId} onChange={(event) => setJobId(event.target.value)} placeholder="Existing job_id for refresh" />
          <Button type="button" onClick={() => attachStream(jobId)} disabled={!jobId}>Attach to job stream</Button>
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
        <div className="mt-4 rounded-2xl border border-gp-border bg-gp-panel-alt p-4">
          <div className="text-xs uppercase tracking-[0.18em] text-gp-text-dim">Event stream</div>
          <div className="mt-3 max-h-[260px] space-y-3 overflow-auto pr-1 text-xs">
            {events.map((event, index) => (
              <div key={`${event.job_id}-${event.updated_at}-${index}`} className="rounded-xl border border-gp-border bg-gp-panel px-3 py-2">
                <div className="font-data text-gp-text">{event.current_step}</div>
                <div className="mt-1 text-gp-text-muted">{event.status} · {new Date(event.updated_at).toLocaleTimeString()}</div>
              </div>
            ))}
            {events.length === 0 && <div className="text-gp-text-muted">SSE events will appear here as the job advances.</div>}
          </div>
        </div>
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