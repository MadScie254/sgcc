import { ChangeEvent, useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Bar, BarChart, Cell, CartesianGrid, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { CenteredState } from "@/components/CenteredState";
import { Button, Panel } from "@/components/ui/primitives";
import { formatMaybeNumber } from "@/lib/dashboard";
import { generateReport, getAnalyticsDashboard, uploadDataset, type DatasetCatalogItem, type ReportResponse } from "@/lib/api";

type ChartEntry = {
  name: string;
  value: number;
};

export function AnalyticsPage() {
  const queryClient = useQueryClient();
  const dashboardQuery = useQuery({ queryKey: ["analytics-dashboard"], queryFn: getAnalyticsDashboard, staleTime: 30_000 });
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [generatedReport, setGeneratedReport] = useState<ReportResponse | null>(null);

  useEffect(() => {
    if (!selectedDatasetId && dashboardQuery.data?.uploads?.length) {
      setSelectedDatasetId(dashboardQuery.data.uploads[0].dataset_id);
    }
  }, [dashboardQuery.data, selectedDatasetId]);

  const uploadMutation = useMutation({
    mutationFn: uploadDataset,
    onSuccess: async (result) => {
      setSelectedDatasetId(result.item.dataset_id);
      setSelectedFile(null);
      await queryClient.invalidateQueries({ queryKey: ["analytics-dashboard"] });
    },
  });

  const reportMutation = useMutation({
    mutationFn: generateReport,
    onSuccess: async (result) => {
      setGeneratedReport(result);
      await queryClient.invalidateQueries({ queryKey: ["analytics-dashboard"] });
    },
  });

  const classDistribution = useMemo<ChartEntry[]>(() => {
    const distribution = dashboardQuery.data?.dataset_summary.class_distribution ?? {};
    return Object.entries(distribution).map(([name, value]) => ({ name, value: Number(value) || 0 }));
  }, [dashboardQuery.data]);

  const riskDistribution = useMemo<ChartEntry[]>(() => {
    const distribution = dashboardQuery.data?.model_metrics.risk_tier_distribution ?? {};
    return [
      { name: "High", value: Number(distribution.high ?? 0) || 0 },
      { name: "Medium", value: Number(distribution.medium ?? 0) || 0 },
      { name: "Low", value: Number(distribution.low ?? 0) || 0 },
    ];
  }, [dashboardQuery.data]);

  const latestUpload = dashboardQuery.data?.uploads?.[0] ?? null;
  const latestReport = dashboardQuery.data?.reports?.[0] ?? null;

  if (dashboardQuery.isError) {
    return <CenteredState title="Analytics unavailable" description="The analytics dashboard could not load model and dataset context from the backend." />;
  }

  const metrics = dashboardQuery.data?.model_metrics;
  const currentSummary = dashboardQuery.data?.dataset_summary;
  const context = dashboardQuery.data?.context as Record<string, any> | undefined;

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-primary">Analytics</h1>
        <p className="mt-1 text-sm text-secondary">A production-oriented dashboard for data snapshots, uploaded dataset verification, and report generation.</p>
      </div>

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Customers" value={currentSummary ? currentSummary.total_customers.toString() : "—"} detail="Current dataset footprint" />
        <MetricCard label="Rows" value={currentSummary ? currentSummary.total_rows.toString() : "—"} detail="Raw observations available for reporting" />
        <MetricCard label="Recall" value={metrics ? formatMaybeNumber(metrics.metrics.recall, 3) : "—"} detail="Current model snapshot" />
        <MetricCard label="Flagged" value={metrics ? metrics.flagged_today.toString() : "—"} detail="Customers above threshold" />
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.15fr_0.85fr]">
        <Panel className="rounded-lg border border-border bg-surface p-4">
          <div className="text-xs uppercase tracking-[0.18em] text-secondary">Dataset class distribution</div>
          <div className="mt-4 h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={classDistribution}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E5EA" />
                <XAxis dataKey="name" stroke="#5B6472" />
                <YAxis stroke="#5B6472" />
                <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #E2E5EA" }} />
                <Bar dataKey="value" fill="#2B5FAD" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel className="rounded-lg border border-border bg-surface p-4">
          <div className="text-xs uppercase tracking-[0.18em] text-secondary">Risk tier mix</div>
          <div className="mt-4 h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #E2E5EA" }} />
                <Pie data={riskDistribution} dataKey="value" nameKey="name" innerRadius={70} outerRadius={110} paddingAngle={4}>
                  {riskDistribution.map((entry) => (
                    <Cell
                      key={entry.name}
                      fill={entry.name === "High" ? "#C0392B" : entry.name === "Medium" ? "#B7791F" : "#1E7A5F"}
                    />
                  ))}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      </div>

      <div className="grid gap-4 xl:grid-cols-[1fr_0.9fr]">
        <Panel className="rounded-lg border border-border bg-surface p-4">
          <div className="text-xs uppercase tracking-[0.18em] text-secondary">Upload and verify dataset</div>
          <p className="mt-2 text-sm text-secondary">Upload a CSV, persist it in the system catalog, and score it against the trained model for report-ready verification.</p>

          <div className="mt-4 space-y-3">
            <input
              type="file"
              accept=".csv,text/csv"
              onChange={(event: ChangeEvent<HTMLInputElement>) => setSelectedFile(event.target.files?.[0] ?? null)}
              className="block w-full rounded-md border border-border bg-surface px-4 py-3 text-sm text-secondary file:mr-4 file:rounded-md file:border-0 file:bg-accent-bg file:px-3 file:py-2 file:text-sm file:font-semibold file:text-accent"
            />
            <div className="flex flex-wrap gap-2">
              <Button type="button" onClick={() => selectedFile && uploadMutation.mutate(selectedFile)} disabled={!selectedFile || uploadMutation.isPending} className="rounded-md border-accent bg-accent-bg text-accent">
                {uploadMutation.isPending ? "Uploading..." : "Upload and verify"}
              </Button>
              <Button type="button" onClick={() => reportMutation.mutate({ dataset_id: selectedDatasetId || undefined })} disabled={reportMutation.isPending} className="rounded-md border-border bg-surface text-primary">
                {reportMutation.isPending ? "Generating..." : "Generate PDF report"}
              </Button>
            </div>
          </div>

          <div className="mt-5 grid gap-3 md:grid-cols-2">
            <MetricCard label="Latest upload" value={latestUpload ? latestUpload.original_filename : "None"} detail={latestUpload ? `${latestUpload.rows} rows · ${latestUpload.status}` : "Upload a CSV to persist it in the system"} compact />
            <MetricCard label="Latest report" value={latestReport ? String(latestReport.report_id ?? "—") : "None"} detail={latestReport ? "Available in the report registry" : "Generate a report to add a PDF artifact"} compact />
          </div>

          {uploadMutation.data?.item ? <DatasetSummaryCard item={uploadMutation.data.item} /> : null}
          {generatedReport ? <ReportResultCard report={generatedReport} /> : null}
        </Panel>

        <Panel className="rounded-lg border border-border bg-surface p-4">
          <div className="text-xs uppercase tracking-[0.18em] text-secondary">Public context feeds</div>
          <div className="mt-4 space-y-3 text-sm">
            <ContextRow label="Country" value={String(context?.country?.name ?? "N/A")} />
            <ContextRow label="Capital" value={String(context?.country?.capital ?? "N/A")} />
            <ContextRow label="Temperature" value={context?.weather?.temperature_2m !== undefined && context?.weather?.temperature_2m !== null ? `${context.weather.temperature_2m}°C` : "N/A"} />
            <ContextRow label="Holiday count" value={Array.isArray(context?.holidays) ? String(context.holidays.length) : "0"} />
          </div>

          <div className="mt-6 border-t border-border pt-4">
            <div className="text-xs uppercase tracking-[0.18em] text-secondary">Selected dataset</div>
            <select
              value={selectedDatasetId}
              onChange={(event) => setSelectedDatasetId(event.target.value)}
              className="mt-3 w-full rounded-md border border-border bg-surface px-4 py-3 text-sm text-primary outline-none"
            >
              <option value="">Use current model snapshot</option>
              {dashboardQuery.data?.uploads?.map((item: DatasetCatalogItem) => (
                <option key={item.dataset_id} value={item.dataset_id}>
                  {item.original_filename} ({item.rows} rows)
                </option>
              ))}
            </select>
          </div>

          {dashboardQuery.data?.uploads?.length ? (
            <div className="mt-6">
              <div className="text-xs uppercase tracking-[0.18em] text-secondary">Catalogued datasets</div>
              <div className="mt-3 space-y-2 max-h-[220px] overflow-auto pr-1">
                {dashboardQuery.data.uploads.map((item) => (
                  <div key={item.dataset_id} className="rounded-md border border-border bg-surface-alt px-3 py-3 text-sm">
                    <div className="flex items-center justify-between gap-3">
                      <div className="font-medium text-primary">{item.original_filename}</div>
                      <span className="text-xs text-secondary">{item.status}</span>
                    </div>
                    <div className="mt-1 text-xs text-secondary">
                      {item.rows} rows · {item.columns} columns · mean probability {formatMaybeNumber(Number(item.summary.mean_probability ?? 0), 3)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </Panel>
      </div>
    </div>
  );
}

function MetricCard({
  label,
  value,
  detail,
  compact = false,
}: {
  label: string;
  value: string;
  detail: string;
  compact?: boolean;
}) {
  return (
    <Panel className="rounded-lg border border-border bg-surface p-4">
      <div className="text-xs uppercase tracking-[0.18em] text-secondary">{label}</div>
      <div className={`mt-3 ${compact ? "text-base" : "text-2xl"} font-medium text-primary`}>{value}</div>
      <p className="mt-2 text-sm text-secondary">{detail}</p>
    </Panel>
  );
}

function ContextRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-md border border-border bg-surface-alt px-3 py-3">
      <span className="text-secondary">{label}</span>
      <span className="font-medium text-primary">{value}</span>
    </div>
  );
}

function DatasetSummaryCard({ item }: { item: DatasetCatalogItem }) {
  return (
    <Panel className="mt-4 rounded-lg border border-border bg-surface-alt p-4">
      <div className="text-xs uppercase tracking-[0.18em] text-secondary">Uploaded dataset summary</div>
      <div className="mt-3 grid gap-2 md:grid-cols-2 text-sm">
        <ContextRow label="Rows" value={String(item.rows)} />
        <ContextRow label="Columns" value={String(item.columns)} />
        <ContextRow label="Status" value={item.status} />
        <ContextRow label="Mean probability" value={formatMaybeNumber(Number(item.summary.mean_probability ?? 0), 3)} />
      </div>
      <div className="mt-4 text-xs uppercase tracking-[0.18em] text-secondary">Top risk rows</div>
      <div className="mt-2 space-y-2">
        {item.top_risk_rows.slice(0, 5).map((row, index) => (
          <div key={index} className="rounded-md border border-border bg-surface px-3 py-2 text-sm">
            {String(row.customer_id ?? row.row_index ?? `Row ${index + 1}`)} · probability {formatMaybeNumber(Number(row.prediction_probability ?? 0), 3)} · {String(row.risk_tier ?? "low")}
          </div>
        ))}
      </div>
    </Panel>
  );
}

function ReportResultCard({ report }: { report: ReportResponse }) {
  return (
    <Panel className="mt-4 rounded-lg border border-border bg-surface-alt p-4">
      <div className="text-xs uppercase tracking-[0.18em] text-secondary">Generated report</div>
      <div className="mt-3 grid gap-2 text-sm">
        <ContextRow label="Report ID" value={report.report_id} />
        <ContextRow label="Dataset" value={report.dataset_label} />
        <ContextRow label="Generated" value={report.generated_at} />
      </div>
      <div className="mt-4 flex flex-wrap gap-2">
        <a href={report.download_url} target="_blank" rel="noreferrer" className="rounded-md border border-accent bg-accent-bg px-4 py-2 text-sm font-semibold text-accent">
          Download PDF
        </a>
      </div>
    </Panel>
  );
}