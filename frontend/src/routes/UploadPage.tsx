import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Badge, Button, Panel } from "@/components/ui/primitives";
import { apiClient } from "@/lib/api-client";

type CsvPreview = {
  headers: string[];
  rows: string[][];
  numericColumns: Array<{ name: string; mean: number; completeness: number }>;
  totalRows: number;
};

export function UploadPage() {
  const [preview, setPreview] = useState<CsvPreview | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [message, setMessage] = useState<string>("Choose a CSV to preview and upload.");

  async function handleFile(file: File) {
    setSelectedFile(file);
    const text = await file.text();
    const parsed = parseCsv(text);
    setPreview(parsed);
    setMessage(`${file.name} parsed successfully.`);
  }

  async function handleUpload() {
    if (!selectedFile) return;
    setMessage("Uploading to prediction endpoint...");
    const blob = await apiClient.uploadBatchPredictions(selectedFile);
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${selectedFile.name.replace(/\.csv$/i, "")}-predictions.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
    setMessage("Predictions downloaded.");
  }

  const chartData = useMemo(() => preview?.numericColumns.slice(0, 8) ?? [], [preview]);

  return (
    <div className="grid gap-4 xl:grid-cols-[0.75fr_1.25fr]">
      <Panel className="p-5">
        <Badge tone="signal">Upload</Badge>
        <h3 className="mt-3 font-display text-2xl font-semibold">Preview and batch predict CSV files</h3>
        <label className="mt-4 block rounded-3xl border border-dashed border-gp-border bg-gp-panel-alt/60 p-6 text-center text-sm text-gp-text-muted">
          <input
            type="file"
            accept=".csv,text/csv"
            className="hidden"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) {
                void handleFile(file);
              }
            }}
          />
          Drop or select a CSV file
        </label>
        <div className="mt-4 space-y-3">
          <Button type="button" onClick={() => void handleUpload()} disabled={!selectedFile}>Upload for predictions</Button>
          <p className="text-sm text-gp-text-muted">{message}</p>
        </div>
        <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
          <InfoTile label="Rows" value={String(preview?.totalRows ?? 0)} />
          <InfoTile label="Columns" value={String(preview?.headers.length ?? 0)} />
        </div>
      </Panel>

      <div className="space-y-4">
        <Panel className="p-5">
          <h3 className="font-display text-2xl font-semibold">Numeric coverage</h3>
          <div className="mt-5 h-[320px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 12, right: 20, bottom: 28, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#233246" />
                <XAxis dataKey="name" stroke="#8a98a8" angle={-20} textAnchor="end" interval={0} height={60} />
                <YAxis stroke="#8a98a8" />
                <Tooltip contentStyle={{ background: "#0f1722", border: "1px solid #233246", borderRadius: 16 }} />
                <Bar dataKey="completeness" fill="#29d39b" radius={[10, 10, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel className="overflow-hidden p-5">
          <h3 className="font-display text-2xl font-semibold">Preview rows</h3>
          <div className="mt-4 max-h-[340px] overflow-auto rounded-2xl border border-gp-border">
            <table className="min-w-full text-left text-xs">
              <thead className="sticky top-0 bg-gp-panel-alt text-gp-text-muted">
                <tr>
                  {(preview?.headers ?? []).map((header) => (
                    <th key={header} className="px-3 py-2 font-medium">{header}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(preview?.rows ?? []).map((row, rowIndex) => (
                  <tr key={rowIndex} className="border-t border-gp-border">
                    {row.map((cell, cellIndex) => (
                      <td key={`${rowIndex}-${cellIndex}`} className="px-3 py-2 text-gp-text-muted">{cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>
    </div>
  );
}

function parseCsv(text: string): CsvPreview {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length === 0) {
    return { headers: [], rows: [], numericColumns: [], totalRows: 0 };
  }

  const headers = splitCsvLine(lines[0]);
  const body = lines.slice(1).map(splitCsvLine);
  const columnStats = headers.map((header, index) => {
    const values = body.map((row) => row[index] ?? "");
    const numeric = values.map((value) => Number(value)).filter((value) => Number.isFinite(value));
    const completeness = values.filter((value) => value !== "" && value !== null && value !== undefined).length / Math.max(values.length, 1);
    return {
      name: header,
      mean: numeric.length ? numeric.reduce((sum, value) => sum + value, 0) / numeric.length : 0,
      completeness,
    };
  });

  return {
    headers,
    rows: body.slice(0, 8),
    numericColumns: columnStats.filter((entry) => entry.completeness > 0).sort((left, right) => right.completeness - left.completeness),
    totalRows: body.length,
  };
}

function splitCsvLine(line: string): string[] {
  const cells: string[] = [];
  let current = "";
  let insideQuotes = false;

  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (char === '"') {
      insideQuotes = !insideQuotes;
      continue;
    }
    if (char === "," && !insideQuotes) {
      cells.push(current.trim());
      current = "";
      continue;
    }
    current += char;
  }

  cells.push(current.trim());
  return cells;
}

function InfoTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-gp-border bg-gp-panel-alt p-4">
      <div className="text-xs uppercase tracking-[0.18em] text-gp-text-dim">{label}</div>
      <div className="mt-2 font-data text-lg text-gp-text">{value}</div>
    </div>
  );
}