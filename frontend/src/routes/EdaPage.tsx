import { useEffect, useState } from "react";
import { Panel } from "@/components/ui/primitives";
import { apiClient } from "@/lib/api-client";
import type { CorrelationMatrixResponse, DatasetSummary } from "@/lib/api-types";

export function EdaPage() {
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [matrix, setMatrix] = useState<CorrelationMatrixResponse | null>(null);

  useEffect(() => {
    void apiClient.datasetSummary().then(setSummary).catch(() => setSummary(null));
    void apiClient.correlationMatrix().then(setMatrix).catch(() => setMatrix(null));
  }, []);

  return (
    <div className="grid gap-4 xl:grid-cols-[0.8fr_1.2fr]">
      <Panel className="p-5">
        <h3 className="font-display text-2xl font-semibold">EDA Summary</h3>
        <div className="mt-4 space-y-3 text-sm text-gp-text-muted">
          <p>Customers: {summary?.total_customers ?? "--"}</p>
          <p>Rows: {summary?.total_rows ?? "--"}</p>
          <p>Features: {summary?.feature_count ?? "--"}</p>
          <p>Zero usage: {summary?.zero_consumption_pct?.toFixed(2) ?? "--"}%</p>
        </div>
      </Panel>
      <Panel className="overflow-hidden p-5">
        <h3 className="font-display text-2xl font-semibold">Correlation Matrix</h3>
        <div className="mt-4 max-h-[620px] overflow-auto rounded-2xl border border-gp-border">
          <table className="min-w-full text-left text-xs">
            <thead className="sticky top-0 bg-gp-panel-alt text-gp-text-muted">
              <tr>
                <th className="px-3 py-2">Feature</th>
                {(matrix?.features ?? []).slice(0, 8).map((feature) => (
                  <th key={feature} className="px-3 py-2 font-medium">{feature}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(matrix?.features ?? []).slice(0, 8).map((feature, rowIndex) => (
                <tr key={feature} className="border-t border-gp-border">
                  <td className="px-3 py-2 font-medium text-gp-text">{feature}</td>
                  {(matrix?.matrix[rowIndex] ?? []).slice(0, 8).map((value, colIndex) => (
                    <td key={`${feature}-${colIndex}`} className="px-3 py-2 text-gp-text-muted">
                      {value.toFixed(2)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}