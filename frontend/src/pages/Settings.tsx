import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, XCircle } from "lucide-react";
import { apiClient, getHealth, getStoredApiKey, setStoredApiKey } from "@/lib/api";
import { Button, Card, CardHeader, PageHeader } from "@/components/ui";

type CheckState = { tone: "ok" | "error" | "idle"; message: string };

export function SettingsPage() {
  const queryClient = useQueryClient();
  const [apiKey, setApiKey] = useState(getStoredApiKey);
  const [check, setCheck] = useState<CheckState>({ tone: "idle", message: "" });
  const [busy, setBusy] = useState(false);
  const health = useQuery({ queryKey: ["health"], queryFn: getHealth, retry: false });

  async function save() {
    setBusy(true);
    setStoredApiKey(apiKey.trim());
    try {
      await apiClient.get("/model/config");
      setCheck({ tone: "ok", message: "Connected. The key is stored in this browser only." });
      await queryClient.invalidateQueries();
    } catch (error) {
      const status = (error as { response?: { status?: number } }).response?.status;
      setCheck({ tone: "error", message: status === 401 ? "The API rejected this key." : "Could not reach the API." });
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <PageHeader title="Settings" description="Connect this dashboard to the API." />
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Card label="API key" className="flex flex-col gap-4 p-[22px]">
          <CardHeader title="API key" subtitle="Sent as the X-API-Key header. Leave empty when the API runs with ENV=development and no key." />
          <form className="flex flex-col gap-3" onSubmit={(event) => { event.preventDefault(); void save(); }}>
            <label htmlFor="api-key" className="text-[13px] text-ink-2">Key</label>
            <input id="api-key" type="password" autoComplete="off" value={apiKey} onChange={(e) => setApiKey(e.target.value)}
              placeholder="Value of the server's API_KEY" className="h-11 rounded-lg border border-line px-3.5 text-sm outline-none focus:border-cobalt" />
            <div className="flex flex-wrap items-center gap-3">
              <Button type="submit" variant="primary" busy={busy}>Save and test</Button>
              {check.tone !== "idle" ? (
                <span role="status" className={`flex items-center gap-1.5 text-sm ${check.tone === "ok" ? "text-cobalt-ink" : "text-risk-text"}`}>
                  {check.tone === "ok" ? <CheckCircle2 className="h-4 w-4" aria-hidden /> : <XCircle className="h-4 w-4" aria-hidden />}{check.message}
                </span>
              ) : null}
            </div>
          </form>
        </Card>
        <Card label="Service health" className="flex flex-col gap-3 p-[22px]">
          <CardHeader title="Service health" />
          <dl className="m-0 flex flex-col gap-2 text-sm">
            {[
              ["Status", health.data?.status ?? (health.isError ? "unreachable" : "…")],
              ["Model loaded", health.data ? (health.data.model_loaded ? "yes" : "no") : "…"],
              ["Model version", health.data?.model_version ?? "…"],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between border-t border-line-soft pt-2 first:border-0 first:pt-0"><dt className="text-ink-2">{k}</dt><dd className="m-0 font-mono">{v}</dd></div>
            ))}
          </dl>
        </Card>
      </div>
    </>
  );
}
