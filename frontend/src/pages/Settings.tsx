import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, XCircle } from "lucide-react";
import { apiErrorMessage, getAudit, getHealth, getMe, getStoredApiKey, setStoredApiKey } from "@/lib/api";
import { fmtDateTime } from "@/lib/format";
import { useIsSupervisor, useMe } from "@/lib/me";
import { ApiError, Button, Card, CardHeader, PageHeader, Skeleton } from "@/components/ui";

type CheckState = { tone: "ok" | "error" | "idle"; message: string };

export function SettingsPage() {
  const queryClient = useQueryClient();
  const [apiKey, setApiKey] = useState(getStoredApiKey);
  const [check, setCheck] = useState<CheckState>({ tone: "idle", message: "" });
  const [busy, setBusy] = useState(false);
  const health = useQuery({ queryKey: ["health"], queryFn: getHealth, retry: false });
  const me = useMe();
  const supervisor = useIsSupervisor();
  const audit = useQuery({ queryKey: ["audit"], queryFn: () => getAudit(30), enabled: supervisor });

  async function save() {
    setBusy(true);
    setStoredApiKey(apiKey.trim());
    try {
      // Health is public; /me is behind the key, so it proves the key works and says whose it is.
      const who = await getMe();
      setCheck({ tone: "ok", message: `Connected as ${who.name} (${who.role}). The key is kept for this tab only.` });
      await queryClient.invalidateQueries();
    } catch (error) {
      setCheck({ tone: "error", message: await apiErrorMessage(error, "The key could not be checked") });
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <PageHeader title="Settings" description="Connect this dashboard to the API." />
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Card label="API key" className="flex flex-col gap-4 p-[22px]">
          <CardHeader title="API key"
            subtitle={me.data && !me.data.auth_required
              ? "This server runs in development mode without keys: you act as the supervisor “developer”."
              : "Your personal key, from your administrator (scripts/make_api_key.py). Every action you take is recorded under its name. Kept for this browser tab only."} />
          <form className="flex flex-col gap-3" onSubmit={(event) => { event.preventDefault(); void save(); }}>
            <label htmlFor="api-key" className="text-[13px] text-ink-2">Key</label>
            <input id="api-key" type="password" autoComplete="off" value={apiKey} onChange={(e) => setApiKey(e.target.value)}
              placeholder="Your API key" className="h-11 rounded-lg border border-line px-3.5 text-sm outline-none focus:border-cobalt" />
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
              ["Database", health.data?.database ?? "…"],
              ["File storage", health.data ? (health.data.blob_store === "s3" ? "object storage (S3)" : "local disk") : "…"],
              ["PDF reports", health.data ? (health.data.reports.available ? `fpdf2 ${health.data.reports.fpdf_version}` : "unavailable") : "…"],
              ["Signed in as", me.data ? `${me.data.name} (${me.data.role})` : "…"],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between border-t border-line-soft pt-2 first:border-0 first:pt-0"><dt className="text-ink-2">{k}</dt><dd className="m-0 font-mono">{v}</dd></div>
            ))}
          </dl>
          {health.data?.problems.map((problem) => <p key={problem} className="m-0 text-[13px] text-risk-text">{problem}</p>)}
          {health.data?.reports.detail ? <p className="m-0 text-[13px] text-risk-text">{health.data.reports.detail}</p> : null}
        </Card>
      </div>
      {supervisor ? (
        <Card label="Audit log" className="flex flex-col gap-3 p-[22px]">
          <CardHeader title="Audit log" subtitle="Who published thresholds, changed cases, uploaded, downloaded or deleted files. Newest first." />
          {audit.data ? (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[640px] border-collapse text-left text-[13px]">
                <thead>
                  <tr className="border-b border-line-soft text-[11px] uppercase tracking-[0.08em] text-ink-3">
                    <th className="pb-2 font-normal">When</th><th className="pb-2 font-normal">Who</th><th className="pb-2 font-normal">Action</th><th className="pb-2 font-normal">Subject</th>
                  </tr>
                </thead>
                <tbody>
                  {audit.data.map((entry, i) => (
                    <tr key={`${entry.at}-${i}`} className="h-9 border-b border-[#F1EFEA]">
                      <td className="whitespace-nowrap pr-3 text-xs text-ink-3">{fmtDateTime(entry.at)}</td>
                      <td className="pr-3">{entry.actor} <span className="text-xs text-ink-3">{entry.role}</span></td>
                      <td className="pr-3 font-mono text-xs">{entry.action}</td>
                      <td className="max-w-[320px] truncate text-ink-2" title={[entry.target, entry.detail].filter(Boolean).join(" · ")}>
                        {[entry.target, entry.detail].filter(Boolean).join(" · ") || "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : audit.isError ? <ApiError error={audit.error} /> : <Skeleton className="h-32" />}
        </Card>
      ) : null}
    </>
  );
}
