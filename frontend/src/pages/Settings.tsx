import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { apiClient, getStoredApiKey, setStoredApiKey } from "@/lib/api";
import { Button, Input, Panel } from "@/components/ui/primitives";

type CheckState = { tone: "ok" | "error" | "idle"; message: string };

export function SettingsPage() {
  const queryClient = useQueryClient();
  const [apiKey, setApiKey] = useState(getStoredApiKey);
  const [check, setCheck] = useState<CheckState>({ tone: "idle", message: "" });

  async function save() {
    setStoredApiKey(apiKey.trim());
    try {
      // Any secured endpoint works; this one is cheap.
      await apiClient.get("/model/config");
      setCheck({ tone: "ok", message: "Connected. The key is stored in this browser only." });
      await queryClient.invalidateQueries();
    } catch (error) {
      const status = (error as { response?: { status?: number } }).response?.status;
      setCheck({ tone: "error", message: status === 401 ? "The API rejected this key." : "Could not reach the API." });
    }
  }

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-primary">Settings</h1>
        <p className="mt-1 text-sm text-secondary">Connect this dashboard to the API.</p>
      </div>

      <Panel className="max-w-xl rounded-lg border border-border bg-surface p-4">
        <form
          className="space-y-3"
          onSubmit={(event) => {
            event.preventDefault();
            void save();
          }}
        >
          <label className="block space-y-1 text-sm">
            <span className="block text-xs uppercase tracking-[0.16em] text-secondary">API key</span>
            <Input
              type="password"
              autoComplete="off"
              value={apiKey}
              onChange={(event) => setApiKey(event.target.value)}
              placeholder="Value of the server's API_KEY"
            />
          </label>
          <p className="text-xs leading-5 text-secondary">
            Sent as the <span className="font-mono">X-API-Key</span> header. Leave empty when the API runs with
            <span className="font-mono"> ENV=development</span> and no key.
          </p>
          <div className="flex items-center gap-3">
            <Button type="submit" className="rounded-md border-accent bg-accent-bg text-accent">
              Save and test
            </Button>
            {check.tone !== "idle" ? (
              <span className={`text-sm ${check.tone === "ok" ? "text-success" : "text-danger"}`}>{check.message}</span>
            ) : null}
          </div>
        </form>
      </Panel>
    </div>
  );
}
