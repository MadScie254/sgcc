import { createHash } from "node:crypto";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { defineConfig, devices } from "@playwright/test";

// The suite starts the API in production mode (built console, named keys, fresh state)
// unless E2E_BASE_URL points at a server that is already running with these keys.
export const KEYS = { supervisor: "e2e-supervisor-key", analyst: "e2e-analyst-key" };
const sha256 = (key: string) => createHash("sha256").update(key).digest("hex");
const PORT = 8765;
const external = process.env.E2E_BASE_URL;

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 90_000,
  expect: { timeout: 20_000 },
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: external ?? `http://127.0.0.1:${PORT}`,
    trace: "retain-on-failure",
    acceptDownloads: true,
    ...devices["Desktop Chrome"],
    viewport: { width: 1440, height: 1000 },
  },
  webServer: external ? undefined : {
    command: `${process.env.PYTHON ?? "python"} -m uvicorn backend.main:app --host 127.0.0.1 --port ${PORT}`,
    cwd: "..",
    url: `http://127.0.0.1:${PORT}/api/health`,
    timeout: 180_000,
    reuseExistingServer: false,
    stdout: "pipe",
    env: {
      ENV: "production",
      API_KEYS: JSON.stringify([
        { name: "e2e-supervisor", role: "supervisor", sha256: sha256(KEYS.supervisor) },
        { name: "e2e-analyst", role: "analyst", sha256: sha256(KEYS.analyst) },
      ]),
      SGCC_STATE_DIR: mkdtempSync(join(tmpdir(), "sgcc-e2e-")),
      // Never the working database: TEST_DATABASE_URL, or SQLite in the temporary state folder.
      DATABASE_URL: process.env.TEST_DATABASE_URL ?? "",
      S3_BUCKET: "",
    },
  },
});
