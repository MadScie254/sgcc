import { expect, test, type Page } from "@playwright/test";
import { KEYS } from "../playwright.config";

async function signIn(page: Page, key: string, who: string) {
  await page.goto("/settings");
  await page.fill("#api-key", key);
  await page.getByRole("button", { name: "Save and test" }).click();
  await expect(page.getByRole("status").filter({ hasText: `Connected as ${who}` })).toBeVisible();
}

test("the API refuses requests without a named key", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("alert").first()).toContainText("Settings page");
  await page.goto("/settings");
  await page.fill("#api-key", "not-a-key");
  await page.getByRole("button", { name: "Save and test" }).click();
  await expect(page.getByText("The API rejected the key")).toBeVisible();
});

test("operations show estimates, never labels", async ({ page }) => {
  await signIn(page, KEYS.analyst, "e2e-analyst (analyst)");
  await page.goto("/");
  await expect(page.getByText("Expected thefts among flagged")).toBeVisible();
  await expect(page.getByText("A flag is a reason to inspect, not evidence of theft.")).toBeVisible();
  await expect(page.getByText(/label/i)).toHaveCount(0);
});

test("investigation flow: analyst reviews and dispatches, supervisor records the outcome", async ({ page }) => {
  await signIn(page, KEYS.analyst, "e2e-analyst (analyst)");
  await page.goto("/cases");
  // A case nobody has touched, so the suite can re-run on a database that keeps earlier runs.
  await page.getByRole("tab", { name: /^New/ }).click();
  const first = page.locator("tbody tr").first();
  await expect(first).toBeVisible();
  const customer = (await first.locator("a").first().getAttribute("title")) as string;
  await page.goto(`/cases/${encodeURIComponent(customer)}`);

  await expect(page.getByText("Dataset label")).toHaveCount(0);
  await expect(page.getByText(/Explanation consistency \(LIME\)/)).toBeVisible({ timeout: 60_000 });
  await page.getByRole("button", { name: "Start review" }).click();
  await page.getByRole("button", { name: "Dispatch field inspection" }).click();
  await expect(page.getByText("A supervisor records the inspection outcome.")).toBeVisible();
  await expect(page.getByRole("button", { name: /Confirm theft/ })).toHaveCount(0);
  await expect(page.getByText("e2e-analyst").first()).toBeVisible();

  await signIn(page, KEYS.supervisor, "e2e-supervisor (supervisor)");
  await page.goto(`/cases/${encodeURIComponent(customer)}`);
  await page.getByRole("button", { name: "Confirm theft…" }).click();
  await expect(page.getByRole("button", { name: "Mark confirmed" })).toBeDisabled();
  await page.fill("#transition-reason", "bypass cable found at the meter");
  await page.fill("#transition-evidence", "INS-2026-0412");
  await page.getByRole("button", { name: "Mark confirmed" }).click();
  await expect(page.getByRole("note").filter({ hasText: "Theft confirmed" })).toContainText("INS-2026-0412");

  await page.getByRole("button", { name: "Reopen case" }).click();
  await page.fill("#transition-reason", "customer appealed");
  await page.getByRole("button", { name: "Reopen", exact: true }).click();
  await expect(page.getByText("Status: confirmed → reviewing (customer appealed)")).toBeVisible();
});

test("only a supervisor publishes a threshold", async ({ page }) => {
  await signIn(page, KEYS.analyst, "e2e-analyst (analyst)");
  await page.goto("/threshold");
  await expect(page.getByText("Publishing a threshold needs the supervisor role.")).toBeVisible();
  await page.getByRole("button", { name: /Wide net/ }).click();
  await expect(page.getByRole("button", { name: /Publish τ/ })).toBeDisabled();

  await signIn(page, KEYS.supervisor, "e2e-supervisor (supervisor)");
  await page.goto("/threshold");
  await page.fill("#capacity", "40");
  await page.fill("#cost", "30");
  await page.fill("#value", "400");
  await expect(page.getByText("Expected net value in the population")).toBeVisible();
  await page.getByRole("button", { name: /More likely than not/ }).click();
  await page.getByRole("button", { name: /Publish τ 0.50/ }).click();
  await expect(page.getByRole("status").filter({ hasText: "Published." })).toBeVisible();
  await page.getByRole("button", { name: /Return to trained threshold/ }).click();
  await expect(page.getByRole("button", { name: /Return to trained threshold/ })).toHaveCount(0);
});

test("uploads that cannot be scored safely are rejected with the reason", async ({ page }) => {
  await signIn(page, KEYS.analyst, "e2e-analyst (analyst)");
  await page.goto("/reports");
  const days = Array.from({ length: 40 }, (_, i) => new Date(Date.UTC(2016, 0, i + 1)));
  const ambiguous = ["CONS_NO", ...days.map((d) => `${String(d.getUTCDate()).padStart(2, "0")}/${String(d.getUTCMonth() + 1).padStart(2, "0")}/2016`)];
  const csv = `${ambiguous.join(",")}\nA,${days.map(() => "1.5").join(",")}\n`;
  await page.locator("#dataset-file").setInputFiles({ name: "ambiguous.csv", mimeType: "text/csv", buffer: Buffer.from(csv) });
  await expect(page.getByRole("alert").filter({ hasText: "Upload rejected" })).toContainText("year first");

  const features = "CONS_NO,missing_ratio\nA,0.2\n";
  await page.locator("#dataset-file").setInputFiles({ name: "features.csv", mimeType: "text/csv", buffer: Buffer.from(features) });
  await expect(page.getByRole("alert").filter({ hasText: "Upload rejected" })).toContainText("model features are missing");
});

test("research and operations reports download as PDFs", async ({ page }) => {
  await signIn(page, KEYS.analyst, "e2e-analyst (analyst)");
  await page.goto("/research");
  await expect(page.getByText(/6,356 test customers/).first()).toBeVisible();
  await expect(page.getByText("PR-AUC 95% CI")).toBeVisible();
  await page.goto("/reports");
  for (const card of ["Portfolio report", "Research report"]) {
    const [download] = await Promise.all([
      page.waitForEvent("download", { timeout: 60_000 }),
      page.getByRole("region", { name: card }).getByRole("button", { name: /Generate and download PDF/ }).click(),
    ]);
    expect(download.suggestedFilename()).toMatch(/\.pdf$/);
  }
});

test("mobile navigation is a keyboard-safe dialog", async ({ page }) => {
  await signIn(page, KEYS.analyst, "e2e-analyst (analyst)");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  const opener = page.getByRole("button", { name: "Open navigation" });
  await opener.focus();
  await page.keyboard.press("Enter");
  const dialog = page.getByRole("dialog", { name: "Navigation" });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole("button", { name: "Close navigation" })).toBeFocused();
  for (let i = 0; i < 15; i += 1) {
    await page.keyboard.press("Tab");
    expect(await dialog.evaluate((node) => node.contains(document.activeElement))).toBe(true);
  }
  await page.keyboard.press("Escape");
  await expect(dialog).toHaveCount(0);
  await expect(opener).toBeFocused();
});
