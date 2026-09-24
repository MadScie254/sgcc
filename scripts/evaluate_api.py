"""
Check the model end to end through a running API.

    uvicorn backend.main:app --port 8000        # in another terminal
    python scripts/evaluate_api.py --url http://127.0.0.1:8000 [--key API_KEY]

Verifies that the served model reproduces its hold-out quality, that every
endpoint scores a customer identically, that SHAP explanations add up to the
predicted probability, that publishing a threshold re-flags customers, and
that a PDF report can be generated and downloaded. Exits non-zero if any
check fails.
"""

import argparse
import json
import math
import sys
import urllib.error
import urllib.request


class Client:
    def __init__(self, base: str, key: str | None):
        self.base = base.rstrip("/") + "/api"
        self.headers = {"Content-Type": "application/json", **({"X-API-Key": key} if key else {})}

    def raw(self, method: str, path: str, body=None) -> bytes:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self.base + path, data=data, method=method, headers=self.headers)
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()

    def call(self, method: str, path: str, body=None):
        return json.loads(self.raw(method, path, body))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the model through the API")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--key", default=None, help="API key (X-API-Key)")
    parser.add_argument("--sample", type=int, default=25, help="customers to spot-check")
    args = parser.parse_args()
    api = Client(args.url, args.key)
    failures = []

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failures.append(name)

    try:
        health = api.call("GET", "/health")
    except urllib.error.URLError as exc:
        print(f"API not reachable at {args.url}: {exc}")
        return 2
    check("health", health.get("model_loaded") is True, f"model v{health.get('model_version')} loaded")

    run = api.call("POST", "/pipeline/runs")
    check("scoring run", run["status"] == "succeeded",
          f"{run['seconds']:.2f} s · " + ", ".join(f"{s['name']} {s['seconds']:.2f}s" for s in run["stages"]))

    metrics = api.call("GET", "/model/metrics")
    trained = metrics["trained_threshold"]
    preview = api.call("GET", f"/predict/threshold-preview?threshold={trained}")
    served_auc = preview["metrics"]["auc"]
    check("hold-out ROC-AUC (served customers)", served_auc >= 0.80,
          f"{served_auc:.4f} on {metrics['customers_monitored']} customers (training report: {metrics['metrics']['auc']:.4f})")

    curve = api.call("GET", "/model/operating-curve")
    # PR-AUC by the step rule over the curve's thresholds (a coarse lower-resolution estimate).
    pts = sorted(curve, key=lambda p: p["recall"])
    pr_auc = sum((b["recall"] - a["recall"]) * b["precision"] for a, b in zip(pts, pts[1:]))
    base_rate = metrics["base_rate"]
    check("PR-AUC vs base rate", pr_auc > 3 * base_rate, f"≈{pr_auc:.3f} (base rate {base_rate:.3f})")
    cm = preview["confusion_matrix"]
    check("precision at trained threshold", preview["metrics"]["precision"] > 0.4,
          f"precision {preview['metrics']['precision']:.3f}, recall {preview['metrics']['recall']:.3f}, "
          f"tp {cm['tp']} fp {cm['fp']} fn {cm['fn']} tn {cm['tn']}")

    ranked = api.call("GET", f"/customers?page_size={min(args.sample, 100)}")["items"]
    worst_gap, worst_shap = 0.0, 0.0
    for item in ranked[: args.sample]:
        single = api.call("POST", "/predict/single", {"customer_id": item["customer_id"]})
        local = api.call("GET", f"/customers/{item['customer_id']}/explanation")
        worst_gap = max(worst_gap, abs(single["probability"] - item["risk_score"]), abs(local["probability"] - item["risk_score"]))
        logit = local["base_value"] + sum(c["shap_value"] for c in local["contributions"])
        worst_shap = max(worst_shap, abs(1 / (1 + math.exp(-logit)) - local["probability"]))
    check("endpoint consistency", worst_gap < 1e-6, f"max score difference across endpoints {worst_gap:.2e} over {len(ranked[:args.sample])} customers")
    check("SHAP additivity", worst_shap < 1e-3, f"max |sigmoid(base + Σ shap) − p| = {worst_shap:.2e}")

    flagged_before = metrics["flagged"]
    raised = api.call("PUT", "/model/threshold", {"threshold": 0.5})
    restored = api.call("PUT", "/model/threshold", {"threshold": None})
    check("threshold publish round trip", raised["flagged"] < flagged_before == restored["flagged"],
          f"flagged {flagged_before} → {raised['flagged']} at τ 0.5 → {restored['flagged']} after reset")

    # Worked cases stay listed after their customer drops below the threshold, so ≥ rather than ==.
    cases = api.call("GET", "/cases?page_size=1")
    check("cases opened for flags", cases["total"] >= restored["flagged"], f"{cases['total']} cases for {restored['flagged']} flagged customers")

    if health["reports"]["available"]:
        report = api.call("POST", "/reports", {"kind": "portfolio"})
        pdf = api.raw("GET", f"/reports/{report['report_id']}/pdf")
        check("portfolio report", pdf.startswith(b"%PDF") and len(pdf) == report["bytes"], f"{report['report_id']}.pdf, {len(pdf):,} bytes")
    else:
        check("portfolio report", False, health["reports"]["detail"])

    print(f"\n{'All checks passed' if not failures else f'{len(failures)} check(s) failed: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
