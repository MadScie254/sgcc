"""
Check the model end to end through a running API.

    uvicorn backend.main:app --port 8000        # in another terminal
    python scripts/evaluate_api.py --url http://127.0.0.1:8000 [--key SUPERVISOR_KEY]

Verifies that the published files passed the integrity check, that the research
record shows the model's test-set quality, that the operational views carry no
labels, that every endpoint scores a customer identically, that SHAP explanations
add up to the raw score, that LIME's consistency check runs, that publishing a
threshold re-flags customers, and that operations and research PDF reports can be
generated and downloaded. Publishing a threshold needs a supervisor key. Exits
non-zero if any check fails.
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
    check("health", health.get("model_loaded") is True and not health.get("problems"),
          f"model v{health.get('model_version')} loaded, manifest verified, {health.get('database')} + {health.get('blob_store')} storage")
    me = api.call("GET", "/me")
    check("identity", me["role"] == "supervisor", f"acting as {me['name']} ({me['role']})")

    run = api.call("POST", "/pipeline/runs")
    check("scoring run", run["status"] == "succeeded",
          f"{run['seconds']:.2f} s · " + ", ".join(f"{s['name']} {s['seconds']:.2f}s" for s in run["stages"]))

    evaluation = api.call("GET", "/research/evaluation")
    em, population = evaluation["metrics"], evaluation["population"]
    check("test-set ROC-AUC", em["auc"] >= 0.80, f"{em['auc']:.4f} on {population['customers']} test customers")
    base_rate = population["theft"] / population["customers"]
    check("test-set PR-AUC vs base rate", em["pr_auc"] > 3 * base_rate, f"{em['pr_auc']:.4f} (base rate {base_rate:.3f})")
    calibration = evaluation["calibration"]
    check("calibration", calibration["platt"]["ece"] < calibration["raw"]["ece"],
          f"test ECE {calibration['raw']['ece']:.4f} raw → {calibration['platt']['ece']:.4f} calibrated")
    rows = api.call("GET", "/research/comparison")
    # The significance tests cover the five pipelines of the study (not the sequence model or the hybrid).
    study_rows = [r for r in rows if r["model"] not in ("wide_deep_cnn", "hybrid")]
    check("significance record", len(study_rows) == 5 and all(r["pr_auc_ci"] for r in study_rows),
          ", ".join(f"{r['model']} {r['pr_auc']:.3f} [{r['pr_auc_ci'][0]:.3f}, {r['pr_auc_ci'][1]:.3f}]" for r in rows if r["pr_auc_ci"]))

    metrics = api.call("GET", "/model/metrics")
    trained = metrics["trained_threshold"]
    check("operations carry no labels", not {"confusion_matrix", "metrics", "base_rate"} & set(metrics),
          f"{metrics['flagged']} of {metrics['customers_monitored']} flagged, ~{metrics['expected_thefts_flagged']:.0f} thefts expected")
    preview = api.call("GET", f"/model/threshold-preview?threshold={trained}")["validation"]
    check("validation precision at trained threshold", preview["precision"] > 0.4,
          f"precision {preview['precision']:.3f}, recall {preview['recall']:.3f} on {preview['customers']} validation customers")

    ranked = api.call("GET", f"/customers?page_size={min(args.sample, 100)}")["items"]
    worst_gap, worst_shap = 0.0, 0.0
    for item in ranked[: args.sample]:
        single = api.call("POST", "/predict/single", {"customer_id": item["customer_id"]})
        local = api.call("GET", f"/customers/{item['customer_id']}/explanation")
        worst_gap = max(worst_gap, abs(single["probability"] - item["risk_score"]), abs(local["probability"] - item["risk_score"]))
        logit = local["base_value"] + sum(c["shap_value"] for c in local["contributions"])
        worst_shap = max(worst_shap, abs(1 / (1 + math.exp(-logit)) - local["raw_score"]))
    check("endpoint consistency", worst_gap < 1e-6, f"max score difference across endpoints {worst_gap:.2e} over {len(ranked[:args.sample])} customers")
    check("SHAP additivity", worst_shap < 1e-3, f"max |sigmoid(base + Σ shap) − raw score| = {worst_shap:.2e}")

    weeks = api.call("GET", f"/customers/{ranked[0]['customer_id']}/sequence-explanation")
    if metrics.get("pipeline") == "hybrid":
        parts = api.call("GET", f"/customers/{ranked[0]['customer_id']}/explanation")["parts"] or []
        check("hybrid explanation", weeks["available"] and len(weeks["weeks"]) == 148 and len(parts) == 2,
              "parts " + ", ".join(f"{p['name']} {p['probability']:.3f}×{p['weight']:.2f}" for p in parts) + "; "
              f"{len(weeks['weeks'])} weeks explained for the sequence model")

    consistency = api.call("GET", f"/customers/{ranked[0]['customer_id']}/explanation-check")
    check("explanation consistency (LIME)", len(consistency["lime"]) == consistency["top_n"], consistency["message"])

    higher = round(min(0.95, trained + 0.25), 2)
    flagged_before = metrics["flagged"]
    raised = api.call("PUT", "/model/threshold", {"threshold": higher})
    restored = api.call("PUT", "/model/threshold", {"threshold": None})
    check("threshold publish round trip", raised["flagged"] < flagged_before == restored["flagged"],
          f"flagged {flagged_before} → {raised['flagged']} at τ {higher} → {restored['flagged']} after reset")

    # Worked cases stay listed after their customer drops below the threshold, so ≥ rather than ==.
    cases = api.call("GET", "/cases?page_size=1")
    check("cases opened for flags", cases["total"] >= restored["flagged"], f"{cases['total']} cases for {restored['flagged']} flagged customers")

    for kind in ("portfolio", "research"):
        if not health["reports"]["available"]:
            check(f"{kind} report", False, health["reports"]["detail"])
            continue
        report = api.call("POST", "/reports", {"kind": kind})
        pdf = api.raw("GET", f"/reports/{report['report_id']}/pdf")
        check(f"{kind} report", pdf.startswith(b"%PDF") and len(pdf) == report["bytes"], f"{report['report_id']}.pdf, {len(pdf):,} bytes")

    print(f"\n{'All checks passed' if not failures else f'{len(failures)} check(s) failed: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
