export function formatMaybeNumber(value: number | null | undefined, decimals = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }

  return value.toFixed(decimals);
}

// Fallback cutoffs, matching backend/services/model.py. Prefer the
// risk_tier_thresholds returned by /api/model/metrics for the deployed model.
export const DEFAULT_RISK_TIER_THRESHOLDS = { high: 0.7, medium: 0.4 };

export function riskTierFromProbability(
  probability: number,
  thresholds: { high: number; medium: number } = DEFAULT_RISK_TIER_THRESHOLDS,
): "high" | "medium" | "low" {
  if (probability >= thresholds.high) {
    return "high";
  }

  if (probability >= thresholds.medium) {
    return "medium";
  }

  return "low";
}

export function riskTierLabel(tier: string): string {
  switch (tier) {
    case "high":
      return "High";
    case "medium":
      return "Medium";
    case "low":
      return "Low";
    default:
      return "Unknown";
  }
}

export function riskTierClasses(tier: string): string {
  switch (tier) {
    case "high":
      return "border-danger bg-danger-bg text-danger";
    case "medium":
      return "border-warning bg-warning-bg text-warning";
    case "low":
      return "border-success bg-success-bg text-success";
    default:
      return "border-border bg-surface-alt text-secondary";
  }
}
