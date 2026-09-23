export function formatMaybeNumber(value: number | null | undefined, decimals = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }

  return value.toFixed(decimals);
}

// Keep aligned with risk_tier_for_probability in backend/services/model.py:
// "medium" starts at the model's decision threshold, "high" at HIGH_RISK_PROBABILITY.
export const HIGH_RISK_PROBABILITY = 0.6;

export function riskTierFromProbability(probability: number, threshold: number): "high" | "medium" | "low" {
  if (probability >= Math.max(HIGH_RISK_PROBABILITY, threshold)) {
    return "high";
  }

  if (probability >= threshold) {
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
