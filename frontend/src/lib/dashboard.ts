export function formatMaybeNumber(value: number | null | undefined, decimals = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }

  return value.toFixed(decimals);
}

// Keep these thresholds aligned with backend/services/model.py.
// If HIGH_RISK_THRESHOLD or MEDIUM_RISK_THRESHOLD change there, update this helper too.
export function riskTierFromProbability(probability: number): "high" | "medium" | "low" {
  if (probability >= 0.7) {
    return "high";
  }

  if (probability >= 0.4) {
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
