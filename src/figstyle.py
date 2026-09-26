"""Figure style shared by the evidence scripts (the palette and settings of scripts/make_thesis_figures.py)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "thesis-figures"

# Categorical slots in fixed order; the last two extend the five pipelines' colours for the added
# pipelines (SMOTE+ENN on raw readings, the deep baseline). Light slots always carry direct labels.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET, SLATE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                                      "#7d5bd6", "#6b6a66")
INK, INK2, GRID, SURFACE, MUTED = "#0b0b0b", "#52514e", "#e6e5e1", "#ffffff", "#b9b8b2"
PIPELINE_COLORS = {
    "proposed": BLUE, "xgboost": ORANGE, "xgboost_default": AQUA, "random_forest_smote": YELLOW,
    "logistic_regression_smote": MAGENTA, "xgboost_smote_enn_raw": VIOLET, "wide_deep_cnn": SLATE,
}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.dpi": 300, "lines.linewidth": 2,
})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)
