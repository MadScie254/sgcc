"""PDF reports: portfolio overview, uploaded-dataset results, and a customer case file.

Built with fpdf2 core fonts only (no network, no font files). Text is limited to
Latin-1, so a few typographic characters are replaced before rendering.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence

import fpdf
from fpdf import FPDF, FontFace
from fpdf.enums import TableCellFillMode, XPos, YPos

from .config import get_paths
from .data import get_customer_timeseries
from .datasets import get_dataset, load_dataset, summarize
from .errors import NotFoundError
from .model import explain_customer, get_decision_threshold, get_model_metrics
from .operations import get_case, list_cases
from .storage import read_json, write_json

MIN_FPDF_VERSION = (2, 7, 8)
_ID_PATTERN = re.compile(r"(portfolio|dataset|case)-[0-9]{14}-[0-9a-f]{8}")
_INDEX_LOCK = Lock()
MAX_REPORTS = 500  # older PDFs are deleted

INK, INK_2, LINE, NIGHT, COBALT = (20, 22, 27), (79, 85, 97), (228, 226, 220), (16, 19, 23), (35, 70, 200)
_REPLACEMENTS = {"→": "->", "—": "-", "–": "-", "τ": "threshold ", "≈": "~", "’": "'", "“": '"', "”": '"', "…": "..."}


class ReportsUnavailable(RuntimeError):
    """PDF reports cannot be produced in this environment; the message says how to fix it."""


def _version(text: str) -> tuple:
    return tuple(int(part) for part in re.findall(r"\d+", text)[:3])


def reports_status() -> Dict[str, Any]:
    installed = getattr(fpdf, "FPDF_VERSION", "unknown")
    ok = installed != "unknown" and _version(installed) >= MIN_FPDF_VERSION
    return {
        "available": ok,
        "fpdf_version": installed,
        "detail": None if ok else (
            f"PDF reports need fpdf2 >= {'.'.join(map(str, MIN_FPDF_VERSION))} (installed: {installed}). "
            "Run: pip install -r requirements.lock"
        ),
    }


def _text(value: Any) -> str:
    text = str(value)
    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


def _pct(value: Optional[float], digits: int = 1) -> str:
    return "-" if value is None else f"{value * 100:.{digits}f}%"


def _num(value: Optional[float], digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


class _Report(FPDF):
    def __init__(self, title: str, subtitle: str, footer_note: str):
        super().__init__(orientation="portrait", unit="mm", format="A4")
        self.report_title, self.subtitle, self.footer_note = title, subtitle, footer_note
        self.set_margins(16, 16, 16)
        self.set_auto_page_break(auto=True, margin=18)
        self.set_title(_text(title))
        self.set_author("GridSentinel")
        self.add_page()

    def header(self) -> None:
        if self.page_no() == 1:
            self.set_fill_color(*NIGHT)
            self.rect(0, 0, self.w, 34, style="F")
            self.set_xy(16, 9)
            self.set_text_color(243, 242, 238)
            self.set_font("Helvetica", "B", 9)
            self.cell(0, 5, "GRIDSENTINEL  ·  REVENUE PROTECTION", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_font("Helvetica", "B", 18)
            self.cell(0, 9, _text(self.report_title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_font("Helvetica", "", 9)
            self.set_text_color(201, 205, 212)
            self.cell(0, 5, _text(self.subtitle), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_y(42)
        else:
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*INK_2)
            self.cell(0, 5, _text(self.report_title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.ln(2)
        self.set_text_color(*INK)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*INK_2)
        self.cell(0, 5, _text(self.footer_note), align="L")
        self.cell(0, 5, f"Page {self.page_no()}/{{nb}}", align="R")

    def heading(self, text: str) -> None:
        if self.get_y() > self.h - 50:
            self.add_page()
        self.ln(3)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*INK)
        self.cell(0, 7, _text(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*LINE)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2.5)

    def paragraph(self, text: str) -> None:
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*INK_2)
        self.multi_cell(0, 5, _text(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*INK)
        self.ln(1)

    def facts(self, pairs: Sequence[tuple], columns: int = 2) -> None:
        """Label/value pairs laid out in ``columns`` column pairs."""
        self.set_font("Helvetica", "", 9)
        self.set_fill_color(255, 255, 255)
        label_style, value_style = FontFace(color=INK_2), FontFace(emphasis="B", color=INK)
        widths = [34, 55] * columns if columns == 2 else [24, 36] * columns
        padded = list(pairs) + [("", "")] * (-len(pairs) % columns)
        with self.table(first_row_as_headings=False, col_widths=widths, borders_layout="NONE",
                        line_height=5.6, text_align="LEFT") as table:
            for i in range(0, len(padded), columns):
                row = table.row()
                for label, value in padded[i:i + columns]:
                    row.cell(_text(label), style=label_style)
                    row.cell(_text(value), style=value_style)
        self.ln(1)

    def grid(self, headings: Sequence[str], rows: Sequence[Sequence[Any]], widths: Sequence[float],
             align: Optional[Sequence[str]] = None) -> None:
        self.set_font("Helvetica", "", 8.5)
        self.set_fill_color(255, 255, 255)
        with self.table(col_widths=widths, line_height=5.4, borders_layout="HORIZONTAL_LINES",
                        cell_fill_color=(246, 245, 241), cell_fill_mode=TableCellFillMode.ROWS,
                        headings_style=FontFace(emphasis="B", color=INK, fill_color=(236, 234, 228)),
                        text_align=tuple(align) if align else "LEFT") as table:
            for values in [headings, *rows]:
                row = table.row()
                for value in values:
                    row.cell(_text(value))
        self.ln(2)

    def monthly_chart(self, months: List[tuple], height: float = 38) -> None:
        """Bars of monthly mean kWh/day; months without readings are shaded."""
        x0, width = self.l_margin, self.w - self.l_margin - self.r_margin
        y0 = self.get_y() + 2
        top = max((v for _, v in months if v is not None), default=1.0) or 1.0
        step = width / max(len(months), 1)
        self.set_draw_color(*LINE)
        for i, (_, value) in enumerate(months):
            x = x0 + i * step
            if value is None:
                self.set_fill_color(251, 231, 223)
                self.rect(x, y0, step, height, style="F")
            else:
                h = height * value / top
                self.set_fill_color(*COBALT)
                self.rect(x + step * 0.15, y0 + height - h, step * 0.7, h, style="F")
        self.line(x0, y0 + height, x0 + width, y0 + height)
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*INK_2)
        self.set_xy(x0, y0 + height + 1)
        self.cell(width / 2, 4, _text(months[0][0] if months else ""), align="L")
        self.cell(width / 2, 4, _text(months[-1][0] if months else ""), align="R")
        self.set_xy(x0, y0 + height + 5)
        self.cell(0, 4, _text(f"Monthly mean, kWh/day (axis top {top:.1f}); shaded months had no meter readings."),
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*INK)
        self.ln(2)


# ---------------------------------------------------------------------------
# Report content
# ---------------------------------------------------------------------------

def _footer(metrics: Dict[str, Any]) -> str:
    return f"Model xgb v{metrics['model_version']}  ·  threshold in service {metrics['threshold']:.3f}"


def _portfolio(generated: str) -> tuple:
    metrics = get_model_metrics()
    cases = list_cases(status=None, tier=None, search=None, page=1, page_size=30)
    m, cm = metrics["metrics"], metrics["confusion_matrix"]
    pdf = _Report("Portfolio risk report", f"Generated {generated}  ·  {metrics['customers_monitored']:,} customers", _footer(metrics))

    pdf.heading("Summary")
    pdf.paragraph(
        f"{metrics['flagged']:,} of {metrics['customers_monitored']:,} served customers are at or above the operating "
        f"threshold ({metrics['threshold']:.3f}); {metrics['risk_tier_distribution']['high']:,} are high risk. "
        f"Among flagged customers, {cm['tp']:,} carry a theft label in the dataset "
        f"({_pct(cm['tp'] / max(cm['tp'] + cm['fp'], 1))} hit rate against a {_pct(metrics['base_rate'])} base rate)."
    )
    pdf.facts([
        ("Customers monitored", f"{metrics['customers_monitored']:,}"), ("Flagged", f"{metrics['flagged']:,}"),
        ("High risk", f"{metrics['risk_tier_distribution']['high']:,}"), ("Medium risk", f"{metrics['risk_tier_distribution']['medium']:,}"),
        ("Threshold in service", f"{metrics['threshold']:.3f}"), ("Trained threshold", f"{metrics['trained_threshold']:.3f}"),
    ])

    pdf.heading("Model quality (hold-out test set)")
    pdf.facts([
        ("ROC-AUC", _num(m["auc"])), ("PR-AUC", _num(m["pr_auc"])),
        ("Precision", _num(m["precision"])), ("Recall", _num(m["recall"])),
        ("F1", _num(m["f1"])), ("MCC", _num(m["mcc"])),
    ])

    pdf.heading("Outcome on served customers at the operating threshold")
    pdf.grid(["", "Actually theft", "Actually honest"],
             [["Flagged", f"{cm['tp']:,} caught", f"{cm['fp']:,} wasted visits"],
              ["Not flagged", f"{cm['fn']:,} missed", f"{cm['tn']:,} correctly cleared"]],
             widths=[40, 69, 69])

    pdf.heading("Case workflow")
    counts = cases["status_counts"]
    pdf.facts([(status.capitalize(), f"{counts[status]:,}") for status in counts], columns=2)

    pdf.heading(f"Top {len(cases['items'])} cases by theft probability")
    pdf.grid(["Rank", "Customer", "Probability", "Tier", "Status", "Strongest signal"],
             [[c["rank"], c["customer_id"][:16], f"{c['risk_score']:.3f}", c["risk_tier"], c["status"],
               f"{c['top_driver']['label']} ({c['top_driver']['display_value']})" if c["top_driver"] else "-"]
              for c in cases["items"]],
             widths=[12, 34, 20, 16, 22, 74], align=["RIGHT", "LEFT", "RIGHT", "LEFT", "LEFT", "LEFT"])
    return pdf, f"{metrics['customers_monitored']:,} customers at threshold {metrics['threshold']:.3f}"


_LABEL_NAMES = {0: "honest", 1: "theft"}


def _dataset(generated: str, dataset_id: str) -> tuple:
    item = get_dataset(dataset_id)
    threshold = get_decision_threshold()
    summary = summarize(load_dataset(dataset_id), threshold, top_n=20)
    metrics = get_model_metrics()
    pdf = _Report("Dataset scoring report", f"Generated {generated}  ·  {item['filename']}", _footer(metrics))

    pdf.heading("Dataset")
    pdf.facts([
        ("File", item["filename"]), ("Uploaded", item["uploaded_at"].replace("T", " ")[:16]),
        ("Format", "SGCC meter data" if summary["format"] == "consumption" else "Model features"),
        ("Customers", f"{summary['customers']:,}"),
        ("Days of readings" if summary["days"] else "Features found",
         f"{summary['days']:,}" if summary["days"] else f"{summary['features_found']} of {summary['features_expected']}"),
        ("Labels", "yes" if summary["labelled"] else "no"),
    ])

    pdf.heading("Results at the operating threshold")
    pdf.facts([
        ("Threshold", f"{threshold:.3f}"), ("Flagged", f"{summary['flagged']:,} ({_pct(summary['flagged'] / max(summary['customers'], 1))})"),
        ("High risk", f"{summary['tiers']['high']:,}"), ("Medium risk", f"{summary['tiers']['medium']:,}"),
        ("Mean probability", _num(summary["mean_probability"])), ("", ""),
    ])
    if summary["label_metrics"]:
        lm = summary["label_metrics"]
        pdf.heading("Agreement with the file's labels")
        pdf.facts([
            ("Labelled theft", f"{lm['theft']:,}"), ("Theft caught", f"{lm['caught']:,}"),
            ("Precision", _num(lm["precision"])), ("Recall", _num(lm["recall"])),
            ("ROC-AUC", _num(lm["roc_auc"])), ("PR-AUC", _num(lm["pr_auc"])),
        ])
    if summary["format"] == "features" and summary["features_found"] < summary["features_expected"]:
        pdf.paragraph(f"{summary['features_expected'] - summary['features_found']} model features were absent "
                      "and treated as missing readings.")

    pdf.heading(f"Top {len(summary['top'])} customers by theft probability")
    headings = ["#", "Customer", "Probability", "Tier"] + (["Label"] if summary["labelled"] else [])
    rows = [[i + 1, t["customer_id"][:32], f"{t['probability']:.3f}", t["risk_tier"]] + ([_LABEL_NAMES[t["label"]]] if summary["labelled"] else [])
            for i, t in enumerate(summary["top"])]
    pdf.grid(headings, rows, widths=[10, 80, 30, 30] + ([28] if summary["labelled"] else []))
    return pdf, item["filename"]


def _case(generated: str, customer_id: str) -> tuple:
    case = get_case(customer_id)
    explanation = explain_customer(customer_id)
    series = get_customer_timeseries(customer_id)
    metrics = get_model_metrics()
    pdf = _Report("Case file", f"Generated {generated}  ·  customer {customer_id}", _footer(metrics))

    points = series["points"]
    observed = [p for p in points if p["kwh"] is not None]
    longest = run = 0
    for p in points:
        run = run + 1 if p["kwh"] is None else 0
        longest = max(longest, run)
    months: Dict[str, List[float]] = {}
    for p in points:
        months.setdefault(p["date"][:7], [])
        if p["kwh"] is not None:
            months[p["date"][:7]].append(p["kwh"])

    pdf.heading("Assessment")
    pdf.facts([
        ("Customer", customer_id), ("Theft probability", f"{case['risk_score']:.3f}"),
        ("Rank", f"{case['rank']:,} of {case['population']:,}"), ("Risk tier", case["risk_tier"]),
        ("Flagged", "yes" if case["flagged"] else "no"), ("Case status", case["status"]),
    ])
    if case["note"]:
        pdf.paragraph(f"Analyst note: {case['note']}")

    pdf.heading("Meter readings")
    pdf.facts([
        ("First reading", observed[0]["date"] if observed else "-"), ("Last reading", observed[-1]["date"] if observed else "-"),
        ("Days read", f"{len(observed):,} of {len(points):,} ({_pct(len(observed) / max(len(points), 1), 0)})"),
        ("Longest gap", f"{longest:,} days"),
        ("Use while reporting", f"{sum(p['kwh'] for p in observed) / len(observed):.2f} kWh/day" if observed else "-"),
        ("Dataset label", "theft" if series["label"] == 1 else "honest"),
    ])
    pdf.monthly_chart([(key, sum(v) / len(v) if v else None) for key, v in months.items()])

    pdf.heading("Why the model scored this customer")
    pdf.paragraph("SHAP contributions in log-odds: positive values push towards theft, negative away from it. "
                  f"Starting from the base rate ({explanation['base_value']:+.2f} log-odds), the contributions add up to "
                  f"the predicted probability of {explanation['probability']:.3f}.")
    pdf.grid(["Signal", "Value", "Contribution"],
             [[c["label"], c["display_value"], f"{c['shap_value']:+.3f}"] for c in explanation["contributions"][:10]],
             widths=[82, 62, 34], align=["LEFT", "LEFT", "RIGHT"])

    pdf.heading("Case history")
    history = case["history"] or [{"at": "", "event": "No case activity yet."}]
    pdf.grid(["When (UTC)", "Event"], [[h["at"].replace("T", " ")[:16], h["event"]] for h in history], widths=[40, 138])
    return pdf, customer_id


# ---------------------------------------------------------------------------
# Index and files
# ---------------------------------------------------------------------------

def _reports_dir():
    path = get_paths()["reports"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_index() -> List[Dict[str, Any]]:
    return read_json(_reports_dir() / "index.json", [])


def generate_report(kind: str, subject_id: Optional[str] = None) -> Dict[str, Any]:
    """Render and store a report. `subject_id` is the dataset id or customer id it covers."""
    status = reports_status()
    if not status["available"]:
        raise ReportsUnavailable(status["detail"])
    now = datetime.now(timezone.utc)
    generated = now.strftime("%d %b %Y %H:%M UTC")
    if kind == "portfolio":
        pdf, subject = _portfolio(generated)
    elif kind == "dataset":
        pdf, subject = _dataset(generated, str(subject_id))
    elif kind == "case":
        pdf, subject = _case(generated, str(subject_id))
    else:
        raise ValueError(f"Unknown report kind {kind!r}")

    report_id = f"{kind}-{now.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    path = _reports_dir() / f"{report_id}.pdf"
    pdf.output(str(path))
    entry = {"report_id": report_id, "kind": kind, "title": pdf.report_title, "subject": subject,
             "created_at": now.isoformat(timespec="seconds"), "bytes": path.stat().st_size}
    with _INDEX_LOCK:
        index = [entry, *_read_index()]
        for expired in index[MAX_REPORTS:]:
            if _ID_PATTERN.fullmatch(str(expired.get("report_id"))):
                (_reports_dir() / f"{expired['report_id']}.pdf").unlink(missing_ok=True)
        write_json(_reports_dir() / "index.json", index[:MAX_REPORTS])
    return entry


def list_reports(limit: int = 50) -> List[Dict[str, Any]]:
    with _INDEX_LOCK:
        return _read_index()[:limit]


def report_path(report_id: str):
    """Path of a generated report; NotFoundError if the id is malformed or the file is gone."""
    if not _ID_PATTERN.fullmatch(report_id):
        raise NotFoundError(f"Unknown report: {report_id}")
    path = _reports_dir() / f"{report_id}.pdf"
    if not path.is_file():
        raise NotFoundError(f"Unknown report: {report_id}")
    return path
