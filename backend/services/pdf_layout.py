"""The PDF page layout for reports (fpdf2).

Imported only when a report is rendered (``backend.services.reports``), so an old or
missing PDF library disables reports, which /api/health explains, instead of the API.
Built with core fonts only (no network, no font files). Text is limited to Latin-1,
so a few typographic characters are replaced before rendering.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from fpdf import FPDF, FontFace
from fpdf.enums import TableCellFillMode, XPos, YPos

INK, INK_2, LINE, NIGHT, COBALT = (20, 22, 27), (79, 85, 97), (228, 226, 220), (16, 19, 23), (35, 70, 200)
_REPLACEMENTS = {"→": "->", "—": "-", "–": "-", "τ": "threshold ", "≈": "~", "’": "'", "“": '"', "”": '"', "…": "..."}


def _text(value: Any) -> str:
    text = str(value)
    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


class Report(FPDF):
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
