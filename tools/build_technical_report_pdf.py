from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer


def _latex_to_text(s: str) -> str:
    """Best-effort conversion for simple inline LaTeX used in the report."""
    # Inline math wrappers
    s = s.replace("\\(", "").replace("\\)", "")

    # A tiny subset of commands we use in this repo
    s = s.replace("\\in", " in ")
    s = s.replace("\\sum", "sum")
    s = s.replace("\\log", "log")

    # Ceil(...) common pattern
    s = re.sub(r"\\lceil\s*", "ceil(", s)
    s = re.sub(r"\s*\\rceil", ")", s)
    s = re.sub(r"ceil\(\s+", "ceil(", s)
    s = re.sub(r"\s+\)", ")", s)

    # Remove any remaining backslashes (e.g., subscripts) to keep the PDF readable.
    s = s.replace("\\", "")

    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _md_inline_to_rl(s: str) -> str:
    """Convert a small subset of Markdown inline syntax to ReportLab paragraph markup."""
    s = _latex_to_text(s)

    # Escape HTML special chars first; we'll re-inject our own tags below.
    s = escape(s)

    # Inline code: `code`
    def repl_code(m: re.Match[str]) -> str:
        txt = escape(m.group(1))
        return f'<font face="Courier">{txt}</font>'

    s = re.sub(r"`([^`]+)`", repl_code, s)

    # Bold: **text**
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)

    # Italic: *text*
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", s)

    return s


def _build_pdf(md_path: Path, pdf_path: Path) -> None:
    styles = getSampleStyleSheet()
    base = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        spaceAfter=6,
    )
    h1 = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        spaceBefore=12,
        spaceAfter=10,
    )
    h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        spaceBefore=10,
        spaceAfter=6,
    )
    h3 = ParagraphStyle(
        "H3",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        spaceBefore=8,
        spaceAfter=4,
    )
    mono = ParagraphStyle(
        "Mono",
        parent=base,
        fontName="Courier",
        fontSize=9.5,
        leading=12,
    )

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=LETTER,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.75 * inch,
        title="Technical Report",
    )

    story: list[object] = []

    lines = md_path.read_text(encoding="utf-8").splitlines()

    in_code = False
    code_buf: list[str] = []
    para_buf: list[str] = []

    def flush_paragraph() -> None:
        nonlocal para_buf
        if not para_buf:
            return
        text = " ".join(x.strip() for x in para_buf if x.strip())
        text = _md_inline_to_rl(text)
        story.append(Paragraph(text, base))
        para_buf = []

    def flush_code() -> None:
        nonlocal code_buf
        if not code_buf:
            return
        story.append(Preformatted("\n".join(code_buf), mono))
        story.append(Spacer(1, 6))
        code_buf = []

    for raw in lines:
        line = raw.rstrip("\n")

        if line.strip().startswith("```"):
            if in_code:
                flush_code()
                in_code = False
            else:
                flush_paragraph()
                in_code = True
            continue

        if in_code:
            code_buf.append(raw)
            continue

        if not line.strip():
            flush_paragraph()
            continue

        if line.startswith("# "):
            flush_paragraph()
            story.append(Paragraph(_md_inline_to_rl(line[2:].strip()), h1))
            continue
        if line.startswith("## "):
            flush_paragraph()
            story.append(Paragraph(_md_inline_to_rl(line[3:].strip()), h2))
            continue
        if line.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(_md_inline_to_rl(line[4:].strip()), h3))
            continue

        m_num = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if m_num:
            flush_paragraph()
            story.append(Paragraph("- " + _md_inline_to_rl(m_num.group(1).strip()), base))
            continue

        if line.lstrip().startswith("- "):
            flush_paragraph()
            story.append(Paragraph("- " + _md_inline_to_rl(line.lstrip()[2:].strip()), base))
            continue

        para_buf.append(line)

    flush_paragraph()
    flush_code()

    def on_page(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 9)
        canvas.setFillGray(0.35)
        canvas.drawString(
            doc_.leftMargin,
            0.5 * inch,
            'Tender/Auction Anomaly Testbed - Technical Report',
        )
        canvas.drawRightString(
            doc_.pagesize[0] - doc_.rightMargin,
            0.5 * inch,
            f"Page {doc_.page}",
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    md_path = root / "docs" / "TECHNICAL_REPORT.md"
    pdf_path = root / "docs" / "TECHNICAL_REPORT.pdf"
    _build_pdf(md_path, pdf_path)
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
