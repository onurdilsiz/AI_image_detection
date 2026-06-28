#!/usr/bin/env python3
"""Render report.md -> report.pdf with embedded figures (reportlab).

Lightweight parser for the controlled markdown in report.md: headings, tables,
bullet lists, fenced code, images (![alt](path)) and **bold**/`code`/*italic*
inline spans. Run from the project root:  python report_build.py
"""
from __future__ import annotations
import html
import os
import re

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, Preformatted, HRFlowable)

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "report.md")
OUT = os.path.join(HERE, "report.pdf")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9.2,
                          leading=12.4, spaceAfter=4))
styles.add(ParagraphStyle("H1x", parent=styles["Title"], fontSize=17, spaceAfter=6))
styles.add(ParagraphStyle("H2x", parent=styles["Heading2"], fontSize=12.5,
                          spaceBefore=8, spaceAfter=3, textColor=colors.HexColor("#1a3c66")))
styles.add(ParagraphStyle("H3x", parent=styles["Heading3"], fontSize=10.5,
                          spaceBefore=5, spaceAfter=2))
styles.add(ParagraphStyle("Cap", parent=styles["BodyText"], fontSize=8,
                          leading=10, textColor=colors.grey, spaceAfter=8))
styles.add(ParagraphStyle("Bul", parent=styles["Body"], leftIndent=12,
                          bulletIndent=2, spaceAfter=2))


def fmt(text: str) -> str:
    """Inline markdown -> reportlab mini-HTML."""
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    return text


def is_table_sep(line: str) -> bool:
    return bool(re.fullmatch(r"\|[\s:|-]+\|", line.strip()))


def build():
    with open(SRC, encoding="utf-8") as fh:
        lines = fh.read().split("\n")

    story = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        s = line.strip()

        if not s:
            story.append(Spacer(1, 3)); i += 1; continue

        if s == "---":
            story.append(HRFlowable(width="100%", thickness=0.6,
                                    color=colors.HexColor("#cccccc"),
                                    spaceBefore=4, spaceAfter=4)); i += 1; continue

        # Images: ![alt](path)
        m = re.match(r"!\[(.*?)\]\((.+?)\)", s)
        if m:
            alt, path = m.group(1), m.group(2)
            fpath = os.path.join(HERE, path)
            if os.path.exists(fpath):
                from PIL import Image as PILImage
                iw, ih = PILImage.open(fpath).size
                maxw = 6.4 * inch
                maxh = 3.1 * inch
                scale = min(maxw / iw, maxh / ih)
                story.append(Image(fpath, iw * scale, ih * scale))
                if alt:
                    story.append(Paragraph(fmt(alt), styles["Cap"]))
            i += 1; continue

        # Fenced code block
        if s.startswith("```"):
            code = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                code.append(lines[i]); i += 1
            i += 1
            story.append(Preformatted("\n".join(code),
                         ParagraphStyle("code", fontName="Courier", fontSize=8,
                                        leading=9.5, backColor=colors.HexColor("#f4f4f4"))))
            story.append(Spacer(1, 4)); continue

        # Headings
        if s.startswith("#### "):
            story.append(Paragraph(fmt(s[5:]), styles["H3x"])); i += 1; continue
        if s.startswith("### "):
            story.append(Paragraph(fmt(s[4:]), styles["H3x"])); i += 1; continue
        if s.startswith("## "):
            story.append(Paragraph(fmt(s[3:]), styles["H2x"])); i += 1; continue
        if s.startswith("# "):
            story.append(Paragraph(fmt(s[2:]), styles["H1x"])); i += 1; continue

        # Tables
        if s.startswith("|") and i + 1 < n and is_table_sep(lines[i + 1]):
            header = [c.strip() for c in s.strip("|").split("|")]
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            data = [[Paragraph(fmt(c), ParagraphStyle("th", parent=styles["Body"],
                     fontSize=8.5, textColor=colors.white)) for c in header]]
            for r in rows:
                data.append([Paragraph(fmt(c), ParagraphStyle("td", parent=styles["Body"],
                             fontSize=8.5)) for c in r])
            tbl = Table(data, hAlign="LEFT")
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3c66")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5f9")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(tbl); story.append(Spacer(1, 6)); continue

        # Bullet list
        if s.startswith("- "):
            story.append(Paragraph(fmt(s[2:]), styles["Bul"], bulletText="\u2022"))
            i += 1; continue

        # Plain paragraph (gather wrapped lines until blank/structural)
        para = [s]
        i += 1
        while i < n and lines[i].strip() and not re.match(
                r"^(#|\||-\s|!\[|```|---)", lines[i].strip()):
            para.append(lines[i].strip()); i += 1
        story.append(Paragraph(fmt(" ".join(para)), styles["Body"]))

    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=0.7 * inch,
                            rightMargin=0.7 * inch, topMargin=0.6 * inch,
                            bottomMargin=0.6 * inch, title="AMLS 2026 Report")
    doc.build(story)
    print(f"[report] wrote {OUT}")


if __name__ == "__main__":
    build()
