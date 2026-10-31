"""Word (.docx) output for transcripts, set up so Telugu and other Indic scripts render properly."""
from __future__ import annotations

import datetime
import os

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

from . import __version__

# Language code -> (display name, Word language tag, font). Nirmala UI ships with Windows 8+ and
# covers all Indian scripts; Word on a Mac substitutes its own Telugu font.
LANGUAGES = {
    "te": ("Telugu", "te-IN", "Nirmala UI"),
}
DEFAULT_LANGUAGE = ("", "en-US", "Nirmala UI")

BODY_PT = 12
NOTE_PT = 9
GREY = RGBColor(0x70, 0x70, 0x70)
ENGLISH_BLUE = RGBColor(0x1F, 0x4E, 0x79)

# Paragraph grouping: start a new paragraph after a pause once a paragraph has some substance,
# and never let one run longer than PARAGRAPH_MAX_S of speech.
PARAGRAPH_MIN_S = 20
PARAGRAPH_MAX_S = 60
PAUSE_S = 0.8


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def group_segments(segments) -> list[list]:
    """Merge short speech chunks into paragraphs (lists of consecutive segments)."""
    paragraphs: list[list] = []
    current: list = []
    for seg in segments:
        if current:
            length = current[-1].end - current[0].start
            pause = seg.start - current[-1].end
            if length >= PARAGRAPH_MAX_S or (length >= PARAGRAPH_MIN_S and pause >= PAUSE_S):
                paragraphs.append(current)
                current = []
        current.append(seg)
    if current:
        paragraphs.append(current)
    return paragraphs


def group_paragraphs(segments) -> list[tuple[float, str]]:
    """(start_seconds, text) for each paragraph."""
    return [(p[0].start, " ".join(s.text for s in p)) for p in group_segments(segments)]


def _set_default_font(document, font: str, lang_tag: str) -> None:
    """Make the Normal style use `font` for every script, sized for complex scripts too."""
    rpr = document.styles["Normal"].element.get_or_add_rPr()
    rfonts = rpr.get_or_add_rFonts()
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), font)
    for tag in ("w:sz", "w:szCs"):  # Word sizes Indic text with szCs, not sz
        el = rpr.find(qn(tag))
        if el is None:
            el = OxmlElement(tag)
            rpr.append(el)
        el.set(qn("w:val"), str(BODY_PT * 2))
    lang = rpr.find(qn("w:lang"))
    if lang is None:
        lang = OxmlElement("w:lang")
        rpr.append(lang)
    lang.set(qn("w:val"), "en-US")
    lang.set(qn("w:bidi"), lang_tag)  # "bidi" holds the complex-script language, Telugu included


def _note(paragraph, text: str) -> None:
    run = paragraph.add_run(text)
    run.font.size = Pt(NOTE_PT)
    run.font.color.rgb = GREY


def write_document(transcript, path: str, english: list[str] | None = None) -> str:
    """Write the transcript; `english`, if given, has one translation per group_paragraphs() entry."""
    name, lang_tag, font = LANGUAGES.get(transcript.language, DEFAULT_LANGUAGE)
    title = os.path.splitext(os.path.basename(transcript.source))[0]
    today = datetime.date.today().isoformat()

    doc = Document()
    _set_default_font(doc, font, lang_tag)
    props = doc.core_properties
    props.title = title
    props.author = "VaniScribe"
    props.language = lang_tag
    props.comments = f"Automatic transcript of {os.path.basename(transcript.source)}"

    doc.add_heading(title, level=1)
    info = doc.add_paragraph()
    _note(info, f"Source: {os.path.basename(transcript.source)}   ·   Length: {_clock(transcript.duration)}"
                f"   ·   Language: {name or transcript.language}")
    info.add_run().add_break()
    _note(info, f"Transcribed offline by VaniScribe {__version__} on {today} (model: {transcript.model}).")
    info.add_run().add_break()
    _note(info, "This is an automatic transcript and may contain mistakes, especially in names and "
                "Sanskrit verses. The time in [brackets] shows where each paragraph starts in the recording.")
    if english:
        info.add_run().add_break()
        _note(info, "The English in blue italics under each paragraph is a rough machine translation "
                    "(NLLB-200), made offline; read it for the gist only.")

    paragraphs = group_paragraphs(transcript.segments)
    if not paragraphs:
        doc.add_paragraph("(No speech was recognised in this recording.)")
    for i, (start, text) in enumerate(paragraphs):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4 if english else 8)
        p.paragraph_format.line_spacing = 1.25
        stamp = p.add_run(f"[{_clock(start)}]  ")
        stamp.font.size = Pt(NOTE_PT)
        stamp.font.color.rgb = GREY
        p.add_run(text)
        if english and i < len(english) and english[i]:
            e = doc.add_paragraph()
            e.paragraph_format.space_after = Pt(12)
            e.paragraph_format.left_indent = Pt(18)
            run = e.add_run(english[i])
            run.italic = True
            run.font.color.rgb = ENGLISH_BLUE

    tmp = path + ".tmp"
    doc.save(tmp)
    os.replace(tmp, path)  # never leave a half-written document behind
    return path
