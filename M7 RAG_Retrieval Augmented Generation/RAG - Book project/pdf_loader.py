"""
pdf_loader.py
─────────────
Extract text from the Deep Learning PDF using PyMuPDF (fitz).

CHANGES vs. original
─────────────────────
① Pages now joined with "\\f" (form-feed) instead of "\\n\\n".
   The chunking.clean_text() function splits on \\f to track which
   page number each character came from — critical for source citations.
   Using "\\n\\n" discards all page boundary information entirely.

② read_pdf_text (pypdf version) kept for compatibility but updated to
   also join with \\f.

Install:  pip install pymupdf pypdf
"""

import re
import fitz   # PyMuPDF  — primary extractor


def read_pdf_text(pdf_path: str) -> str:
    """
    Extract text from a PDF using PyMuPDF.
    Pages are separated by \\f so chunking.clean_text() can track page numbers.

    Args:
        pdf_path: Path to the .pdf file.

    Returns:
        Full document text with \\f between pages.
    """
    doc = fitz.open(pdf_path)
    # Join with \f — NOT \n\n — so clean_text() can split on page boundaries
    text = "\f".join(page.get_text() for page in doc)
    doc.close()
    return text


# ── Legacy pypdf version (kept for backward compatibility) ────────────────────
try:
    from pypdf import PdfReader as _PdfReader

    def read_pdf_text_pypdf(pdf_path: str) -> str:
        """
        Fallback extractor using pypdf (less accurate for complex layouts).
        Also joins pages with \\f for consistent page tracking.
        """
        reader = _PdfReader(pdf_path)
        pages_text = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            txt = txt.replace("\x00", " ")
            txt = re.sub(r"[ \t]+", " ", txt)
            pages_text.append(txt)
        return "\f".join(pages_text)

except ImportError:
    pass  # pypdf optional — fitz is the primary extractor
