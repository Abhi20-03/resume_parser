from __future__ import annotations

import io
import re
from typing import Optional

from docx import Document
import pdfplumber


EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_RE = re.compile(
    r"(\+?\d{1,3}[\s.-]?)?(\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}"
)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        pages = pdf.pages
        text_parts = []
        for p in pages:
            page_text = p.extract_text() or ""
            if not page_text.strip():
                # Fallback for multi-column / complex layouts:
                # build text from words when extract_text() misses content.
                try:
                    words = p.extract_words() or []
                    if words:
                        page_text = " ".join(w.get("text", "") for w in words).strip()
                except Exception:
                    page_text = ""
            if page_text.strip():
                text_parts.append(page_text)
        return "\n".join(text_parts).strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    parts = []
    for para in doc.paragraphs:
        if para.text and para.text.strip():
            parts.append(para.text)
    return "\n".join(parts).strip()


def extract_text_from_doc_best_effort(file_bytes: bytes) -> Optional[str]:
    """
    Best-effort DOC parsing.

    Note: this requires `textract` plus the relevant OS dependencies.
    """
    try:
        import textract  # type: ignore
    except Exception:
        return None

    try:
        # textract works with file paths; we use a temporary file.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".doc", delete=True) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            text = textract.process(tmp.name)
        if isinstance(text, bytes):
            return text.decode("utf-8", errors="ignore").strip()
        return str(text).strip()
    except Exception:
        return None


def extract_contact_hints(text: str) -> dict:
    email_match = EMAIL_RE.search(text)
    phone_match = PHONE_RE.search(text)
    return {
        "email": email_match.group(0) if email_match else None,
        "phone": phone_match.group(0) if phone_match else None,
    }


def extract_text(file_bytes: bytes, filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    if lower.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    if lower.endswith(".doc"):
        extracted = extract_text_from_doc_best_effort(file_bytes)
        if extracted is not None:
            return extracted
        raise ValueError(
            "DOC parsing requires additional dependencies (textract). Please upload DOCX/PDF instead."
        )
    raise ValueError("Unsupported file type. Use .pdf, .docx, or .doc")


def truncate_text(text: str, max_chars: int) -> str:
    # Avoid cutting in the middle of a word when possible.
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 1]
    last_space = cut.rfind(" ")
    return cut[:last_space] if last_space > 0 else cut

