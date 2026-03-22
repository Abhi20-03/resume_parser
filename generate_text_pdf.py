from __future__ import annotations

import sys
from pathlib import Path
from textwrap import wrap

PAGE_WIDTH = 612
PAGE_HEIGHT = 792
LEFT_MARGIN = 54
TOP_MARGIN = 54
BOTTOM_MARGIN = 54
FONT_SIZE = 10
LEADING = 14
MAX_COLS = 88


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def paginate(lines: list[str]) -> list[list[str]]:
    usable_height = PAGE_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN
    lines_per_page = usable_height // LEADING
    pages: list[list[str]] = []
    for i in range(0, len(lines), lines_per_page):
        pages.append(lines[i : i + lines_per_page])
    return pages or [[]]


def markdown_to_lines(text: str) -> list[str]:
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            out.append("")
            continue
        if line.startswith("#"):
            line = line.lstrip("#").strip().upper()
        elif line.startswith("- "):
            line = "* " + line[2:]
        wrapped = wrap(line, width=MAX_COLS, replace_whitespace=False, drop_whitespace=False)
        if not wrapped:
            out.append("")
        else:
            out.extend(x.rstrip() for x in wrapped)
    return out


def build_content_stream(page_lines: list[str]) -> bytes:
    y = PAGE_HEIGHT - TOP_MARGIN
    commands = ["BT", f"/F1 {FONT_SIZE} Tf", f"{LEFT_MARGIN} {y} Td"]
    first = True
    for line in page_lines:
        if not first:
            commands.append(f"0 -{LEADING} Td")
        commands.append(f"({pdf_escape(line)}) Tj")
        first = False
    commands.append("ET")
    return "\n".join(commands).encode("latin-1", errors="replace")


def write_pdf(text: str, output_path: Path) -> None:
    lines = markdown_to_lines(text)
    pages = paginate(lines)

    objects: list[bytes] = []

    def add_object(data: bytes) -> int:
        objects.append(data)
        return len(objects)

    font_obj = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")
    placeholder_pages_obj = add_object(b"<< /Type /Pages /Count 0 /Kids [] >>")
    page_obj_ids: list[int] = []

    for page_lines in pages:
        stream = build_content_stream(page_lines)
        content_obj = add_object(
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"\nendstream"
        )
        page_obj = add_object(
            f"<< /Type /Page /Parent {placeholder_pages_obj} 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] /Resources << /Font << /F1 {font_obj} 0 R >> >> /Contents {content_obj} 0 R >>".encode("ascii")
        )
        page_obj_ids.append(page_obj)

    kids = " ".join(f"{obj_id} 0 R" for obj_id in page_obj_ids)
    objects[placeholder_pages_obj - 1] = f"<< /Type /Pages /Count {len(page_obj_ids)} /Kids [{kids}] >>".encode("ascii")
    catalog_obj = add_object(f"<< /Type /Catalog /Pages {placeholder_pages_obj} 0 R >>".encode("ascii"))

    chunks = [b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"]
    offsets = [0]
    current = len(chunks[0])

    for idx, obj in enumerate(objects, start=1):
        offsets.append(current)
        body = f"{idx} 0 obj\n".encode("ascii") + obj + b"\nendobj\n"
        chunks.append(body)
        current += len(body)

    xref_offset = current
    xref = [f"xref\n0 {len(objects) + 1}\n".encode("ascii"), b"0000000000 65535 f \n"]
    for off in offsets[1:]:
        xref.append(f"{off:010d} 00000 n \n".encode("ascii"))
    chunks.extend(xref)
    chunks.append(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_obj} 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )

    output_path.write_bytes(b"".join(chunks))


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python generate_text_pdf.py <input-md> <output-pdf>")
        return 1
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    text = input_path.read_text(encoding="utf-8")
    write_pdf(text, output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
