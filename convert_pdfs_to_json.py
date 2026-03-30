from __future__ import annotations

import json
import re
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
PDF_DIR = ROOT / "pdf_input"
OUT_DIR = ROOT / "pramana_engine" / "data" / "external_json_output"


def chunk_text(text: str, max_chars: int = 1200) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end].strip())
        start = end

    return [c for c in chunks if c]


def pdf_to_json(pdf_path: Path, out_path: Path) -> None:
    reader = PdfReader(str(pdf_path))
    records: list[dict[str, object]] = []

    for page_idx, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        for chunk_idx, chunk in enumerate(chunk_text(raw_text), start=1):
            records.append(
                {
                    "id": f"{pdf_path.stem}_p{page_idx}_c{chunk_idx}",
                    "source": f"{pdf_path.name}:page_{page_idx}",
                    "text": chunk,
                    "content": chunk,
                    "supports": ["testimony", "inference"],
                    "tags": ["pdf", pdf_path.stem.lower()],
                }
            )

    payload = {
        "source_file": pdf_path.name,
        "chunks": records,
    }

    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in: {PDF_DIR}")
        print("Place your PDF files there and run again.")
        return

    converted = 0
    for pdf in pdf_files:
        out_file = OUT_DIR / f"{pdf.stem}.json"
        try:
            pdf_to_json(pdf, out_file)
            print(f"Converted: {pdf.name} -> {out_file.name}")
            converted += 1
        except Exception as exc:
            print(f"Failed: {pdf.name} ({exc})")

    print(f"Done. Converted {converted}/{len(pdf_files)} PDFs.")
    print(f"Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
