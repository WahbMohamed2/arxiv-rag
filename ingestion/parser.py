# ingestion/parser.py
import json
import re
import requests
from pathlib import Path
from loguru import logger
from config import PARSED_DIR, GROBID_HOST


# ── Grobid ─────────────────────────────────────────────────────────────────

def parse_with_grobid(pdf_path: Path) -> dict:
    """
    Send PDF to Grobid, get back TEI-XML.
    We extract: title, authors, abstract, sections, references.
    """
    url = f"{GROBID_HOST}/api/processFulltextDocument"
    try:
        with open(pdf_path, "rb") as f:
            resp = requests.post(
                url,
                files={"input": f},
                data={"includeRawCitations": "1"},
                timeout=120,
            )
        resp.raise_for_status()
        tei = resp.text

        section_titles = re.findall(r'<head[^>]*>(.*?)</head>', tei, re.DOTALL)
        section_titles = [s.strip() for s in section_titles if s.strip()]

        ref_count = len(re.findall(r'<biblStruct', tei))

        return {
            "grobid_tei":      tei,
            "section_titles":  section_titles,
            "reference_count": ref_count,
            "status":          "ok",
        }

    except requests.exceptions.Timeout:
        logger.error(f"Grobid timeout on {pdf_path.name}")
        return {"grobid_tei": None, "section_titles": [], "reference_count": 0, "status": "timeout"}
    except Exception as e:
        logger.error(f"Grobid failed on {pdf_path.name}: {e}")
        return {"grobid_tei": None, "section_titles": [], "reference_count": 0, "status": "failed"}


# ── PyMuPDF + pdfplumber ────────────────────────────────────────────────────

def parse_with_marker(pdf_path: Path) -> dict:
    """
    Extract text from PDF using PyMuPDF + pdfplumber.
    No GPU needed. Fast. Works perfectly on arXiv born-digital PDFs.
    """
    import fitz  # PyMuPDF
    import pdfplumber

    try:
        logger.info(f"Extracting text from {pdf_path.name} ...")

        # PyMuPDF pass — gets clean text with layout order
        doc = fitz.open(str(pdf_path))
        full_text_pages = []

        for page in doc:
            text = page.get_text("text")
            if text.strip():
                full_text_pages.append(text)

        doc.close()
        full_text = "\n".join(full_text_pages)

        # pdfplumber pass — better for two-column layout cleanup
        with pdfplumber.open(str(pdf_path)) as pdf:
            plumber_pages = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    plumber_pages.append(text)

        plumber_text = "\n".join(plumber_pages)
        if len(plumber_text) > len(full_text):
            full_text = plumber_text

        sections = _split_markdown_into_sections(full_text)

        return {
            "markdown": full_text,
            "sections": sections,
            "metadata": {},
            "status":   "ok",
        }

    except Exception as e:
        logger.error(f"PDF extraction failed on {pdf_path.name}: {e}")
        return {"markdown": None, "sections": [], "metadata": {}, "status": "failed"}


# ── Section Splitter ────────────────────────────────────────────────────────

def _split_markdown_into_sections(text: str) -> list[dict]:
    """
    Split plain text (arXiv PDFs) into sections.
    Detects headings like: '1 Introduction', 'INTRODUCTION', '2.1 Related Work'
    """
    if not text:
        return []

    heading_pattern = re.compile(
        r'^(\d+\.?\d*\.?\s+[A-Z][^\n]{2,50}|[A-Z][A-Z\s]{4,50})$',
        re.MULTILINE
    )

    matches = list(heading_pattern.finditer(text))

    if not matches:
        return [{"heading": "body", "text": text.strip()}]

    sections = []

    # Text before first heading
    if matches[0].start() > 100:
        preamble = text[:matches[0].start()].strip()
        if preamble:
            sections.append({"heading": "preamble", "text": preamble})

    # Each section runs from its heading to the next
    for i, match in enumerate(matches):
        heading = re.sub(r'\s+', ' ', match.group(0).strip())
        start   = match.end()
        end     = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body    = text[start:end].strip()

        if len(body) > 80:
            sections.append({"heading": heading, "text": body})

    return sections


# ── Orchestrator ───────────────────────────────────────────────────────────

def parse_paper(pdf_path: Path, paper_meta: dict) -> dict | None:
    """
    Full parse pipeline for one paper:
      1. Grobid  → section titles, reference count, TEI-XML
      2. PyMuPDF → clean text, split into sections
      3. Merge   → one JSON record saved to PARSED_DIR

    Returns the merged record, or None on total failure.
    """
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PARSED_DIR / f"{paper_meta['arxiv_id']}.json"

    if out_path.exists():
        logger.info(f"Already parsed, loading: {paper_meta['arxiv_id']}")
        return json.loads(out_path.read_text(encoding="utf-8"))

    logger.info(f"Parsing: {paper_meta['arxiv_id']}")

    grobid = parse_with_grobid(pdf_path)
    marker = parse_with_marker(pdf_path)

    if grobid["status"] == "failed" and marker["status"] == "failed":
        logger.error(f"Total parse failure: {paper_meta['arxiv_id']}")
        return None

    record = {
        # Original arXiv metadata
        **paper_meta,

        # Text extraction output
        "markdown":  marker["markdown"],
        "sections":  marker["sections"],

        # Grobid output
        "section_titles":  grobid["section_titles"],
        "reference_count": grobid["reference_count"],
        "grobid_tei":      grobid["grobid_tei"],

        # Parse health
        "parse_status": {
            "marker": marker["status"],
            "grobid": grobid["status"],
        },
    }

    out_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    logger.success(f"Saved parsed record → {out_path}")
    return record