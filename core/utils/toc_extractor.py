import logging
import time
from typing import List, Optional

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions

from core.utils.llm_config import get_chat_model
from core.schemas.toc import TOCDetection, TOCContent, TOCChapter

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 1.0


def load_document(pdf_path: str, max_pages: int = 10):
    """Load PDF using Docling with EasyOCR enabled."""
    ocr_options = EasyOcrOptions()

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = ocr_options

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    result = converter.convert(pdf_path, page_range=(1, max_pages))
    return result.document


def extract_pages(doc) -> List[str]:
    """Extract text per page using Docling provenance data."""
    page_numbers = sorted(doc.pages.keys())
    if not page_numbers:
        return []

    pages_content = {no: "" for no in page_numbers}

    for item, _level in doc.iterate_items():
        if hasattr(item, "text") and item.prov:
            pno = item.prov[0].page_no
            if pno in pages_content:
                pages_content[pno] += item.text + "\n"

    return [pages_content[no].strip() for no in page_numbers]


def _invoke_with_retry(structured_llm, prompt: str, max_retries: int = MAX_RETRIES):
    """Invoke LLM with retry logic."""
    for attempt in range(max_retries):
        try:
            return structured_llm.invoke(prompt)
        except Exception as e:
            logger.warning(
                f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return None


def llm_detect_toc(text: str, llm) -> bool:
    """Detect if a page contains a Table of Contents."""
    if not text or len(text.strip()) < 20:
        return False

    structured_llm = llm.with_structured_output(TOCDetection)

    prompt = """Analyze if this page is a Table of Contents (TOC).

POSITIVE SIGNS:
- Words like "Contents", "Table of Contents", "Chapter", "Section"
- Topics followed by page numbers (Roman numerals or integers)
- Hierarchical numbering (1.1, 2.3.4)
- Fix OCR spacing errors (e.g., 'Co m ple x ity' -> 'Complexity')

NEGATIVE SIGNS:
- Copyright pages, acknowledgments, prefaces without chapter lists
- Random titles without page mappings

TEXT:
{text}

Answer: is_toc=True or is_toc=False""".format(
        text=text[:3000]
    )

    result = _invoke_with_retry(structured_llm, prompt)
    return result.is_toc if result else False


def find_toc_pages(pages: List[str], llm) -> List[int]:
    """Find all TOC pages with gap tolerance."""
    toc_indices = []
    max_gap = 1
    gap_counter = 0

    logger.info(f"Scanning {len(pages)} pages for TOC...")

    for i, page_text in enumerate(pages):
        if not page_text.strip():
            continue

        is_toc = llm_detect_toc(page_text, llm)

        if is_toc:
            toc_indices.append(i)
            gap_counter = 0
            logger.debug(f"Page {i + 1}: TOC detected")
        elif toc_indices:
            gap_counter += 1
            if gap_counter > max_gap:
                break

    logger.info(f"Found {len(toc_indices)} TOC pages: {toc_indices}")
    return toc_indices


def extract_toc_content(pages: List[str], llm, toc_indices: List[int]) -> str:
    """Extract and clean TOC text from detected pages."""
    if not toc_indices:
        return ""

    toc_raw = "\n\n".join([pages[idx] for idx in toc_indices])

    structured_llm = llm.with_structured_output(TOCContent)

    prompt = """Clean and format this Table of Contents:
1. Fix OCR spacing errors
2. Maintain hierarchy (Chapter -> Section)
3. Preserve page numbers
4. Return cleaned text

RAW TOC:
{text}

Return: toc_text=<cleaned text>""".format(
        text=toc_raw[:5000]
    )

    result = _invoke_with_retry(structured_llm, prompt)
    return result.toc_text if result else toc_raw


def parse_toc_to_chapters(toc_text: str, llm) -> List[TOCChapter]:
    """Parse cleaned TOC into structured chapter objects."""
    if not toc_text:
        return []

    structured_llm = llm.with_structured_output(TOCContent)

    prompt = """Parse this Table of Contents into structured chapters.

TOC TEXT:
{text}

Return chapters with: number, title, page, subsections""".format(
        text=toc_text[:5000]
    )

    result = _invoke_with_retry(structured_llm, prompt)
    return result.chapters if result else []


def detect_page_index(toc_text: str, llm) -> bool:
    """Check if TOC contains page numbers."""
    if not toc_text:
        return False

    from core.schemas.toc import PageIndexDetection

    structured_llm = llm.with_structured_output(PageIndexDetection)

    prompt = "Does this TOC contain page numbers?\n\n{text}".format(
        text=toc_text[:2000]
    )
    result = _invoke_with_retry(structured_llm, prompt)
    return result.page_index_given_in_toc if result else False
