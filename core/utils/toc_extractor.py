from utils.llm_config import get_chat_model
from schemas.toc import TOCDetection, PageIndexDetection
from typing import List
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions

def load_document(pdf_path: str, max_pages: int = 10):
    """Loads PDF using Docling with EasyOCR enabled."""
    
    # You can specify languages here, e.g., lang=["en"]
    ocr_options = EasyOcrOptions() 
    
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = ocr_options
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    result = converter.convert(
        pdf_path, 
        page_range=(1, max_pages) 
    )

    return result.document
def extract_pages(doc) -> List[str]:
    """Correctly extracts text per page using Docling's provenance data."""
    # doc.pages is a dict mapping page_no (int) to page objects
    page_numbers = sorted(doc.pages.keys())
    if not page_numbers:
        return []

    # Initialize storage for each page found
    pages_content = {no: "" for no in page_numbers}

    # Iterate items in reading order
    for item, _level in doc.iterate_items():
        if hasattr(item, "text") and item.prov:
            pno = item.prov[0].page_no
            if pno in pages_content:
                pages_content[pno] += item.text + "\n"

    return [pages_content[no].strip() for no in page_numbers]

def llm_detect_toc(text: str, llm) -> bool:
    structured_llm = llm.with_structured_output(TOCDetection)

    prompt = f"""
    You are an expert document analyzer. Determine if the following text is part of a "Table of Contents" (TOC).

    ### GUIDELINES:
    1. POSITIVE SIGNS:
       - Presence of the words "Contents", "Table of Contents", "Sect ion", or "Chapter".
       - A list of topics followed by page numbers (often Roman numerals like 'xi' or integers like '1', '29', '75').
       - Hierarchical numbering like '1.1', '2.3.4'.
       - IMPORTANT: Text might have extra spaces between letters (e.g., 'Co m ple x it y' instead of 'Complexity'). Ignore these spaces.

    2. NEGATIVE SIGNS (Not a TOC):
       - Legal text, copyright info, or "Library of Congress" data (this is a Copyright page).
       - Acknowledgments or Prefaces that don't list other chapters.
       - Random book titles or author names without page mappings.

    ### TEXT TO ANALYZE:
    {text}

    ### FINAL TASK:
    Is this page a Table of Contents? Answer with is_toc=True or False.
    """
    try:
        response = structured_llm.invoke(prompt)
        return response.is_toc
    except Exception as e:
        print(f"Error: {e}")
        return False

def find_toc_pages(pages: List[str], llm) -> List[int]:
    toc_indices = []
    # Allow for up to 1 "non-TOC" page to occur between TOC pages 
    # (e.g. a blank page or an illustration)
    max_gap = 1 
    gap_counter = 0

    for i, page_text in enumerate(pages):
        if not page_text.strip(): continue
        
        is_toc = llm_detect_toc(page_text, llm)
        
        if is_toc:
            toc_indices.append(i)
            gap_counter = 0 # Reset gap if we find another TOC page
        elif toc_indices:
            gap_counter += 1
            if gap_counter > max_gap:
                break # Only stop if we've seen too many non-TOC pages in a row
                
    return toc_indices

def extract_toc_content(pages: List[str], llm, toc_indices: List[int]) -> str:
    if not toc_indices:
        return ""

    toc_raw = "\n".join([pages[idx] for idx in toc_indices])
    
    # Use a schema that expects a content string
    structured_llm = llm.with_structured_output(TOCDetection) 

    prompt = f"""
    Clean and format the following Table of Contents. 
    1. Fix OCR spacing errors (e.g., 'Cla ss' -> 'Class').
    2. Maintain the hierarchy (Chapter -> Section).
    3. Ensure page numbers are preserved.

    Text:
    {toc_raw}
    """
    try:
        response = structured_llm.invoke(prompt)
        # Make sure your TOCDetection schema actually has a 'toc_content' field!
        return response.toc_content 
    except Exception as e:
        print(f"Extraction Error: {e}")
        return ""

def detect_page_index(toc_text: str, llm) -> bool:
    """Checks if the cleaned TOC actually contains page numbers."""
    if not toc_text:
        return False
        
    structured_llm = llm.with_structured_output(PageIndexDetection)
    prompt = f"Does this Table of Contents contain page numbers?\n\n{toc_text}"
    
    try:
        response = structured_llm.invoke(prompt)
        return response.page_index_given_in_toc
    except Exception:
        return False