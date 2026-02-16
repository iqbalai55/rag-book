import os
import pandas as pd
import pdfplumber
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

def clean_text(text: str) -> str:
    """Remove empty lines and duplicate headers/footers."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    seen = set()
    clean_lines = []
    for line in lines:
        if line not in seen:
            clean_lines.append(line)
            seen.add(line)
    return "\n".join(clean_lines)



def extract_pdf_to_documents(pdf_path, included_pages_intervals=None):
    """
    Extract text + tables from PDF and convert to LangChain Documents.
    Returns list of Document objects.
    """
    pdf_reader = pdfplumber.open(pdf_path)
    book_name = os.path.basename(pdf_path)

    # Decide which pages to include (1-based)
    if included_pages_intervals is None:
        included_pages = list(range(1, len(pdf_reader.pages) + 1))
    else:
        included_pages = []
        for interval in included_pages_intervals:
            included_pages += list(range(interval[0], interval[1] + 1))

    def include_page(page_number):
        return (page_number + 1) in included_pages

    def include_text(obj):
        return 'size' in obj and obj['size'] >= 10

    def extract_single_page(page):
        # Extract body text
        f_page = page.filter(include_text)
        text = f_page.extract_text() or ""

        # Extract tables
        tables = page.find_tables()
        table_text = ""
        for table in tables:
            table_df = pd.DataFrame.from_records(table.extract())
            if not table_df.empty:
                if not ((table_df == '').values.sum() + table_df.isnull().values.sum() ==
                        table_df.shape[0] * table_df.shape[1]):
                    table_text += "\n\n" + table_df.to_html(header=False, index=False)

        return text + "\n\n" + table_text

    documents = []
    seen_texts = set()

    for page_number, page in tqdm(enumerate(pdf_reader.pages),
                                  total=len(pdf_reader.pages),
                                  desc=f"Extracting {book_name}"):
        if include_page(page_number):
            content = clean_text(extract_single_page(page))
            if content and content not in seen_texts:
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": book_name,
                        "page": page_number + 1
                    }
                )
                documents.append(doc)
                seen_texts.add(content)

    pdf_reader.close()
    return documents


def create_or_merge_faiss(documents, save_path="faiss_index", embedding_model='sentence-transformers/all-MiniLM-L6-v2', device='cpu'):
    """
    Create new FAISS or merge with existing one.
    """
    print("Loading embedding model...")
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": device}
    )

    # Check if FAISS DB exists
    if os.path.exists(save_path):
        print(f"Loading existing FAISS from {save_path}...")
        vectordb = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
        print(f"Adding {len(documents)} new documents...")
        vectordb.add_documents(documents)
    else:
        print(f"Creating new FAISS with {len(documents)} documents...")
        vectordb = FAISS.from_documents(documents, embeddings)

    print(f"Saving FAISS to {save_path} ...")
    vectordb.save_local(save_path)
    print("Done.")


if __name__ == "__main__":
    pdf_paths = [
        r"book\Lean Software Development An Agile Toolkit (Mary Poppendieck  Tom Poppendieck) (Z-Library).pdf",
        r"book\Refactoring for Software Design Smells Managing Technical Debt (Girish Suryanarayana) (Z-Library).pdf",  # add more books here
        r"book\EffectivePython.pdf",
        r"book\Sebastian Buczyński - Implementing the Clean Architecture (2020, Sebastian Buczyński) - libgen.li.pdf",
        r"book\Shape Up Stop Running in Circles and Ship Work that Matters (Ryan Singer) (Z-Library).pdf"
    ]

    all_documents = []
    for pdf_path in pdf_paths:
        docs = extract_pdf_to_documents(pdf_path)
        all_documents.extend(docs)

    print(f"Total documents extracted: {len(all_documents)}")

    create_or_merge_faiss(all_documents, save_path="faiss_index_hp")
