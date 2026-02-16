from typing import List, Tuple
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

def load_vector_db(
    faiss_path: str,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu"
) -> FAISS:
    """
    Load FAISS vector database
    """

    embeddings = HuggingFaceInstructEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": device}
    )

    vectordb = FAISS.load_local(
        faiss_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectordb


def retrieve_context(
    vectordb: FAISS,
    query: str,
    k: int = 3
) -> Tuple[str, List[Tuple[str,int]]]:

    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    docs: List[Document] = retriever.invoke(query)

    merged_context = ""
    pages: List[Tuple[str,int]] = []

    seen_texts = set()

    for doc in docs:
        text = "\n".join([line.strip() for line in doc.page_content.splitlines() if line.strip()])
        if text not in seen_texts:
            merged_context += "\n\n" + text
            seen_texts.add(text)

        # store (source, page)
        src = doc.metadata.get("source", "unknown")
        pg = doc.metadata.get("page", 0)
        pages.append((src, pg))

    return merged_context, pages