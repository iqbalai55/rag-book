import logging
from typing import List, Dict
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
import os

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document loading, chunking, and metadata extraction."""
    
    def __init__(self, chunker, tokenizer, max_tokens=256):
        """Initialize the document processor."""
        self.max_tokens = max_tokens
        self.chunker = chunker if chunker is not None else HybridChunker(tokenizer=tokenizer)
        logger.info(f"Initialized DocumentProcessor with tokenizer: {tokenizer}")

    def load_document(self, file_path: str):
        """Load and convert a document using Docling."""
        try:
            logger.info(f"Loading document from: {file_path}")
            converter = DocumentConverter()
            result = converter.convert(source=file_path)
            logger.info("Document loaded successfully")
            return result.document
        except Exception as e:
            logger.error(f"Error loading document: {e}")
            raise

    def chunk_document(self, document, file_path: str) -> List[Dict]:
        """
        Chunk a single document into pieces with metadata.
        """
        try:
            logger.info(f"Chunking document with max_tokens: {self.max_tokens}")
            
            chunks = self.chunker.chunk(dl_doc=document, max_tokens=self.max_tokens)
            chunked_with_metadata = []
            filename = os.path.basename(file_path)

            for chunk in chunks:
                chunk_metadata = chunk.model_dump(exclude="text", exclude_none=True)
                pages = set()

                meta = chunk_metadata.get("meta", {})
                doc_items = meta.get("doc_items", [])

                for item in doc_items:
                    for prov in item.get("prov", []):
                        page_no = prov.get("page_no")
                        if page_no is not None:
                            pages.add(page_no)

                metadata = {
                    "pages": sorted(pages),
                    "source": filename
                }
                chunked_with_metadata.append({
                    "text": chunk.text,
                    "metadata": metadata
                })

            logger.info(f"Document chunked into {len(chunked_with_metadata)} pieces")
            return chunked_with_metadata

        except Exception as e:
            logger.error(f"Error chunking document: {e}")
            raise

    def process_document(self, file_path: str) -> List[Dict]:
        """
        Process a single document and return chunks.
        """
        document = self.load_document(file_path)
        return self.chunk_document(document, file_path)

    def process_documents(self, file_paths: List[str]) -> List[Dict]:
        """
        Process a list of documents, returning a flat list of chunks.
        
        Args:
            file_paths: List of document paths
        
        Returns:
            List[Dict]: All chunks from all documents
        """
        all_chunks = []
        for path in file_paths:
            logger.info(f"Processing document: {path}")
            chunks = self.process_document(path)
            all_chunks.extend(chunks)
        logger.info(f"Processed {len(file_paths)} documents into {len(all_chunks)} total chunks")
        return all_chunks
