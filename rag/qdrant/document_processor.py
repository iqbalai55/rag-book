import logging
from typing import List, Dict
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from rag.qdrant.config import Config
import os

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading, chunking, and metadata extraction."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.chunker = HybridChunker(tokenizer=Config.CHUNK_TOKENIZER)
        logger.info(f"Initialized DocumentProcessor with tokenizer: {Config.CHUNK_TOKENIZER}")
    
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
        Chunk the document into smaller pieces with metadata (page numbers and source).

        Args:
            document: The loaded document object
            file_path: Path of the original file (used for 'source' metadata)

        Returns:
            List[Dict]: Each dict has 'text' and 'metadata' (with keys 'pages' and 'source')
        """
        try:
            logger.info(f"Chunking document with max_tokens: {Config.MAX_TOKENS}")
            
            chunks = self.chunker.chunk(dl_doc=document, max_tokens=Config.MAX_TOKENS)
            chunked_with_metadata = []
            filename = os.path.basename(file_path)  # extract filename from path

            for chunk in chunks:
                # Extract metadata from the chunk
                chunk_metadata = chunk.model_dump(exclude="text", exclude_none=True)
                
                pages = set()

                meta = chunk_metadata.get("meta", {})
                doc_items = meta.get("doc_items", [])

                for item in doc_items:
                    for prov in item.get("prov", []):
                        page_no = prov.get("page_no")
                        if page_no is not None:
                            pages.add(page_no)
                
                #print(pages)

                metadata = {
                    "pages": sorted(pages),
                    "source": filename
                }
                chunked_with_metadata.append({
                    "text": chunk.text,
                    "metadata": metadata
                })

            logger.info(f"Document chunked into {len(chunked_with_metadata)} pieces with metadata")
            return chunked_with_metadata

        except Exception as e:
            logger.error(f"Error chunking document: {e}")
            raise
        
    def process_document(self, file_path: str) -> List[Dict]:
        """
        Complete document processing pipeline.
        
        Returns:
            List[Dict]: Chunked text with metadata
        """
        document = self.load_document(file_path)
        chunked_with_metadata = self.chunk_document(document, file_path)
        return chunked_with_metadata
