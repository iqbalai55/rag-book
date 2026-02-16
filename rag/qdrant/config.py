import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for the Qdrant RAG chatbot."""
    
    
    CHUNK_TOKENIZER = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Chunking parameters
    MAX_TOKENS = 256
    
