"""
Logic for splitting documents into chunks.
"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .types import Document, Chunk

def chunk_document(doc: Document) -> List[Chunk]:
    """
    Splits a Document into a list of Chunks using a recursive character splitter.

    Args:
        doc: The Document to be chunked.

    Returns:
        A list of Chunks, each linked to the parent Document by document_id.
    """
    
    # Initialize the splitter with our chosen parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    # Split the document text
    split_texts = text_splitter.split_text(doc.extracted_text)
    
    chunks = []
    for i, text in enumerate(split_texts):
        # Create a new Chunk object for each piece of text
        # The embedding field is left as None, to be filled in later.
        chunk = Chunk(
            chunk_id=f"{doc.doc_id}-chunk-{i}",  # Create a unique ID for the chunk
            chunk_text=text,
            doc_id=doc.doc_id,
            chunk_index=i,
            embedding=None  # Embedding is not calculated at this stage
        )
        chunks.append(chunk)
        
    return chunks