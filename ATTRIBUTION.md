# Attribution

## Libraries
The following Python libraries were used in this project (see requirements.txt):

- Flask
- LangChain
- ChromaDB
- Sentence-Transformers
- PyPDF
- NumPy
- Python-Dotenv
- Pytest
- Jinja2
- Pico.css


## Models
- all-MiniLM-L6-v2: A sentence-transformer model used for generating vector embeddings of document chunks. Accessed via langchain-huggingface.
- GPT 4o mini: The default Large Language Model (LLM) used for generating answers. Accessed through the OpenRouter API.


## Generative AI Usage
Tools Used: Cursor IDE chatbot, Gemini, ChatGPT

AI was used to write the majority/entirety of the code syntax for this project. I wrote many of the method stubs and defined arguments, and made all design and implementation decisions (see below).

Errors were typically logical which I debugged by being incredibly explicit in specifying the design/implementation details.

Design/implementation decisions and details included:
- Directory structure of src
- Web backend/frontend frameworks
    - Flask, Jinja2, Pico.css
    - SQLite for the database
- Document ingestion
    - Used pypdf for extracting text from PDF slides and handouts.
    - Support for text based formats: .txt, .md, pdf
    - Document deduplication: hashes the documents so that if the same document is ingested (even with a different name), it is not chunked. 
    - Pipeline of extracting text, hashing/saving metadata, chunking, embedding/storing
- Chunking
    - Using LangChain’s recursive splitter
    - Fixed length chunk size of 1000 characters
    - Chunk overlap of 200 characters
    - Merging step where chunks of <300 characters are merged into previous neighbor (happens at Document cutoffs).
- Embeddings 
    - Used sentence-transformers/all-MiniLM-L6-v2 model from HuggingFace. Passed to ChromaDB and called internally through the similarity_search_with_score() and add_documents() methods
    - Normalize vectors for the matching step
- Matching
    - Top 5 chunks are retrieved based on cosine similarity with current Problem embedding
    - The search for a Problem is restricted to the Exam scope of which the Problem belongs to
    - Chunk deduplication step after retrieval fetches max(k * 4, k + 5) chunks to account for duplication within matched Chunks (some chunks like Title/header or Announcement slides have the exact same data and be in many different Documents)
- Class Hierarchy and Database Schema
    - Hierarchy of classes (in order): Course, Exam, Problem, Question (Documents/Chunks belong to a Course and possibly many Exams)
