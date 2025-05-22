import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

def extract_text_from_pdf(path: str) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(path)
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)

def ingest_pdfs(pdf_dir: str):
    """Ingest PDFs from a directory and return a list of documents."""
    pdf_files = [
        os.path.join(pdf_dir, fname)
        for fname in os.listdir(pdf_dir)
        if fname.lower().endswith(".pdf")
    ]

    docs = []
    for pdf_path in pdf_files:
        print(f"Processing {os.path.basename(pdf_path)}...")
        raw_text = extract_text_from_pdf(pdf_path)
        docs.append(Document(page_content=raw_text, metadata={"source": os.path.basename(pdf_path)}))

    return docs

def create_faiss_index(docs):
    """Create a FAISS index from the documents."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    texts = splitter.split_documents(docs)
    print(f"Total chunks created: {len(texts)}")

    # Update the embedding model to use sentence-transformers/LaBSE
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")
    vectordb = FAISS.from_documents(texts, embeddings)
    return vectordb

def main():
    load_dotenv()  # Load environment variables

    # Step 1: Ingest PDFs
    pdf_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    if not os.path.exists(pdf_dir):
        raise FileNotFoundError(f"The data directory does not exist: {pdf_dir}")
    docs = ingest_pdfs(pdf_dir)

    # Step 2: Create FAISS index
    print("Creating FAISS index...")
    vectordb = create_faiss_index(docs)

    # Step 3: Test retrieval
    query = "¿Cómo preparar al paciente antes de la cirugía?"
    print(f"Query: {query}")
    results = vectordb.similarity_search(query, k=3)

    print("\n=== Retrieved Chunks ===")
    for i, result in enumerate(results, 1):
        print(f"\n--- Chunk {i} (source: {result.metadata['source']}) ---")
        print(result.page_content[:500].strip(), "...")

def get_retriever():
    """
    Create and return a retriever from the FAISS index.
    """
    load_dotenv()  # Load environment variables

    # Step 1: Ingest PDFs
    pdf_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    if not os.path.exists(pdf_dir):
        raise FileNotFoundError(f"The data directory does not exist: {pdf_dir}")
    docs = ingest_pdfs(pdf_dir)

    # Step 2: Create FAISS index
    print("Creating FAISS index...")
    vectordb = create_faiss_index(docs)

    # Persist the FAISS index for reuse
    vectordb.save_local("faiss_index")

    # Return the retriever
    return vectordb.as_retriever()

# Expose the retriever
retriever = get_retriever()

if __name__ == "__main__":
    main()
