"""
main.py — RAG CLI using:
- LangChain 1.0+
- ChromaDB
- HuggingFaceEmbeddings
- Ollama (mistral-7b)

This script loads a text file, splits it into chunks,
creates/loads a Chroma vector database, and runs a
Retrieval-Augmented Generation (RAG) QA loop in the terminal.
"""

import os
from pathlib import Path
from typing import List

# LangChain imports
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# -------- CONFIGURATION --------
DATA_PATH = Path("speech.txt")                    # The file to ingest
CHROMA_PERSIST_DIR = "chroma_store"               # Where vector DB will be stored
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # HF embedding model
CHUNK_SIZE = 500                                   # Size of text chunks
CHUNK_OVERLAP = 50                                 # Overlap between chunks
TOP_K = 3                                          # Retrieval depth



def load_documents(path: Path):
    """Load 'speech.txt' from the root directory.

    This function reads the entire file and wraps it into
    a LangChain Document object. It also checks for file existence
    to avoid path-related errors.
    """
    if not path.exists():
        raise FileNotFoundError(f"❌ speech.txt not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    from langchain.schema import Document
    return [Document(page_content=text)]


def split_documents(docs):
    """Split text into chunks for embedding.
    Uses CharacterTextSplitter to break long text into manageable
    pieces for vector storage and retrieval.
    """
    splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator="\n\n",     # Split on paragraph breaks
    )
    return splitter.split_documents(docs)


def create_or_load_vectorstore(chunks, embeddings):
    """Create a new Chroma vectorstore or load an existing one.
    If the persist directory already contains data, reuse it.
    Otherwise, build a new vector database from the text chunks.
    """
    persist_dir = CHROMA_PERSIST_DIR

    # If Chroma directory exists and contains files → load it
    if os.path.isdir(persist_dir) and any(Path(persist_dir).iterdir()):
        print("🔁 Using existing Chroma store...")
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    # Otherwise create a new vector store
    print("📄 Creating new Chroma vectorstore...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectordb.persist()
    print("✅ Chroma store created.")
    return vectordb


def build_chain(vectordb):
    """Construct the RetrievalQA chain using Ollama.
    Connects the retriever (ChromaDB) to the local LLM (Mistral)
    and creates a QA chain using LangChain.
    """
    try:
        # Create an LLM client for Ollama
        llm = Ollama(model="mistral", verbose=False)
    except Exception as e:
        raise RuntimeError("❌ Ollama not running or model not pulled.") from e

    # Build Retrieval-Augmented QA chain
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": TOP_K}),
        chain_type="stuff",
        return_source_documents=False,
    )


def main():
    """Main CLI loop.
    Loads data → builds embeddings → initializes QA system →
    enters an interactive question-answering loop.
    """

    print("📚 RAG CLI — Chroma + Embeddings + Ollama")

    # Step 1: Load and split text
    docs = load_documents(DATA_PATH)
    chunks = split_documents(docs)

    # Step 2: Initialize embedding model
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Step 3: Load or create vector DB
    vectordb = create_or_load_vectorstore(chunks, embeddings)

    # Step 4: Build RetrievalQA chain
    qa = build_chain(vectordb)

    print("\n✅ Ready! Ask questions (type 'exit' to quit).")

    # Step 5: Start CLI Q&A loop
    while True:
        q = input("\nQuestion> ").strip()

        # Exit condition
        if q.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        # Strict prompting: answer ONLY based on retrieved context
        prompt = f"""
STRICT INSTRUCTIONS:

- ONLY answer using the retrieved context.
- If the question cannot be answered DIRECTLY from context,
  reply EXACTLY with:
  "I cannot find that information in the provided documents."

DO NOT infer. DO NOT guess. DO NOT elaborate.

QUESTION: {q}
"""

        try:
            # Run query through the QA chain
            ans = qa.invoke({"query": prompt})["result"]
        except Exception as e:
            ans = f"❌ Error: {e}"

        print("\nAnswer:\n", ans)


if __name__ == "__main__":
    main()
