"""
main.py — RAG CLI using:
- LangChain 1.0+
- ChromaDB
- HuggingFaceEmbeddings
- Ollama (mistral-7b)
"""

import os
from pathlib import Path
from typing import List


from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# -------- CONFIG --------
DATA_PATH = Path("speech.txt")
CHROMA_PERSIST_DIR = "chroma_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
# ------------------------

def load_documents(path: Path):
    """Load the speech.txt file (Windows-safe, no pwd dependency)."""
    if not path.exists():
        raise FileNotFoundError(f"❌ speech.txt not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    from langchain.schema import Document
    return [Document(page_content=text)]


def split_documents(docs):
    """Split into chunks."""
    splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator="\n\n",
    )
    return splitter.split_documents(docs)

def create_or_load_vectorstore(chunks, embeddings):
    """Build or load Chroma vectorstore."""
    persist_dir = CHROMA_PERSIST_DIR

    if os.path.isdir(persist_dir) and any(Path(persist_dir).iterdir()):
        print("🔁 Using existing Chroma store...")
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

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
    """Build RetrievalQA chain."""
    try:
        llm = Ollama(model="mistral", verbose=False)
    except Exception as e:
        raise RuntimeError("❌ Ollama not running or model not pulled.") from e

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": TOP_K}),
        chain_type="stuff",
        return_source_documents=False,
    )

def main():
    print("📚 RAG CLI — Chroma + Embeddings + Ollama")
    
    docs = load_documents(DATA_PATH)
    chunks = split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = create_or_load_vectorstore(chunks, embeddings)
    qa = build_chain(vectordb)

    print("\n✅ Ready! Ask questions (type 'exit' to quit).")

    while True:
        q = input("\nQuestion> ").strip()
        if q.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break
        
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
            ans = qa.invoke({"query": prompt})["result"]
        except Exception as e:
            ans = f"❌ Error: {e}"


        print("\nAnswer:\n", ans)

if __name__ == "__main__":
    main()
