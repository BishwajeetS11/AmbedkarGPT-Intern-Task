🧠 RAG CLI – Speech Q&A (LangChain + ChromaDB + Ollama)

This project implements a simple Retrieval-Augmented Generation (RAG) command-line chatbot.
It loads speech.txt (Dr. B. R. Ambedkar’s speech), splits it into chunks, creates embeddings, stores them in ChromaDB, retrieves relevant chunks for user questions and generates answers using Ollama’s local Mistral 7B model.

No API keys.
No online services.
100% local.

🚀 Features

Loads and processes "speech.txt"

Splits text into manageable chunks

Generates embeddings using

sentence-transformers/all-MiniLM-L6-v2

Stores embeddings in ChromaDB (local vector DB)

Retrieves relevant chunks using semantic search

Uses Ollama (Mistral 7B) as the offline LLM

Answers questions ONLY from the speech

CLI interface (python main.py)


📂 Project Structure:

rag-app/

│── main.py                   # Main RAG CLI app

│── speech.txt                # Provided Ambedkar speech

│── requirements.txt          # Python dependencies

│── README.md                 # Documentation

└── chroma_store/             # Auto-generated vector store (keep this folder)


The chroma_store/ directory is created automatically when running the program for the first time.


🛠️ 1. Installation

Step 1 — Create a fresh Python environment

(Recommended: conda )

Using Conda:
conda create -n rag python=3.10 -y

conda activate rag

Using venv:
python -m venv venv

source venv/bin/activate           # Mac/Linux

venv\Scripts\activate              # Windows

🔧 2. Install Dependencies


Run:
pip install -r requirements.txt

🤖 3. Install Ollama

Ollama runs LLMs locally.

Mac / Linux:
curl -fsSL https://ollama.ai/install.sh | sh

Windows:

Download from:
https://ollama.com/download/windows

📥 4. Pull the Mistral model

Run this:
ollama pull mistral

Verify installation:
ollama list

You should see mistral in the list.

▶️ 5. Run the RAG CLI App

Navigate to your project folder:

cd rag-app
python main.py


You should see:

📚 RAG CLI — Chroma + Embeddings + Ollama

📄 Creating new Chroma vectorstore...

✅ Chroma store created.

✅ Ready! Ask questions (type 'exit' to quit).


Ask questions such as:

What is the real remedy according to the speaker?
Why does the speaker criticize social reform?

The chatbot will answer using ONLY the speech content.

📦 requirements.txt:

This is the recommended clean version:


langchain==0.1.6

langchain-community==0.0.20

langchain-core==0.1.23

langchain-text-splitters==1.0.0

chromadb==1.3.4

sentence-transformers==5.1.2

transformers==4.57.1

ollama==0.6.1


📘 How It Works (Short Explanation)

1. Load text
The speech is loaded using TextLoader.

2. Split into chunks
Using CharacterTextSplitter(chunk_size=500, overlap=50).

3. Generate embeddings
With HuggingFace model all-MiniLM-L6-v2.

4. Store embeddings
ChromaDB persists vectors in the chroma_store directory.

5. Retrieve relevant chunks
Using semantic search (k=3).

6. Generate response
Using Ollama Mistral 7B, guided by a strict prompt that forces answers ONLY from the context.

🧪 Example Questions:

✔ What does the speaker describe as the “real remedy”?

✔ Why can’t caste and belief in the shastras coexist?

✔ Why does social reform fail according to the speaker?

✔ What is the “real enemy”?

✔ Why can people never get rid of caste?

All answers will come directly from the speech.




