# RAG, Embeddings & Vector Databases — Chat With Your Documents

A beginner-friendly Retrieval-Augmented Generation (RAG) application. Upload PDF, TXT, or DOCX files, create text chunks and embeddings, store them in ChromaDB, retrieve relevant passages, and ask questions using a local Ollama LLM.

## Architecture

`Document → Text Extraction → Chunking → Ollama Embeddings → ChromaDB → Similarity Search → Retrieved Context → Ollama LLM → Answer`

## Tech stack

- Python + Streamlit
- Ollama (`llama3.2` for chat and `nomic-embed-text` for embeddings)
- LangChain
- ChromaDB vector database
- PyPDF and python-docx

## 1. Install prerequisites

Install Python 3.10+ and Ollama from the official Ollama website: https://ollama.com/

Open a terminal and download the models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Make sure Ollama is running. On Windows, opening the Ollama application normally starts its local service.

## 2. Set up the project

Open this project folder in VS Code. In the terminal:

```bash
python -m venv venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows Command Prompt**
```cmd
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create your environment file:

**Windows PowerShell**
```powershell
Copy-Item .env.example .env
```

**macOS/Linux**
```bash
cp .env.example .env
```

No OpenAI API key is required. The app uses Ollama locally.

## 3. Run the app

```bash
streamlit run app.py
```

Your browser should open the Streamlit app. Upload documents in **Upload & index**, click **Process documents**, then go to **Ask questions**.

## 4. Project structure

```text
rag_embeddings_vector_db_project/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── data/
```

ChromaDB creates the `chroma_db/` folder automatically after indexing documents. It is ignored by Git because it contains your local vector database.

## 5. Important notes

- Supported file types: PDF, TXT, DOCX.
- Scanned/image-only PDFs may not contain extractable text; OCR is not included in this starter.
- The first embedding or chat request may take longer while the local model loads.
- Keep Ollama running while using the app.
- For larger projects, add page-level PDF metadata, OCR, authentication, and more advanced retrieval/reranking.

## Troubleshooting

**Connection refused / cannot connect to Ollama**
- Open the Ollama desktop app.
- Check that `http://localhost:11434` is reachable.
- Run `ollama list` to confirm the models are installed.

**Model not found**
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

**PowerShell does not allow activation**
You can use Command Prompt activation (`venv\\Scripts\\activate`) or adjust the PowerShell execution policy for your user account.

## Learning objectives covered

1. Understand why LLMs hallucinate and how RAG grounds answers in retrieved information.
2. Learn embeddings, semantic similarity, document processing, chunking, and vector databases.
3. Connect retrieved context to an LLM to build document-based Q&A.
4. Build a complete “Chat With Your Documents” workflow.
