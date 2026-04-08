# 📄 RAG Chatbot Pipeline

A Retrieval-Augmented Generation (RAG) chatbot built from scratch with **bi-encoder retrieval**, **cross-encoder reranking**, and **Groq LLM** — designed for deep understanding of RAG architecture.

---

## 🏗️ Architecture

```
INDEXING
PDF → Pages → Chunks → Bi-Encoder → ChromaDB

QUERYING
Query → Bi-Encoder → ChromaDB (top-10)
                          ↓
                 Cross-Encoder Rerank (top-4)
                          ↓
                    Groq LLM → Answer
```

---

## 🛠️ Tech Stack

| Layer          | Tool                                  |
|----------------|---------------------------------------|
| PDF Parsing    | `pypdf`                               |
| Chunking       | `RecursiveCharacterTextSplitter`      |
| Bi-Encoder     | `all-MiniLM-L6-v2` (384 dims)         |
| Vector DB      | `ChromaDB` (local persistent)         |
| Cross-Encoder  | `ms-marco-MiniLM-L-6-v2`             |
| Pipeline       | `LangGraph` StateGraph                |
| LLM            | `Groq` → `llama-3.1-8b-instant`      |
| UI             | `Streamlit`                           |

---

## 📁 Project Structure

```
RAG-Chatbot-Pipeline/
├── data/
│   └── document.pdf          ← your PDF goes here
├── chroma_store/             ← auto-created by ChromaDB
├── venv/                     ← virtual environment
├── .env                      ← API keys (never commit)
├── .gitignore
├── requirements.txt
├── rag_pipeline.py           ← core RAG logic
├── rag_pipeline.ipynb        ← cell by cell learning notebook
└── main.py                   ← Streamlit UI
```

---

## ⚙️ Setup (GitHub Codespaces)

**1. Clone the repo**
```bash
git clone https://github.com/your-username/RAG-Chatbot-Pipeline.git
cd RAG-Chatbot-Pipeline
```

**2. Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Register Jupyter kernel**
```bash
python -m ipykernel install --user --name=rag-core --display-name "Python (rag-core)"
```

**5. Create `.env` file**
```bash
echo "GROQ_API_KEY=your_key_here" > .env
```
> Get your free Groq API key at [console.groq.com](https://console.groq.com)

**6. Add your PDF**
```bash
cp your_document.pdf data/document.pdf
```

---

## 🚀 Usage

### Notebook (Learning)
Open `rag_pipeline.ipynb` in VS Code, select **Python (rag-core)** kernel, and run cells top to bottom.

> The notebook builds the ChromaDB index — run it before the Streamlit app.

### Streamlit App
```bash
streamlit run main.py
```

### Terminal (CLI)
```bash
python rag_pipeline.py
```

---

## 📊 Retrieval Configuration

| Parameter       | Value | Reason                              |
|-----------------|-------|-------------------------------------|
| `chunk_size`    | 512   | ~128 tokens, good retrieval granularity |
| `chunk_overlap` | 100   | Prevents context loss at boundaries |
| `TOP_K_RETRIEVE`| 10    | Wide net for high recall            |
| `TOP_K_RERANK`  | 4     | Precise context for LLM             |
| `temperature`   | 0.1   | Factual answers, low randomness     |

---

## 🔄 LangGraph Pipeline

```
retrieve_node  →  rerank_node  →  generate_node  →  END
```

| Node            | Responsibility                              |
|-----------------|---------------------------------------------|
| `retrieve_node` | Embed query → ChromaDB top-10 retrieval     |
| `rerank_node`   | Cross-encoder score → sort → keep top-4     |
| `generate_node` | Build prompt → Groq LLM → return answer     |

---

## 💡 Key Concepts

**Why bi-encoder + cross-encoder?**
```
Bi-encoder   → fast, independent encoding  → high recall
Cross-encoder → joint query+doc encoding   → high precision
```

**Why chunk overlap?**
```
Prevents important context from being cut at chunk boundaries
chunk_overlap=100 means up to 100 chars repeated between chunks
```

**Why cosine similarity?**
```
Ignores vector magnitude, only measures directional similarity
Better than L2 distance for text embeddings
```

---

## 🔑 Environment Variables

| Variable       | Description         |
|----------------|---------------------|
| `GROQ_API_KEY` | Groq API key        |

---