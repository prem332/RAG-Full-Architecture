import os
from dotenv import load_dotenv

load_dotenv()

os.environ["ANONYMIZED_TELEMETRY"] = "False"

GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
PDF_PATH       = "data/Indian_Rental_Agreement_v2.pdf"
CHROMA_DIR     = "chroma_store"
COLLECTION     = "rag_collection"
TOP_K_RETRIEVE = 10
TOP_K_RERANK   = 4

# ── PDF Loading ───────────────────────────────────────────────────────────────
from pypdf import PdfReader

def load_pdf(path: str) -> str:
    reader   = PdfReader(path)
    full_text = ""
    for i, page in enumerate(reader.pages):
        text      = page.extract_text() or ""
        full_text += f"\n\n[PAGE {i+1}]\n{text}"
    return full_text

# ── Chunking ──────────────────────────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(full_text: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(full_text)

# ── Embeddings ────────────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer, CrossEncoder

bi_encoder    = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── ChromaDB ──────────────────────────────────────────────────────────────────
import shutil
import chromadb

def build_vectorstore(chunks: list) -> chromadb.Collection:
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    chunk_ids  = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas  = [{"chunk_index": i, "source": PDF_PATH} for i in range(len(chunks))]
    embeddings = bi_encoder.encode(chunks, batch_size=32, convert_to_numpy=True)

    BATCH = 100
    for start in range(0, len(chunks), BATCH):
        end = min(start + BATCH, len(chunks))
        collection.add(
            ids=chunk_ids[start:end],
            documents=chunks[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=metadatas[start:end],
        )

    return collection

def load_vectorstore() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(name=COLLECTION)

# ── Prompt ────────────────────────────────────────────────────────────────────
def build_prompt(query: str, context_chunks: list) -> str:
    context = "\n\n---\n\n".join(
        [f"[Context {i+1}]:\n{chunk}" for i, chunk in enumerate(context_chunks)]
    )
    return f"""You are a precise document assistant. Answer the user's question using ONLY the context provided below.
If the answer is not found in the context, say: "I could not find relevant information in the document."
Do not use any outside knowledge.

=== CONTEXT ===
{context}

=== QUESTION ===
{query}

=== ANSWER ==="""

# ── LLM ──────────────────────────────────────────────────────────────────────
from groq import Groq

groq_client = Groq(api_key=GROQ_API_KEY)

def generate_answer(prompt: str) -> str:
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful document assistant."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.1,
        max_tokens=512,
    )
    return response.choices[0].message.content

# ── LangGraph Pipeline ────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END
from typing import TypedDict

class RAGState(TypedDict):
    query               : str
    query_embedding     : list
    retrieved_docs      : list
    retrieved_distances : list
    reranked_docs       : list
    ce_scores           : list
    prompt              : str
    answer              : str

def retrieve_node(state: RAGState) -> RAGState:
    collection  = load_vectorstore()
    q_embedding = bi_encoder.encode(state["query"]).tolist()
    results     = collection.query(
        query_embeddings=[q_embedding],
        n_results=TOP_K_RETRIEVE,
        include=["documents", "distances"],
    )
    return {
        **state,
        "query_embedding"    : q_embedding,
        "retrieved_docs"     : results["documents"][0],
        "retrieved_distances": results["distances"][0],
    }

def rerank_node(state: RAGState) -> RAGState:
    pairs  = [(state["query"], doc) for doc in state["retrieved_docs"]]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return {
        **state,
        "reranked_docs": [state["retrieved_docs"][i] for i in ranked[:TOP_K_RERANK]],
        "ce_scores"    : [float(scores[i])           for i in ranked[:TOP_K_RERANK]],
    }

def generate_node(state: RAGState) -> RAGState:
    prompt = build_prompt(state["query"], state["reranked_docs"])
    answer = generate_answer(prompt)
    return {**state, "prompt": prompt, "answer": answer}

def build_rag_app():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank",   rerank_node)
    graph.add_node("generate", generate_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank",   "generate")
    graph.add_edge("generate", END)
    return graph.compile()

def rag_query(query: str, rag_app) -> dict:
    return rag_app.invoke({
        "query"              : query,
        "query_embedding"    : [],
        "retrieved_docs"     : [],
        "retrieved_distances": [],
        "reranked_docs"      : [],
        "ce_scores"          : [],
        "prompt"             : "",
        "answer"             : "",
    })

# ── Index + Run ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    full_text  = load_pdf(PDF_PATH)
    chunks     = chunk_text(full_text)
    build_vectorstore(chunks)
    rag_app    = build_rag_app()

    while True:
        query = input("You: ").strip()
        if query.lower() in ("exit", "quit", ""):
            break
        result = rag_query(query, rag_app)
        print(f"\nBot: {result['answer']}\n")