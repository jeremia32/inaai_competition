import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.guardrails.triage import MedicalTriageGuardrails
from app.pii.redact import MedicalPIIRedactor
from app.rag.chunking import MedicalTextChunker
from app.rag.embedding import MedicalEmbeddingModel
from app.rag.loader import MedicalPDFLoader
from app.rag.retrieval import MedicalHybridRetriever
from app.rag.vectordb import MedicalVectorDB
from app.reasoning.cot import MedicalPromptBuilder
from app.reasoning.gemini_client import GeminiMedicalLLM

PDF_FOLDER = os.getenv("PDF_FOLDER", "./data/raw")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "clinical_rag")
TOP_K_DEFAULT = 5

app = FastAPI(
    title="Clinical RAG Inference API",
    description="Inference endpoint for medical retrieval-augmented generation with PII redaction and guardrails.",
)

# Serve a simple static frontend (chat UI) from /ui
# frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
# if os.path.isdir(frontend_dir):
#     app.mount("/ui", StaticFiles(directory=frontend_dir, html=True), name="frontend")
# else:
#     print(f"[WARN] frontend directory not found: {frontend_dir}")

# Serve a simple static frontend (chat UI) from /ui
# Cukup mundur satu langkah (..) dari folder 'app' menuju root proyek
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

if os.path.isdir(frontend_dir):
    app.mount("/ui", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    print(f"[WARN] frontend directory not found: {frontend_dir}")

# Global pipeline resources
pipeline_ready: bool = False
redactor: Optional[MedicalPIIRedactor] = None
guardrails: Optional[MedicalTriageGuardrails] = None
retriever: Optional[MedicalHybridRetriever] = None
prompt_builder: Optional[MedicalPromptBuilder] = None
llm: Optional[GeminiMedicalLLM] = None


class InferenceRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Medical query for the RAG agent.")
    top_k: int = Field(
        TOP_K_DEFAULT,
        ge=1,
        le=10,
        description="Number of sources to retrieve.",
    )


class SourceItem(BaseModel):
    rank: int
    metadata: Dict[str, Any]
    snippet: str


class InferenceResponse(BaseModel):
    query: str
    safe_query: str
    blocked: bool
    safety_message: str
    sources: List[SourceItem]
    answer: str


@app.on_event("startup")
def startup_event() -> None:
    global pipeline_ready
    global redactor
    global guardrails
    global retriever
    global prompt_builder
    global llm

    loader = MedicalPDFLoader()
    documents = loader.load_folder(PDF_FOLDER)

    if not documents:
        raise RuntimeError(
            f"No documents found in PDF_FOLDER={PDF_FOLDER}."
        )

    chunker = MedicalTextChunker(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = chunker.chunk_documents(documents)

    embedding_manager = MedicalEmbeddingModel()
    embedding_model = embedding_manager.get_model()

    vectordb_manager = MedicalVectorDB(
        embedding_model=embedding_model,
        persist_directory=CHROMA_DB_DIR,
        collection_name=COLLECTION_NAME,
    )

    vectordb = vectordb_manager.load_or_build(chunks)
    vectordb_manager.info()

    redactor = MedicalPIIRedactor()
    guardrails = MedicalTriageGuardrails()
    retriever = MedicalHybridRetriever(
        vectordb=vectordb,
        documents=chunks,
        top_k=TOP_K_DEFAULT,
    )
    prompt_builder = MedicalPromptBuilder()
    llm = GeminiMedicalLLM()

    pipeline_ready = True


@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "pipeline_ready": pipeline_ready,
        "pdf_folder": PDF_FOLDER,
        "chroma_db_dir": CHROMA_DB_DIR,
    }


@app.post("/infer", response_model=InferenceResponse)
def infer(request: InferenceRequest) -> InferenceResponse:
    if not pipeline_ready:
        raise HTTPException(
            status_code=503,
            detail="Pipeline is not ready. Check /health for startup status.",
        )

    safe_query = redactor.redact(request.query)
    triage_result = guardrails.triage(safe_query)
    safety_message = guardrails.generate_guardrail_message(
        triage_result
    )

    if triage_result["risk_level"] != "LOW":
        return InferenceResponse(
            query=request.query,
            safe_query=safe_query,
            blocked=True,
            safety_message=safety_message,
            sources=[],
            answer=safety_message,
        )

    results = retriever.hybrid_search(
        query=safe_query,
        k_dense=request.top_k,
        k_sparse=request.top_k,
    )

    sources = []
    for idx, doc in enumerate(results, start=1):
        sources.append(
            SourceItem(
                rank=idx,
                metadata=doc.metadata,
                snippet=doc.page_content[:400].strip(),
            )
        )

    context = retriever.format_context(results)
    safe_context = redactor.redact(context)

    final_prompt = prompt_builder.build_prompt(
        query=safe_query,
        context=safe_context,
    )

    answer = llm.generate(final_prompt)

    return InferenceResponse(
        query=request.query,
        safe_query=safe_query,
        blocked=False,
        safety_message=safety_message,
        sources=sources,
        answer=answer,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
