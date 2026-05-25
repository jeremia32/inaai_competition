from app.rag.loader import MedicalPDFLoader
from app.rag.chunking import MedicalTextChunker
from app.rag.embedding import MedicalEmbeddingModel
from app.rag.vectordb import MedicalVectorDB
from app.rag.retrieval import MedicalHybridRetriever
from app.rag.reranker import MedicalReranker
from app.rag.recency import MedicalRecencyWeighter

from app.pii.redact import MedicalPIIRedactor

from app.guardrails.triage import (
    MedicalTriageGuardrails
)

from app.reasoning.cot import (
    MedicalPromptBuilder
)

from app.reasoning.gemini_client import (
    GeminiMedicalLLM
)

# =====================================================
# CONFIG
# =====================================================

PDF_FOLDER = "./data/raw"

QUERY = "Apa itu leukemia?"

CHROMA_DB_DIR = "./chroma_db"

COLLECTION_NAME = "clinical_rag"


def main():

    print("\n==============================")
    print("CLINICAL-RAG AGENT STARTED")
    print("==============================\n")

    # =================================================
    # STEP 1 — LOAD DOCUMENTS
    # =================================================

    print("[STEP 1] Loading PDF documents...\n")

    loader = MedicalPDFLoader()

    documents = loader.load_folder(
        PDF_FOLDER
    )

    print(
        f"\nTotal loaded pages: "
        f"{len(documents)}"
    )

    # =================================================
    # STEP 2 — CHUNKING
    # =================================================

    print("\n[STEP 2] Chunking documents...\n")

    chunker = MedicalTextChunker(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = chunker.chunk_documents(
        documents
    )

    print(
        f"\nTotal chunks: {len(chunks)}"
    )

    # optional preview
    chunker.preview_chunks(
        chunks,
        num_chunks=2,
    )

    # =================================================
    # STEP 3 — EMBEDDING MODEL
    # =================================================

    print(
        "\n[STEP 3] Loading embedding model...\n"
    )

    embedding_manager = (
        MedicalEmbeddingModel()
    )

    embedding_model = (
        embedding_manager.get_model()
    )

    # optional test
    embedding_manager.test_embedding()

    # =================================================
    # STEP 4 — LOAD CHROMADB
    # =================================================

    print(
        "\n[STEP 4] Loading ChromaDB...\n"
    )

    vectordb_manager = MedicalVectorDB(
        embedding_model=embedding_model,
        persist_directory=CHROMA_DB_DIR,
        collection_name=COLLECTION_NAME,
    )

    # =============================================
    # LOAD OR BUILD DATABASE
    # =============================================
    # If the Chroma DB exists and can be loaded, use it.
    # Otherwise build it automatically from the chunked documents.
    # =============================================

    vectordb = vectordb_manager.load_or_build(
        chunks
    )

    vectordb_manager.info()

    # =================================================
    # STEP 5 — QUERY PII REDACTION
    # =================================================

    print(
        "\n[STEP 5] Query PII Redaction...\n"
    )

    redactor = MedicalPIIRedactor()

    safe_query = redactor.redact(
        QUERY
    )

    print("Original Query:")
    print(QUERY)

    print("\nSafe Query:")
    print(safe_query)

    # =================================================
    # STEP 6 — MEDICAL GUARDRAILS
    # =================================================

    print(
        "\n[STEP 6] Medical Guardrails...\n"
    )

    guardrails = (
        MedicalTriageGuardrails()
    )

    triage_result = (
        guardrails.triage(
            safe_query
        )
    )

    print(triage_result)

    safety_message = (
        guardrails.generate_guardrail_message(
            triage_result
        )
    )

    print("\nSafety Message:\n")

    print(safety_message)

    if triage_result["risk_level"] != "LOW":
        print(
            "\n[PIPELINE STOPPED] Guardrail blocked retrieval and generation."
        )
        return

    # =================================================
    # STEP 7 — HYBRID RETRIEVAL
    # =================================================

    print(
        "\n[STEP 7] Hybrid Retrieval...\n"
    )

    retriever = MedicalHybridRetriever(
        vectordb=vectordb,
        documents=chunks,
        top_k=5,
    )

    results = retriever.hybrid_search(
        query=safe_query
    )

    retriever.debug_results(results)

    # =================================================
    # STEP 8 — RERANKER
    # =================================================

    print(
        "\n[STEP 8] Reranking Results...\n"
    )

    reranker = MedicalReranker()

    reranked_results = reranker.rerank(
        query=safe_query,
        results=results,
        top_k=3,
    )

    reranker.debug_rerank(
        reranked_results
    )

    # =================================================
    # STEP 9 — RECENCY WEIGHTING
    # =================================================

    print(
        "\n[STEP 9] Recency Weighting...\n"
    )

    recency_weighter = (
        MedicalRecencyWeighter()
    )

    weighted_results = (
        recency_weighter.apply_weighting(
            reranked_results
        )
    )

    recency_weighter.debug_results(
        weighted_results
    )

    # =================================================
    # STEP 10 — FORMAT CONTEXT
    # =================================================

    print(
        "\n[STEP 10] Final Context...\n"
    )

    final_docs = [

        item["document"]

        for item in weighted_results
    ]

    context = retriever.format_context(
        final_docs
    )

    print(context[:3000])

    # =================================================
    # STEP 11 — CONTEXT PII REDACTION
    # =================================================

    print(
        "\n[STEP 11] Context PII Redaction...\n"
    )

    safe_context = redactor.redact(
        context
    )

    print(safe_context[:3000])

    # =================================================
    # STEP 12 — BUILD MEDICAL PROMPT
    # =================================================

    print(
        "\n[STEP 12] Building Medical Prompt...\n"
    )

    prompt_builder = (
        MedicalPromptBuilder()
    )

    final_prompt = (
        prompt_builder.build_prompt(
            query=safe_query,
            context=safe_context,
        )
    )

    print(final_prompt[:5000])

    # =================================================
    # STEP 13 — GEMINI GENERATION
    # =================================================

    print(
        "\n[STEP 13] Gemini Medical Generation...\n"
    )

    llm = GeminiMedicalLLM()

    response = llm.generate(
        final_prompt
    )

    # =================================================
    # FINAL RESPONSE
    # =================================================

    print(
        "\n========== FINAL RESPONSE ==========\n"
    )

    print(response)

    print("\n==============================")
    print("PIPELINE FINISHED")
    print("==============================\n")


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    main()