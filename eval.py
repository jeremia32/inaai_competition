import json
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from app.evaluation.framework import (
    EvaluationExample,
    MedicalEvaluator,
)
from app.guardrails.triage import MedicalTriageGuardrails
from app.pii.redact import MedicalPIIRedactor
from app.rag.chunking import MedicalTextChunker
from app.rag.embedding import MedicalEmbeddingModel
from app.rag.loader import MedicalPDFLoader
from app.rag.retrieval import MedicalHybridRetriever
from app.rag.vectordb import MedicalVectorDB
from app.reasoning.cot import MedicalPromptBuilder
from app.reasoning.gemini_client import GeminiMedicalLLM

load_dotenv()

PDF_FOLDER = os.getenv("PDF_FOLDER", "./data/raw")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "clinical_rag")
EVAL_EMBEDDING_MODELS = os.getenv(
    "EVAL_EMBEDDING_MODELS",
    "intfloat/multilingual-e5-small,paraphrase-multilingual-mpnet-base-v2",
).split(",")
EVAL_DATASET_PATH = Path("./data/eval_dataset.json")
TOP_K = 5


def load_eval_dataset(path: Path) -> List[EvaluationExample]:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            raw_data = json.load(handle)
    else:
        raw_data = [
            {
                "query": "Apa terapi awal hipertensi?",
                "reference_answer": "Terapi awal hipertensi meliputi modifikasi gaya hidup dan obat antihipertensi sesuai pedoman.",
                "relevant_sources": [
                    {
                        "file_name": "sample.pdf",
                        "page": 1,
                    }
                ],
                "description": "Contoh kueri hipertensi.",
            },
            {
                "query": "Bagaimana menangani pasien dengan risiko overdosis obat?",
                "reference_answer": "Untuk pasien berisiko overdosis, segera rujuk ke layanan gawat darurat dan hentikan obat yang dicurigai.",
                "relevant_sources": [
                    {
                        "file_name": "sample.pdf",
                        "page": 2,
                    }
                ],
                "description": "Contoh kueri berisiko tinggi.",
            },
        ]

    return [
        EvaluationExample(
            query=item["query"],
            reference_answer=item["reference_answer"],
            relevant_sources=item.get("relevant_sources", []),
            description=item.get("description"),
        )
        for item in raw_data
    ]


def build_pipeline(model_name: str):
    loader = MedicalPDFLoader()
    documents = loader.load_folder(PDF_FOLDER)

    chunker = MedicalTextChunker(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = chunker.chunk_documents(documents)

    embedding_manager = MedicalEmbeddingModel(
        model_name=model_name,
    )
    embedding_model = embedding_manager.get_model()

    model_slug = model_name.replace("/", "_").replace(".", "_")
    persist_dir = Path(CHROMA_DB_DIR) / model_slug
    persist_dir.mkdir(parents=True, exist_ok=True)

    vectordb_manager = MedicalVectorDB(
        embedding_model=embedding_model,
        persist_directory=str(persist_dir),
        collection_name=COLLECTION_NAME,
    )
    vectordb = vectordb_manager.load_or_build(chunks)
    vectordb_manager.info()

    retriever = MedicalHybridRetriever(
        vectordb=vectordb,
        documents=chunks,
        top_k=TOP_K,
    )

    return (
        MedicalPIIRedactor(),
        MedicalTriageGuardrails(),
        retriever,
        MedicalPromptBuilder(),
    )


def main():
    print("\n========== EVALUATION START ==========")

    dataset = load_eval_dataset(EVAL_DATASET_PATH)
    print(f"[INFO] evaluation dataset loaded: {len(dataset)} examples")

    for model_name in EVAL_EMBEDDING_MODELS:
        model_name = model_name.strip()
        if not model_name:
            continue

        print("\n===========================================")
        print(f"[INFO] Running evaluation with embedding model: {model_name}")
        print("===========================================\n")

        redactor, guardrails, retriever, prompt_builder = build_pipeline(
            model_name=model_name,
        )

        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            llm = GeminiMedicalLLM()
        else:
            llm = None
            print("[WARN] HF_TOKEN not set. Generation will be skipped, but retrieval and PII metrics can still run.")

        evaluator = MedicalEvaluator(
            redactor=redactor,
            guardrails=guardrails,
            retriever=retriever,
            prompt_builder=prompt_builder,
            llm=llm,
            top_k=TOP_K,
        )

        results = []
        for example in dataset:
            print(f"[INFO] evaluating query: {example.query}")
            result = evaluator.evaluate_example(example)
            results.append(result)

        summary = evaluator.summarize_results(results)
        model_slug = model_name.replace("/", "_").replace(".", "_")
        output_path = Path(f"eval_results_{model_slug}.json")
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "model_name": model_name,
                    "summary": summary,
                    "examples": [r.__dict__ for r in results],
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )

        print("\n========== EVALUATION SUMMARY ==========")
        for key, value in summary.items():
            print(f"{key}: {value}")

        print(f"\nResults saved to {output_path.resolve()}")
        print("========== MODEL RUN COMPLETE ==========")


if __name__ == "__main__":
    main()
