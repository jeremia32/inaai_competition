from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class MedicalTextChunker:
    """
    Chunking pipeline untuk Medical RAG.

    Features:
    - Recursive chunking
    - Metadata preservation
    - Overlap support
    - Medical-document friendly
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ):
        """
        Parameters:
        ----------
        chunk_size:
            ukuran maksimal chunk

        chunk_overlap:
            overlap antar chunk
            agar context tidak terpotong
        """

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "; ",
                ", ",
                " ",
                "",
            ],
        )

    # =========================================================
    # CHUNK SINGLE DOCUMENT
    # =========================================================
    def chunk_document(
        self,
        document: Document,
    ) -> List[Document]:
        """
        Chunk satu document.
        """

        split_texts = self.text_splitter.split_text(
            document.page_content
        )

        chunks = []

        for chunk_id, chunk_text in enumerate(split_texts):

            metadata = dict(document.metadata)

            metadata.update(
                {
                    "chunk_id": chunk_id,
                    "chunk_size": len(chunk_text),
                }
            )

            chunk_doc = Document(
                page_content=chunk_text,
                metadata=metadata,
            )

            chunks.append(chunk_doc)

        return chunks

    # =========================================================
    # CHUNK MULTIPLE DOCUMENTS
    # =========================================================
    def chunk_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """
        Chunk multiple documents.
        """

        all_chunks = []

        for document in documents:

            chunks = self.chunk_document(document)

            all_chunks.extend(chunks)

        print(
            f"[INFO] total chunks created: "
            f"{len(all_chunks)}"
        )

        return all_chunks

    # =========================================================
    # PREVIEW CHUNKS
    # =========================================================
    def preview_chunks(
        self,
        chunks: List[Document],
        num_chunks: int = 3,
    ):
        """
        Preview hasil chunking.
        """

        print("\n========== CHUNK PREVIEW ==========\n")

        for idx, chunk in enumerate(chunks[:num_chunks]):

            print(f"Chunk #{idx+1}")

            print("Metadata:")
            print(chunk.metadata)

            print("\nContent:")
            print(chunk.page_content[:500])

            print("\n" + "=" * 50 + "\n")