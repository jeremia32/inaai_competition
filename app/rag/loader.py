from pathlib import Path
from typing import List, Optional
import re

from pypdf import PdfReader
from langchain_core.documents import Document


class MedicalPDFLoader:
    """
    Loader untuk dokumen medis PNPK.
    
    Features:
    - Load single PDF
    - Load multiple PDFs
    - Metadata extraction
    - Year extraction
    - Basic text cleaning
    """

    def __init__(self):
        pass

    # =========================================================
    # CLEAN TEXT
    # =========================================================
    def clean_text(self, text: str) -> str:
        """
        Membersihkan text hasil ekstraksi PDF.
        """

        # hapus multiple spaces
        text = re.sub(r"\s+", " ", text)

        # hapus karakter aneh
        text = text.replace("\x00", "")

        return text.strip()

    # =========================================================
    # EXTRACT YEAR
    # =========================================================
    def extract_year(self, filename: str) -> Optional[int]:
        """
        Ambil tahun dari nama file.

        Contoh:
        PNPK_Diabetes_2025.pdf -> 2025
        """

        match = re.search(r"(20\d{2})", filename)

        if match:
            return int(match.group(1))

        return None

    # =========================================================
    # LOAD SINGLE PDF
    # =========================================================
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load satu file PDF.
        """

        pdf_file = Path(pdf_path)

        if not pdf_file.exists():
            raise FileNotFoundError(
                f"PDF tidak ditemukan: {pdf_path}"
            )

        reader = PdfReader(str(pdf_file))

        documents = []

        # extract metadata dasar
        file_name = pdf_file.name
        source_name = pdf_file.stem
        year = self.extract_year(file_name)

        for page_number, page in enumerate(reader.pages, start=1):

            try:
                text = page.extract_text()

                if not text:
                    continue

                text = self.clean_text(text)

                document = Document(
                    page_content=text,
                    metadata={
                        "source": source_name,
                        "file_name": file_name,
                        "page": page_number,
                        "year": year,
                        "document_type": "PNPK",
                    },
                )

                documents.append(document)

            except Exception as e:
                print(
                    f"[ERROR] gagal membaca halaman "
                    f"{page_number} dari {file_name}: {e}"
                )

        print(
            f"[INFO] Loaded {len(documents)} pages "
            f"from {file_name}"
        )

        return documents

    # =========================================================
    # LOAD FOLDER
    # =========================================================
    def load_folder(self, folder_path: str) -> List[Document]:
        """
        Load semua PDF dalam folder.
        """

        folder = Path(folder_path)

        if not folder.exists():
            raise FileNotFoundError(
                f"Folder tidak ditemukan: {folder_path}"
            )

        all_documents = []

        pdf_files = sorted(folder.glob("*.pdf"))

        if not pdf_files:
            raise ValueError(
                "Tidak ada file PDF ditemukan."
            )

        print(f"[INFO] ditemukan {len(pdf_files)} file PDF")

        for pdf_file in pdf_files:

            docs = self.load_pdf(str(pdf_file))

            all_documents.extend(docs)

        print(
            f"[INFO] total loaded documents: "
            f"{len(all_documents)}"
        )

        return all_documents