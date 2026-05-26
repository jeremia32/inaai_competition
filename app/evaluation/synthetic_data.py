import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.reasoning.gemini_client import GeminiMedicalLLM


class SyntheticDialogueGenerator:
    """Generate synthetic patient queries for evaluation and cost simulation."""

    DEFAULT_EXAMPLES = [
        {
            "query": "Saya berusia 45 tahun dan sering merasa pusing, apakah ini tanda hipertensi?",
            "reference_answer": "Hipertensi sering menyebabkan pusing dan sebaiknya cek tekanan darah serta perbaiki gaya hidup.",
            "relevant_sources": [{"file_name": "PNPK_Hipertensi_2025.pdf", "page": 2}],
            "description": "Simulasi gejala hipertensi pasien dewasa.",
        },
        {
            "query": "Nama saya Siti, NIK 3216549870123456. Apakah obat flu aman untuk ibu hamil 7 bulan?",
            "reference_answer": "Ibu hamil sebaiknya berkonsultasi dengan dokter sebelum minum obat flu dan pilih obat yang aman untuk kehamilan.",
            "relevant_sources": [{"file_name": "PNPK_Kehamilan_2024.pdf", "page": 4}],
            "description": "Pertanyaan yang mengandung PII dan konteks kehamilan.",
        },
        {
            "query": "Apa tanda serangan jantung pada wanita dan kapan harus ke UGD?",
            "reference_answer": "Tanda serangan jantung dapat berupa nyeri dada, sesak napas, dan mual. jika muncul gejala mendadak, segera ke UGD.",
            "relevant_sources": [{"file_name": "PNPK_Kardiovaskular_2024.pdf", "page": 1}],
            "description": "Simulasi gejala darurat kardiovaskular.",
        },
        {
            "query": "Saya memiliki riwayat asma dan ingin tahu apakah boleh menggunakan inhaler bronkodilator setiap hari.",
            "reference_answer": "Penggunaan inhaler bronkodilator harian harus sesuai anjuran dokter dan dikombinasikan dengan kontrol asma.",
            "relevant_sources": [{"file_name": "PNPK_Asma_2024.pdf", "page": 3}],
            "description": "Pertanyaan perawatan asma jangka panjang.",
        },
        {
            "query": "Berapa dosis paracetamol untuk anak 8 tahun dengan demam?",
            "reference_answer": "Dosis paracetamol anak bergantung pada berat badan, biasanya 10-15 mg/kg setiap 4-6 jam, dengan batas maksimal 5 dosis per hari.",
            "relevant_sources": [{"file_name": "PNPK_Anak_2024.pdf", "page": 5}],
            "description": "Pertanyaan dosis obat anak untuk demam.",
        },
    ]

    def generate_examples(
        self,
        num_examples: int = 20,
        llm: Optional[GeminiMedicalLLM] = None,
    ) -> List[Dict[str, Any]]:
        if llm is not None:
            prompt = self.build_generation_prompt(num_examples)
            raw_output = llm.generate(prompt)
            parsed = self._parse_json_list(raw_output)
            if parsed:
                return parsed

        return self._fallback_examples(num_examples)

    def build_generation_prompt(self, num_examples: int) -> str:
        return f"""
Buat {num_examples} contoh kueri medis pasien dalam bahasa Indonesia.
Setiap contoh harus disajikan sebagai objek JSON dengan kunci:
- query
- reference_answer
- relevant_sources (daftar objek dengan file_name dan page)
- description

Contoh output:
[
  {{
    "query": "...",
    "reference_answer": "...",
    "relevant_sources": [{{"file_name": "PNPK_Example.pdf", "page": 1}}],
    "description": "..."
  }}
]

Buat variasi tema: hipertensi, diabetes, kehamilan, asma, overdose, dosis obat, maag, dan keluhan umum.
""".strip()

    @staticmethod
    def _parse_json_list(text: str) -> Optional[List[Dict[str, Any]]]:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            return None

        try:
            payload = text[start : end + 1]
            examples = json.loads(payload)
            if isinstance(examples, list):
                return examples
        except json.JSONDecodeError:
            return None

        return None

    def _fallback_examples(self, num_examples: int) -> List[Dict[str, Any]]:
        examples = []
        for i in range(num_examples):
            sample = self.DEFAULT_EXAMPLES[i % len(self.DEFAULT_EXAMPLES)].copy()
            sample["description"] = (
                f"Simulasi percakapan medis pasien #{i + 1}. "
                + sample["description"]
            )
            sample["query"] = sample["query"]
            sample["reference_answer"] = sample["reference_answer"]
            sample["relevant_sources"] = sample["relevant_sources"]
            examples.append(sample)
        random.shuffle(examples)
        return examples

    def save_examples(self, path: Path, examples: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(examples, handle, ensure_ascii=False, indent=2)

    def load_examples(self, path: Path) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_or_generate(
        self,
        path: Path,
        num_examples: int = 20,
        llm: Optional[GeminiMedicalLLM] = None,
    ) -> List[Dict[str, Any]]:
        if path.exists():
            return self.load_examples(path)

        examples = self.generate_examples(num_examples=num_examples, llm=llm)
        self.save_examples(path, examples)
        return examples
