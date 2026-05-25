# class MedicalPromptBuilder:
#     """
#     Medical reasoning prompt builder.

#     Features:
#     - Grounded medical QA
#     - Anti hallucination
#     - Citation-aware prompting
#     """

#     def __init__(self):

#         self.system_prompt = """
# Anda adalah AI Assistant medis.

# ATURAN:
# 1. Jawab HANYA berdasarkan context yang diberikan.
# 2. Jangan mengarang informasi medis.
# 3. Jika informasi tidak tersedia:
#    katakan bahwa data tidak ditemukan.
# 4. Jangan memberi diagnosis pasti.
# 5. Jangan memberi keputusan medis final.
# 6. Untuk kondisi darurat:
#    sarankan konsultasi dokter/IGD.
# 7. Sertakan sumber jika tersedia.
# """

#     # =====================================================
#     # BUILD PROMPT
#     # =====================================================

#     def build_prompt(
#         self,
#         query: str,
#         context: str,
#     ) -> str:
#         """
#         Build final grounded prompt.
#         """

#         prompt = f"""
# {self.system_prompt}

# ==============================
# CONTEXT
# ==============================

# {context}

# ==============================
# USER QUESTION
# ==============================

# {query}

# ==============================
# INSTRUCTIONS
# ==============================

# - Jawab berdasarkan context.
# - Gunakan bahasa Indonesia.
# - Buat jawaban ringkas namun jelas.
# - Jangan berhalusinasi.
# - Jika tidak yakin:
#   katakan informasi tidak tersedia.
# - Sertakan sumber jika ada.

# ==============================
# FINAL ANSWER
# ==============================
# """

#         return prompt

#     # =====================================================
#     # DEBUG
#     # =====================================================

#     def debug_prompt(
#         self,
#         query: str,
#         context: str,
#     ):

#         final_prompt = self.build_prompt(
#             query=query,
#             context=context,
#         )

#         print(
#             "\n========== FINAL PROMPT ==========\n"
#         )

#         print(final_prompt[:5000])

class MedicalPromptBuilder:
    """
    Medical reasoning prompt builder (production-grade citation aware)
    """

    def __init__(self):

        self.system_prompt = """
Anda adalah AI Assistant medis berbasis dokumen.

ATURAN UTAMA:
1. Jawab HANYA berdasarkan context yang diberikan.
2. Jangan mengarang informasi medis.
3. Setiap informasi medis WAJIB memiliki citation seperti [1], [2].
4. Jika tidak ada di context, katakan: "tidak ditemukan dalam dokumen".
5. Jangan memberikan diagnosis pasti.
6. Jangan memberikan keputusan medis final.
7. Untuk kondisi darurat, sarankan IGD/dokter segera.
8. Jangan membuat sumber sendiri.

FORMAT WAJIB:
- Gunakan citation angka [1], [2] di dalam kalimat.
- Semua fakta harus punya minimal 1 citation.
"""

    def build_prompt(self, query: str, context: str) -> str:

        prompt = f"""
{self.system_prompt}

==============================
CONTEXT (NUMBERED SOURCES)
==============================

{context}

==============================
USER QUESTION
==============================

{query}

==============================
INSTRUCTION FOR MODEL
==============================

- Gunakan ONLY context di atas.
- WAJIB gunakan citation [1], [2] sesuai context.
- Jangan membuat informasi baru.
- Jawaban harus dalam Bahasa Indonesia.
- Jika context tidak cukup:
  jawab "tidak ditemukan dalam dokumen".

==============================
FINAL ANSWER
==============================
"""

        return prompt

    def debug_prompt(self, query: str, context: str):

        final_prompt = self.build_prompt(query, context)

        print("\n========== FINAL PROMPT ==========\n")
        print(final_prompt[:5000])