# **🏥 Clinical-RAG Agent: Asisten AI Medis**

**Clinical-RAG Agent** adalah sebuah sistem *Retrieval-Augmented Generation* (RAG) cerdas untuk domain medis yang dilengkapi dengan **Multi-step Reasoning**, **Guardrails (Sistem Keamanan)**, dan **Proteksi Data Pribadi (PII)**. Proyek ini dibangun sebagai bagian dari kompetisi INA AI.

Sistem ini dirancang tidak hanya untuk menjawab pertanyaan medis berdasarkan dokumen referensi, tetapi juga memastikan bahwa privasi pengguna terjaga dan pertanyaan yang bersifat darurat/berisiko tinggi ditangani dengan aman.

## **✨ Fitur Utama & Implementasi Inti**

Proyek ini tidak menggunakan RAG konvensional biasa, melainkan menerapkan *pipeline* kompleks yang terdiri dari beberapa tahap:

### **1\. Proteksi PII (Personal Identifiable Information) Redaction**

Sebelum kueri pengguna masuk ke sistem pencarian atau diproses oleh LLM, modul MedicalPIIRedactor akan menyensor data sensitif (seperti NIK, No. HP, Email, Nama).

* **Diimplementasikan pada:** app.pii.redact

### **2\. Medical Triage Guardrails**

Sistem dilengkapi dengan penyaring keamanan (MedicalTriageGuardrails) yang mendeteksi tingkat risiko kueri. Jika terdeteksi pertanyaan medis darurat (risiko tinggi), sistem akan **memblokir** proses LLM dan langsung memberikan pesan rujukan medis darurat.

* **Diimplementasikan pada:** app.guardrails.triage

### **3\. Hybrid Retrieval & Reranking**

Pencarian dokumen menggunakan dua metode sekaligus untuk akurasi maksimal:

* **Sparse Retrieval:** BM25 (Pencarian presisi berbasis kata kunci).  
* **Dense Retrieval:** Vector Embeddings menggunakan model HuggingFace & ChromaDB (Pencarian berbasis makna/konteks).  
* **Reranking:** Menggunakan CrossEncoder (ms-marco-MiniLM-L-6-v2) dan pembobotan kebaruan dokumen (Recency Weighting) untuk mengurutkan hasil terbaik.  
* **Diimplementasikan pada:** app.rag.retrieval, app.rag.reranker, app.rag.recency

### **4\. Estimasi Biaya Inferensi**

Dilengkapi dengan CostTracker untuk melacak jumlah token (prompt & response) serta mengestimasi biaya (dalam USD) dari setiap kueri yang dieksekusi.

## **💻 Contoh Kode Inti (Core Pipeline)**

Alur utama sistem ini berjalan melalui API FastAPI (api.py) atau skrip CLI (main.py). Berikut adalah potongan inti dari proses inferensinya:

\# 1\. Sensor kueri pengguna dari data sensitif (PII)  
safe\_query \= redactor.redact(request.query)

\# 2\. Periksa keamanan pertanyaan (Guardrails)  
triage\_result \= guardrails.triage(safe\_query)  

`if triage\_result\["risk\_level"\] \!= "LOW":  
    return {"blocked": True, "answer": guardrails.generate\_guardrail\_message(triage\_result)}`

\# 3\. Pencarian Hibrida (Hybrid Search)  

`results \= retriever.hybrid\_search(  
    query=safe\_query,  
    k\_dense=request.top\_k,  
    k\_sparse=request.top\_k,  
)`

\# 4\. Reranking & Format Konteks  
context \= retriever.format\_context(results)  
safe\_context \= redactor.redact(context) \# Sensor juga hasil pencariannya

\# 5\. Bangun Prompt Medis & Generate Jawaban (LLM)  
`final\_prompt \= prompt\_builder.build\_prompt(query=safe\_query, context=safe\_context)  
answer \= llm.generate(final\_prompt)
`
## **🚀 Cara Menjalankan Aplikasi**

Aplikasi ini menyediakan dua cara eksekusi:

### **A. Menjalankan via FastAPI Endpoint (Disarankan)**

Server API telah dikonfigurasi untuk menyediakan *endpoint* /infer dan antarmuka UI *chat* statis di /ui.

1. Pastikan semua *dependencies* di requirements.txt telah terinstal.  
2. Jalankan server:  
   `uvicorn app.api:app \--host 0.0.0.0 \--port 8000`

3. Akses antarmuka pengguna di: http://localhost:8000/ui  
4. Cek ui kesehatan pipeline di: http://localhost:8000/ui

### **B. Menjalankan via CLI (Terminal)**

Untuk melihat detail *step-by-step* dari 13 proses RAG di terminal:

`python main.py`

## **📊 Metrik Evaluasi Sistem**

Sistem ini telah dievaluasi menggunakan dataset sintetis *ground-truth* dengan model *embedding* intfloat/multilingual-e5-small. Berikut adalah sebagian hasilnya:

| Metrik | Skor / Nilai |
| :---- | :---- |
| **PII Redaction Rate** | 100% |
| **Judge Faithfulness Score** | 70 |
| **Judge Relevance Score** | 85 |
| **Average Latency** | 5.76 detik |
| **Estimated Cost per Query** | $0.00326 |

*Catatan Limitasi:* Saat ini sistem masih mengalami kendala pada skor MRR@5 dan guardrails yang masih mengandalkan pendeteksian berbasis kata kunci. Pengembangan lanjutan difokuskan pada penguatan logika *retrieval*.

*Dibuat oleh Syahrial Jeremia Sinaga \- Institut Teknologi Del*
