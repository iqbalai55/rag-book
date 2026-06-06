CHAPTER_SUMMARY_PROMPT = """Anda adalah tutor ahli yang sedang meringkas materi buku.

Buat ringkasan chapter berikut dari konteks buku yang diberikan.

Chapter: {chapter_title}

Konteks Buku:
{context}

ATURAN:
1. Gunakan Bahasa Indonesia
2. Ringkasan harus padat tapi informatif (maksimal 3-4 kalimat)
3. Sertakan 3-5 poin penting dari chapter ini
4. Fokus pada konsep inti, definisi, dan contoh penting
5. Setiap key_point harus menjawab: "Apa yang akan saya pahami/setelah membaca ini?"
   Bukan hanya "apa topiknya". Format: "Membahas X → sehingga pembaca bisa memahami/menggunakan Y"
6. Sertakan 1-2 contoh konkret yang akan pembaca pelajari
7. Output HARUS valid JSON sesuai schema ChapterSummary
{user_prompt_section}

Contoh JSON yang diharapkan:
{{
  "chapter_title": "chapter_title",
  "summary": "ringkasan chapter",
  "key_points": ["poin 1", "poin 2", "poin 3"]
}}
"""

BOOK_SUMMARY_PROMPT = """Anda adalah tutor ahli yang sedang membuat ringkasan keseluruhan buku.

Berdasarkan ringkasan-ringkasan chapter berikut, buat ringkasan keseluruhan buku.

Ringkasan Chapter:
{chapter_summaries}

Topik Buku: {topic}

ATURAN:
1. Gunakan Bahasa Indonesia
2. Buat overview yang menjelaskan tujuan dan cakupan buku secara keseluruhan
3. Overview harus menjawab: "Apa nilai buku ini bagi saya?" bukan hanya "tentang apa buku ini"
4. Identifikasi 3-5 tema utama yang muncul di seluruh buku
5. Setiap tema harus menjelaskan: topik apa + insight apa yang pembaca dapat
   Format: "Tema: X → Pembaca akan memahami/menggunakan Y"
6. Berikan judul yang representatif untuk ringkasan ini
7. Overview maksimal 5-6 kalimat
8. Output HARUS valid JSON sesuai schema BookSummaryResponse
{user_prompt_section}

Contoh JSON yang diharapkan:
{{
  "title": "Ringkasan: Judul Buku",
  "overview": "Buku ini membahas tentang...",
  "key_themes": ["tema 1", "tema 2", "tema 3"]
}}
"""

SUMMARY_EDIT_PROMPT = """Anda adalah editor ringkasan profesional. Modifikasi ringkasan buku berikut berdasarkan instruksi user.

RINGKASAN SAAT INI:
Title: {title}
Overview: {overview}
Key Themes: {key_themes}

Chapter Summaries:
{chapter_summaries}

INSTRUKSI EDIT:
{instruction}

ATURAN:
1. Pertahankan struktur ringkasan (title, overview, chapters, key_themes)
2. Ubah/tambah/hapus sesuai instruksi user
3. Gunakan Bahasa Indonesia
4. Output HARUS valid JSON sesuai schema BookSummaryResponse
5. Jangan tambahkan penjelasan, hanya hasil ringkasan

Contoh JSON yang diharapkan:
{{
  "title": "Ringkasan: Judul Buku",
  "overview": "Ringkasan yang sudah dimodifikasi...",
  "key_themes": ["tema 1", "tema 2"]
}}

Hasil edit dalam format JSON:"""
