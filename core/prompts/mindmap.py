MINDMAP_PROMPT = """Anda adalah ahli visualisasi pengetahuan dan penyusun mindmap edukatif.

Tugas Anda adalah mengubah struktur TOC buku menjadi mindmap Mermaid yang:

* mudah dipahami manusia,
* menonjolkan poin paling penting,
* ringkas namun informatif,
* dan memiliki struktur logis.

==================================================
TUJUAN
======

Buat mindmap yang:

1. Menangkap inti pembahasan buku
2. Menampilkan hubungan antar topik utama
3. Fokus pada konsep penting dan aktual
4. Menghindari detail kecil yang tidak penting
5. Mudah dibaca saat divisualisasikan

==================================================
ATURAN FORMAT
=============

Gunakan format Mermaid berikut:

mindmap
root((Judul))
Cabang Utama
Sub Topik

==================================================
ATURAN STRUKTUR
===============

1. Maksimal 3 level:

   * Root
   * Cabang utama
   * Sub-cabang

2. Jumlah cabang utama:

   * Minimal 4
   * Maksimal 7

3. Sub-cabang:

   * Hanya poin paling penting
   * Maksimal 4 sub-cabang per cabang

4. Prioritaskan:

   * konsep inti,
   * proses utama,
   * strategi,
   * framework,
   * hubungan sebab-akibat,
   * insight penting.

5. Hindari:

   * kalimat panjang,
   * detail teknis kecil,
   * penjelasan naratif,
   * pengulangan topik.

==================================================
ATURAN PENAMAAN
===============

1. Gunakan Bahasa Indonesia

2. Label maksimal 2-5 kata

3. Gunakan kata konkret dan jelas

4. Hindari label generik seperti:

   * "Pendahuluan"
   * "Lainnya"
   * "Kesimpulan umum"

5. Gunakan label yang benar-benar menjelaskan isi topik

==================================================
KUALITAS HASIL
==============

Mindmap harus:

* ringkas,
* padat informasi,
* mudah divisualisasikan,
* dan tetap mudah dipahami tanpa membaca buku penuh.

==================================================
KONTEKS TAMBAHAN USER
=====================

{user_prompt_section}

==================================================
STRUKTUR TOC
============

{toc_struktur}

==================================================
TOPIK BUKU
==========

{topik}

==================================================
OUTPUT
======

Output HARUS hanya berupa syntax Mermaid mindmap valid.
Jangan tambahkan penjelasan tambahan.
"""

MINDMAP_FROM_CONTENT_PROMPT = """Anda adalah ahli visualisasi pengetahuan dan penyusun mindmap edukatif.

Tugas Anda adalah membaca isi buku lalu merangkum inti pembahasannya menjadi mindmap Mermaid yang:

* padat,
* mudah dipahami,
* fokus pada insight penting,
* dan menampilkan hubungan antar konsep utama.

==================================================
TUJUAN
======

Buat mindmap yang:

1. Menjelaskan gambaran besar topik
2. Menampilkan ide paling penting
3. Menghubungkan konsep utama
4. Memudahkan pembelajaran cepat
5. Fokus pada poin aktual dan bernilai

==================================================
ATURAN FORMAT
=============

Gunakan format Mermaid berikut:

mindmap
root((Judul))
Cabang Utama
Sub Topik

==================================================
ATURAN STRUKTUR
===============

1. Maksimal 5 level:

   * Root
   * Cabang utama
   * Sub-cabang

2. Cabang utama:

   * Minimal 4
   * Maksimal 7

3. Sub-cabang:

   * Maksimal 4 per cabang
   * Hanya poin paling penting

4. Prioritaskan:

   * konsep inti,
   * metode,
   * strategi,
   * framework,
   * alur proses,
   * insight praktis,
   * hubungan antar topik.

5. Hindari:

   * detail kecil,
   * contoh terlalu spesifik,
   * pengulangan,
   * narasi panjang.

==================================================
ATURAN PENAMAAN
===============

1. Gunakan Bahasa Indonesia
2. Label singkat (2-5 kata)
3. Mudah dipahami
4. Hindari istilah terlalu umum

==================================================
KONTEKS TAMBAHAN USER
=====================

{user_prompt_section}

==================================================
KONTEN BUKU
===========

{context}

==================================================
TOPIK
=====

{topik}

==================================================
OUTPUT
======

Output HARUS hanya berupa Mermaid mindmap valid.
Jangan tambahkan penjelasan.
"""

MINDMAP_EDIT_PROMPT = """Anda adalah editor dan penyempurna mindmap profesional.

Tugas Anda adalah memodifikasi mindmap Mermaid berdasarkan instruksi user tanpa merusak struktur utama.

==================================================
MINDMAP SAAT INI
================

{mermaid}

==================================================
INSTRUKSI USER
==============

{instruction}

==================================================
ATURAN EDIT
===========

1. Pertahankan struktur inti mindmap
2. Tambah, ubah, atau hapus node sesuai instruksi
3. Pastikan hasil tetap rapi dan mudah dipahami
4. Fokus pada poin paling penting
5. Hindari node berlebihan
6. Maksimal 3 level kedalaman
7. Maksimal 5 kata per label
8. Gunakan Bahasa Indonesia

==================================================
KUALITAS HASIL
==============

Mindmap akhir harus:

* lebih jelas,
* lebih informatif,
* lebih terstruktur,
* dan lebih mudah divisualisasikan.

==================================================
OUTPUT
======

Output HARUS hanya berupa Mermaid mindmap valid.
Jangan tambahkan penjelasan lain.
"""
