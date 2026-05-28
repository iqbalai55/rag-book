MINDMAP_PROMPT = """Buat mindmap dari struktur TOC buikut dalam format Mermaid mindmap syntax.

Aturan:
1. Gunakan syntax: mindmap\n  root((Judul))\n    Cabang1\n      SubCabang1
2. Maksimal 3 level kedalaman (root -> cabang -> sub-cabang)
3. Jumlah cabang utama: 4-7 (sesuai jumlah bab utama)
4. Gunakan Bahasa Indonesia
5. Label harus ringkas (max 5-6 kata)
6. Sertakan sumber di akhir

STRUKTUR TOC:
{toc_struktur}

Topik Buku: {topik}

Generate Mermaid mindmap:"""

MINDMAP_FROM_CONTENT_PROMPT = """Buat mindmap dari konten buku berikut dalam format Mermaid mindmap syntax.

Aturan:
1. Gunakan syntax: mindmap\n  root((Judul))\n    Cabang1\n      SubCabang1
2. Maksimal 3 level kedalaman (root -> cabang -> sub-cabang)
3. Jumlah cabang utama: 4-7 (sesuai jumlah topik utama)
4. Gunakan Bahasa Indonesia
5. Label harus ringkas (max 5-6 kata)
6. Sertakan sumber di akhir

KONTEN:
{context}

Topik: {topik}

Generate Mermaid mindmap:"""
