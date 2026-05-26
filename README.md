# Detectra AI

Detectra AI merupakan aplikasi web berbasis Streamlit dan model BERT untuk
mendeteksi apakah sebuah teks memiliki pola yang lebih dekat dengan teks
manusia atau teks hasil generasi kecerdasan buatan.

## Model yang Digunakan

Model tersimpan pada Hugging Face Hub:

```python
MODEL_REPO = "runiftrynd/detectra-ai-bert"
```

Model merupakan hasil fine-tuning `bert-base-uncased` untuk klasifikasi biner:

| Label | Kelas |
|---|---|
| 0 | Teks Manusia |
| 1 | Teks AI |

## Fitur Aplikasi

- Input teks melalui antarmuka web.
- Prediksi teks manusia atau teks AI.
- Tampilan probabilitas kedua kelas.
- Informasi jumlah kata dan karakter.
- Catatan keterbatasan hasil deteksi.

## Menjalankan Secara Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Aplikasi dirancang untuk dideploy menggunakan Streamlit Community Cloud,
sedangkan model dipanggil langsung dari Hugging Face Hub.

## Batasan Model

Model dilatih menggunakan dataset berbahasa Inggris. Oleh karena itu,
prediksi terhadap teks bahasa Indonesia belum dapat dianggap valid tanpa
pengujian atau pelatihan tambahan menggunakan dataset bahasa Indonesia.

Hasil deteksi tidak dapat digunakan sebagai satu-satunya dasar untuk
menentukan plagiarisme, orisinalitas tulisan, atau pelanggaran akademik.
