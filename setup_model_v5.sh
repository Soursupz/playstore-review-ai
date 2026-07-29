#!/bin/bash
# ============================================================
# Script bantu: pindahkan file model v5 ke struktur folder
# yang dibutuhkan predictor.py
# ============================================================
# Cara pakai:
#   1. Taruh script ini di root project chatbot kamu
#   2. Taruh folder hasil training "indobert_multikategori_v5"
#      (dari Cell 16 Enhanced_Model_v5_MultiKategori.ipynb) di
#      folder yang sama dengan script ini
#   3. Jalankan: bash setup_model_v5.sh
# ============================================================

set -e

SOURCE_DIR="indobert_multikategori_v5"
TARGET_DIR="predictor/sentiment_model_v5"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: Folder '$SOURCE_DIR' tidak ditemukan."
    echo "Pastikan kamu sudah download & extract hasil Cell 16 training v5 ke sini."
    exit 1
fi

mkdir -p "$TARGET_DIR"

REQUIRED_FILES=(
    "model_weights.pt"
    "tokenizer.json"
    "vocab.txt"
    "config.json"
    "special_tokens_map.json"
    "tokenizer_config.json"
)

for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "$SOURCE_DIR/$f" ]; then
        cp "$SOURCE_DIR/$f" "$TARGET_DIR/$f"
        echo "  Copied: $f"
    else
        echo "  PERINGATAN: $f tidak ditemukan di $SOURCE_DIR -- cek lagi hasil trainingnya."
    fi
done

# File referensi tambahan (opsional, tidak dipakai predictor.py saat runtime,
# tapi bagus buat dokumentasi/lampiran skripsi)
for f in "training_history.csv" "model_metadata.csv" "evaluasi_per_kategori.csv"; do
    if [ -f "$SOURCE_DIR/$f" ]; then
        cp "$SOURCE_DIR/$f" "$TARGET_DIR/$f"
        echo "  Copied (referensi): $f"
    fi
done

echo ""
echo "Selesai. Isi $TARGET_DIR:"
ls -la "$TARGET_DIR"
echo ""
echo "Sekarang predictor.py akan otomatis load model v5 dari folder ini."
