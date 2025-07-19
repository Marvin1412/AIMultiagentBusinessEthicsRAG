# 📦 Imports
import os
import pytesseract
from pdf2image import convert_from_path
import fitz  # PyMuPDF
from transformers import MarianMTModel, MarianTokenizer
import torch

# 🔧 Tesseract OCR Settings
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
ocr_language = "deu"

# Translator vorbereiten: Englisch → Deutsch
model_name = "Helsinki-NLP/opus-mt-en-de"
tokenizer = MarianTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MarianMTModel.from_pretrained(model_name).to(device)

def translate_text(text, max_chunk_len=512):
    sentences = text.split('. ')
    translated = []
    buffer = ""
    for sentence in sentences:
        if len(buffer + sentence) < max_chunk_len:
            buffer += sentence + ". "
        else:
            tokens = tokenizer.prepare_seq2seq_batch([buffer], return_tensors="pt").to(device)
            translated_ids = model.generate(**tokens)
            translated.append(tokenizer.batch_decode(translated_ids, skip_special_tokens=True)[0])
            buffer = sentence + ". "
    if buffer:
        tokens = tokenizer.prepare_seq2seq_batch([buffer], return_tensors="pt").to(device)
        translated_ids = model.generate(**tokens)
        translated.append(tokenizer.batch_decode(translated_ids, skip_special_tokens=True)[0])
    return " ".join(translated)

# Funktion: Direkttext aus PDF extrahieren
def extract_text_direct(pdf_path):
    doc = fitz.open(pdf_path)
    text = "".join([seite.get_text() for seite in doc])
    doc.close()
    return text.strip()

# OCR-Fallback
def extract_text_ocr(pdf_path):
    images = convert_from_path(pdf_path)
    ocr_text = ""
    for i, img in enumerate(images):
        print(f" OCR Seite {i+1}")
        ocr_text += pytesseract.image_to_string(img, lang=ocr_language) + "\n"
    return ocr_text.strip()

# 📂 Verzeichnis mit PDFs
data_folder = "./data"

if not os.path.exists(data_folder):
    print(f"Ordner {data_folder} nicht gefunden!")
    exit(1)

print("Dateien im data-Ordner:", os.listdir(data_folder))

for filename in os.listdir(data_folder):
    print("Prüfe Datei:", filename)
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(data_folder, filename)
        print(f"\nVerarbeite: {pdf_path}")

        # 1. Versuche, Text direkt zu extrahieren
        text = extract_text_direct(pdf_path)
        if not text:
            print(" Kein Text gefunden – OCR wird gestartet...")
            text = extract_text_ocr(pdf_path)
        else:
            print(" Text direkt gefunden.")

        # 2. Bereinigen
        clean_text = text.replace("\n", " ").replace("  ", " ").strip()

        # 3. Übersetzen
        print(" Übersetze ins Deutsche...")
        translated_text = translate_text(clean_text)

        # 4. Speichern als .txt
        output_path = os.path.splitext(pdf_path)[0] + "_deutsch.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(translated_text)
        print(f" (Übersetzter Text gespeichert als: {output_path})")