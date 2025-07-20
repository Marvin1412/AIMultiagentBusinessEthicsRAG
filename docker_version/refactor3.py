#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 12:20:00 2025

@author: lee
"""
# %% Imports


import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from deep_translator import GoogleTranslator
import random
from transformers import pipeline
import os
import fitz  # PyMuPDF für PDF-Verarbeitung
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import pytesseract
from pdf2image import convert_from_path
import pandas as pd
from rank_bm25 import BM25Okapi
from transformers import BitsAndBytesConfig, AutoModelForCausalLM

# %%
print("CUDA verfügbar:", torch.cuda.is_available())
print("GPU erkannt:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Keine GPU")


# %% definitions

# Modellname

drive_folder = os.getcwd()
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


bnb = BitsAndBytesConfig(
    load_in_4bit=True,                   # or load_in_8bit=True
    bnb_4bit_quant_type="nf4",           # optimal quant type
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb,
    device_map="auto",
)

#  Tokenizer & Modell laden
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

# Explizites PAD-Token setzen, falls nicht vorhanden
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)


# 📌 Beispielhafte Antwortlisten für Agenten
GESINNUNGEN = {
    "ökologisch": [
        "Wir müssen Ressourcen schützen.",
        "Nachhaltigkeit ist der Schlüssel zur Zukunft.",
        "Unsere Umwelt darf nicht leiden.",
        "Ökologie sollte an erster Stelle stehen.",
        "Lass uns gemeinsam Verantwortung übernehmen."
    ],
    "ökonomisch": [
        "Wirtschaftswachstum sichert Arbeitsplätze.",
        "Profite sind entscheidend für Fortschritt.",
        "Effizienz ist der Schlüssel zum Erfolg.",
        "Ohne Wirtschaft kein Wohlstand.",
        "Investitionen bringen uns voran."
    ],
    "innovativ": [
        "Neue Technologien können beide Seiten verbinden.",
        "Wir müssen über den Tellerrand hinaus denken.",
        "Innovation ist die beste Lösung für die Zukunft.",
        "Es gibt immer eine kreative Lösung.",
        "Lass uns neue Wege erkunden!"
    ],
    "winwinsituation": [
        "Wir brauchen eine Lösung, die für beide vorteilhaft ist.",
        "Gemeinsam können wir eine bessere Zukunft gestalten.",
        "Ein fairer Ausgleich bringt langfristig den größten Nutzen.",
        "Lasst uns eine Strategie entwickeln, von der beide profitieren.",
        "Zusammenarbeit ist der Schlüssel für nachhaltigen Erfolg."
    ]
}

SPIELREGEL = "Du bist ein intelligenter Debatten-Chatbot mit einer klaren politischen Haltung. \
Antworte immer mit einem einzigen, klaren Argument in einem Satz. \
Vermeide Füllwörter und irrelevante Aussagen."
SPIELREGEL_MOD = "Du bist ein intelligenter Antwort-Chatbot. Antworte stets mit einem einzigen, klaren Argument in einem Satz.Vermeide Wörter wie: Chatbot."
   
    
    
# %% Functions

#  Funktion zur Generierung von Antworten
def generate_response(prompt, max_new_tokens=200):
    """Generiert eine Antwort auf ein gegebenes Prompt mit korrekter Attention Mask."""

    # Falls das Modell eine Chat-Vorlage benötigt, wende sie an
    if hasattr(tokenizer, "apply_chat_template"):
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        formatted_prompt = prompt

    # Konvertiere den Prompt in Tokens
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # ⚠️ Setze eine explizite Attention Mask
    attention_mask = inputs.input_ids.ne(tokenizer.pad_token_id).to(model.device)

    # Generiere die Antwort mit optimierten Parametern
    output = model.generate(
        inputs.input_ids,
        attention_mask=attention_mask,  # <-- Hier wird die Attention Mask gesetzt
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False  # Aktiviert Sampling für kreativere Antworten
        #temperature=0.7,  # Kontrolliert die Zufälligkeit der Antwort
        #top_p=0.9,  # Begrenzt die Auswahl auf die wahrscheinlichsten Token
        #repetition_penalty=1.2  # Verhindert endlose Wiederholungen
    )

    # Dekodiere die Antwort
    response = tokenizer.decode(output[:, inputs.input_ids.shape[-1]:][0], skip_special_tokens=True).strip()

    # Falls die Antwort leer ist, gib eine alternative Antwort
    if not response or len(response) < 5:
        response = "Entschuldigung, ich bin mir nicht sicher. Kannst du die Frage anders formulieren?"

    return response

def get_first_two_sentences(text):
    # 🔹 Trenne den Text korrekt an Satzzeichen (.!?), aber achte auf nachfolgende Zahlen
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    # 🔹 Entferne angehängte Zahlen am Satzende (z. B. ".3.", "4.")
    sentences = [re.sub(r'(\.\d+|\d+\.)$', '.', sentence).strip() for sentence in sentences]

    # 🔹 Nimm nur die ersten zwei saetze und setze sie wieder zusammen
    return " ".join(sentences[:2])



def translate_to_german(text):
    return GoogleTranslator(source='auto', target='de').translate(text)


def generate_gesinnung(gesinnung):
    return random.choice(GESINNUNGEN.get(gesinnung, ["Ich bin mir nicht sicher."]))

# Summarization-Modell laden
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def extrahiere_schlüsselwörter(text, num_words=3):
    """
    Extrahiert die ersten num_words sinnvollen Wörter aus der englischen Zusammenfassung.
    - Ignoriert Artikel (the, a, an) und andere irrelevante Wörter.
    """
    words = text.split()
    stopwords = {"the", "a", "an", "of", "and", "in", "on", "by", "with", "for", "to", "at", "from"}
    relevante_wörter = []

    for word in words:
        # Entfernt Sonderzeichen und macht es klein
        clean_word = re.sub(r"[^\w]", "", word).lower()
        if clean_word and clean_word not in stopwords:
            relevante_wörter.append(clean_word)
            if len(relevante_wörter) == num_words:
                break

    return " ".join(relevante_wörter) if relevante_wörter else "unknown"

def generiere_schlüsselwort_und_zusammenfassung(deutscher_satz):
    """
    Erstellt eine kurze englische Zusammenfassung eines deutschen Satzes und generiert Schlüsselwörter.

    Rückgabe: (Schlüsselwörter, Zusammenfassung)
    """
    try:
        # 🏗 Schritt 1: Deutsche Zusammenfassung erzeugen
        summary_de = summarizer(deutscher_satz, max_length=40, min_length=5, do_sample=False)[0]["summary_text"]

        # 🏗 Schritt 2: Ins Englische übersetzen
        summary_en = GoogleTranslator(source='de', target='en').translate(summary_de)

        # 🏗 Schritt 3: Schlüsselwörter extrahieren
        key_words = extrahiere_schlüsselwörter(summary_en, num_words=3)

        return key_words, summary_en
    except Exception as e:
        return "error", f"Fehler: {str(e)}"

# %% classes

class RAGModerator:
    def __init__(self, name, gesinnung, text_path):
        self.name = name
        self.gesinnung = gesinnung
        self.text_path = text_path
        self.erster_prompt = f"Du bist der Moderator {self.name}. Deine Aufgabe ist es, beide Seiten zur Kooperation zu fördern."
        self.texte, self.quellen = self.lade_texte()
        self.saetze = self.texte 
        self.model = SentenceTransformer("distiluse-base-multilingual-cased")
        self.index, self.embeddings, _ = self.build_faiss_index()

        # BM25 zur Schlagwortsuche
        tokenized_corpus = [t.split() for t in self.texte]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def lade_texte(self, path="./data"):
            
        gesamt_text = []
        quellen     = []
        for file_name in os.listdir(path):
            if not file_name.lower().endswith('.txt'):
                continue

            file_path = os.path.join(path, file_name)
            try:
                # 1) Read the UTF‑8 text
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_text = f.read()

                # 2) Clean up (strip extra linebreaks, headers like "Seite", etc.)
                clean = re.sub(r'\n+', ' ', raw_text)
                clean = re.sub(r'\s{2,}', ' ', clean)
                clean = re.sub(r'(Kapitel [0-9]+|Seite [0-9]+|Autor:.*?|Datum:.*?)',
                            '', clean, flags=re.IGNORECASE)

                # 3) Split into sentence‑sized passages
                passages = [
                    s.strip() for s in re.split(r'(?<=[.!?])\s+', clean)
                    if len(s) > 50
                ]

                # 4) Collect
                gesamt_text.extend(passages)
                quellen.extend([file_name] * len(passages))

            except Exception as e:
                print(f"Fehler beim Laden von {file_name}: {e}")

        # Fallback if folder was empty
        if not gesamt_text:
            gesamt_text = ["Keine relevanten Texte gefunden."]
            quellen     = ["Unbekannte Quelle"]

        return gesamt_text, quellen


    def build_faiss_index(self):
        """Erstellt einen FAISS-Index für die semantische Suche."""
        if not self.texte:
            return None, None, None

        embeddings = self.model.encode(self.texte, convert_to_numpy=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        return index, embeddings, self.texte

    def abrufen_wissen(self, suchwörter, return_source=False):
        """Sucht eine relevante Passage mit BM25 oder FAISS und gibt optional die Quelle aus."""
        if not self.texte:
            return "Keine relevanten Passagen gefunden."

        query_words = suchwörter.split()
        bm25_scores = self.bm25.get_scores(query_words)

        # Höchstbewerteten Satz finden
        top_idx = np.argmax(bm25_scores)
        if bm25_scores[top_idx] > 0:
            return (self.saetze[top_idx], self.quellen[top_idx]) if return_source else self.saetze[top_idx]

        # Falls BM25 nichts findet, FAISS als Backup verwenden
        query_embedding = self.model.encode([suchwörter], convert_to_numpy=True)
        _, I = self.index.search(query_embedding, k=3)
        top_passages = [self.saetze[i] for i in I[0]]
        top_sources = [self.quellen[i] for i in I[0]]

        if not top_passages:
            return "Keine relevanten Passagen gefunden."

        return (" ".join(top_passages), " ".join(top_sources)) if return_source else " ".join(top_passages)

    def moderieren(self, letzter_satz, return_source=False):
        """Der Moderator gibt eine kurze Antwort basierend auf dem letzten Satz & RAG."""
        rag_passage = self.abrufen_wissen(letzter_satz, return_source=return_source)
        if return_source:
            if isinstance(rag_passage, tuple):
                if len(rag_passage) == 2:
                    return rag_passage  # Gibt Passage und Quelle zurück
                return rag_passage[0], "Unbekannte Quelle"  # Falls keine Quelle gefunden wurde
            elif isinstance(rag_passage, str):
                return rag_passage, "Unbekannte Quelle"
            elif isinstance(rag_passage, list) and len(rag_passage) >= 2:
                return rag_passage[0], rag_passage[1]  # Nimmt die ersten zwei Werte als Passage und Quelle
            return rag_passage, "Unbekannte Quelle"

class Agent:
    def __init__(self, name, gesinnung):
        self.name = name
        self.gesinnung = gesinnung
        self.erster_gedanke = generate_gesinnung(gesinnung)
        self.last_sentence = ""  # Um den letzten Satz des Gesprächspartners zu speichern

    def generate_with_tiny_llama(self,prompt,max_tokens=256):
        messages = [
            {"role": "system", "content": SPIELREGEL},  # Instruktionen für den Agenten
            {"role": "user", "content": prompt}  # Die eigentliche Eingabe
        ]

        # 🔹 Teste, ob das Modell besser antwortet, wenn man die `apply_chat_template` umgeht
        formatted_prompt = f"{SPIELREGEL_MOD}\nBenutzer: {prompt}\nAntwort:"

        # Generierung mit einer Mischung aus Zufälligkeit und Kontrolle
        outputs = pipe(
            formatted_prompt,
            max_new_tokens=max_tokens,
            do_sample=True,  # Kreativität aktivieren
            temperature=0.7,  # Kontrolle über die Varianz
            top_k=50,         # Begrenzung der Auswahl
            top_p=0.95,        # Begrenzung auf wahrscheinlichste Token
            repetition_penalty=1.2,  # Verhindert generische Wiederholungen
           no_repeat_ngram_size=2
        )

        # 🔹 Extraktion der Antwort – stelle sicher, dass nur der generierte Teil genommen wird
        response = outputs[0]["generated_text"]

        # 🔹 Falls das Modell das Prompt "spiegelt", schneiden wir es sauber ab
        if prompt in response:
            response = response.split(prompt)[-1].strip()

        return response.strip()

    def sprechen(self, letzter_satz=""):
        if letzter_satz:
            self.last_sentence = letzter_satz

        # Dynamischer Prompt basierend auf Gesinnung und letztem Satz
        prompt = (f"Du bist {self.name} mit der Gesinnung {self.gesinnung}. "
                  f"Dein vorheriger Standpunkt war: '{self.erster_gedanke}'. "
                  f"Der letzte Satz deines Gegenübers lautete: '{self.last_sentence}'. "
                  "Bringe ein starkes Argument gegen diesen Satz.")

        return self.generate_with_tiny_llama(prompt)

    def entscheide(self):
        entscheidungs_prompts = [
            "Wie entscheidest du dich: kooperieren oder nicht kooperieren?",
            "Bitte antworte: kooperieren oder nicht kooperieren?",
            "Triff deine Wahl: Kooperieren oder nicht kooperieren?",
            f"Basierend auf deiner Gesinnung ({self.gesinnung}), wirst du kooperieren oder nicht kooperieren?"
        ]

        # Zufällige Entscheidungsfrage wählen
        prompt = random.choice(entscheidungs_prompts)
        entscheidung = self.generate_with_tiny_llama(prompt,max_tokens=10)

        # Filterung: Nur "kooperieren" oder "nicht kooperieren" akzeptieren
        if "kooperieren" in entscheidung.lower():
            return "kooperieren"
        elif "nicht kooperieren" in entscheidung.lower():
            return "nicht kooperieren"
        else:
            return random.choice(["kooperieren", "nicht kooperieren"])  # Falls unsicher, zufällig wählen

# %% Run Functions

def run_no_mod(runden=25, output_csv="agenten_dialog_pure.csv"):
    """
    Führt eine Simulation ohne Moderator durch und speichert die Ergebnisse im Unterordner 'no_mod'.
    Das Format entspricht dem von run_with_mod.
    """
    # Erstelle zwei Agenten mit unterschiedlichen Eigenschaften
    agent_a = Agent(name="Agent Ökologisch A", gesinnung="ökologisch")
    agent_b = Agent(name="Agent Ökonomisch B", gesinnung="ökonomisch")

    protokoll = []
    letzter_satz = " Es gab noch kein Gespräch. "
    letzter_satz_a = letzter_satz
    letzter_satz_b = letzter_satz

    for runde in range(1, runden + 1):
        print(f"\n **Runde {runde}**")

        agent_a_antwort = agent_a.sprechen(letzter_satz_b)
        agent_b_antwort = agent_b.sprechen(letzter_satz_a)

        antwort_a_bereinigt = translate_to_german(get_first_two_sentences(agent_a_antwort.replace("<|assistant|>", "").replace("Assistant:", "").replace("\n","").replace("Antwort:","").strip()))
        antwort_b_bereinigt = translate_to_german(get_first_two_sentences(agent_b_antwort.replace("<|assistant|>", "").replace("Assistant:", "").replace("\n","").replace("Antwort:","").strip()))

        agent_a_entscheidung = agent_a.entscheide()
        agent_b_entscheidung = agent_b.entscheide()

        print(f" **Agent A:** {antwort_a_bereinigt}...")
        print(f" **Agent A entscheidet:** {agent_a_entscheidung}")
        print("**")
        print(f" **Agent B:** {antwort_b_bereinigt}...")
        print(f" **Agent B entscheidet:** {agent_b_entscheidung}")
        print("-" * 50)

        # Format analog zu run_with_mod
        protokoll.append({
            "runde": runde,
            "antwort_a": antwort_a_bereinigt,
            "antwort_b": antwort_b_bereinigt,
            "entscheidung_a": agent_a_entscheidung,
            "entscheidung_b": agent_b_entscheidung,
            "moderation_a": None,
            "moderation_b": None,
            "quelle_a": None,
            "quelle_b": None
        })

        letzter_satz_b = antwort_b_bereinigt.replace("<|assistant|>", "").replace("Assistant:", "").strip()
        letzter_satz_a = antwort_a_bereinigt.replace("<|assistant|>", "").replace("Assistant:", "").strip()

        print(f" **Runde {runde} gespeichert!**")
        print("-" * 50)

    protokoll_df = pd.DataFrame(protokoll)
    # Finde einen freien Dateinamen mit Nummerierung (_0000, _0001, ...)
    base, ext = os.path.splitext(output_csv)
    i = 0
    # Speichere Ergebnisse im Unterordner 'no_mod'
    output_dir = os.path.join(drive_folder, "no_mod")
    os.makedirs(output_dir, exist_ok=True)
    while True:
        numbered_csv = f"{base}_{i:04d}{ext}"
        full_path = os.path.join(output_dir, numbered_csv)
        if not os.path.exists(full_path):
            break
        i += 1
    protokoll_df.to_csv(full_path, mode="w", header=True, index=False, encoding="utf-8")

    print("\n **Simulation beendet!**")

def run_with_mod(runden=25, output_csv="agenten_dialog_mod.csv", text_path=None):
    """
    Führt eine Simulation mit Moderator durch und speichert die Ergebnisse im Unterordner 'with_mod'.
    """
    # Erstelle zwei Agenten und einen Moderator
    # Erstelle zwei Agenten und einen Moderator
    agent_c = Agent(name="Agent Innovativ C", gesinnung="innovativ")
    agent_d = Agent(name="Agent WinWin D", gesinnung="winwinsituation")
    moderator = RAGModerator(name="Moderator", gesinnung="neutral", text_path=text_path or os.getcwd())

    protokoll = []
    letzter_c = "Was denkst Du?"
    letzter_d = "Was denkst Du?"

    for runde in range(1, runden + 1):
        print(f"\n Runde {runde}\n{'='*30}")

        agent_c_antwort = agent_c.sprechen(letzter_d)
        agent_d_antwort = agent_d.sprechen(letzter_c)

        antwort_c_clean = translate_to_german(get_first_two_sentences(agent_c_antwort.replace("<|assistant|>", "").replace("Assistant:", "").replace("\n","").replace("Antwort:","").strip()))
        antwort_d_clean = translate_to_german(get_first_two_sentences(agent_d_antwort.replace("<|assistant|>", "").replace("Assistant:", "").replace("\n","").replace("Antwort:","").strip()))

        entscheidung_c = agent_c.entscheide()
        entscheidung_d = agent_d.entscheide()

        print(f"{agent_c.name} sagt: {antwort_c_clean}")
        print(f"{agent_d.name} sagt: {antwort_d_clean}")
        print(f"{agent_c.name} Entscheidung: {entscheidung_c}")
        print(f"{agent_d.name} Entscheidung: {entscheidung_d}")

        moderation_c = moderator.moderieren(antwort_c_clean, return_source=True)
        if isinstance(moderation_c, tuple) and len(moderation_c) == 2:
            text_c, quelle_c = moderation_c
        else:
            text_c, quelle_c = "Moderation fehlgeschlagen oder leer", "Unbekannt"

        if "•" in text_c:
            liste_c, haupttext_c = [t.strip() for t in text_c.split("•", 1)]
        else:
            liste_c = None
            haupttext_c = text_c

        print("Liste:", liste_c)
        print("Haupttext:", haupttext_c)
        print("Quelle:", quelle_c)

        moderation_d = moderator.moderieren(antwort_d_clean, return_source=True)
        if isinstance(moderation_d, tuple) and len(moderation_d) == 2:
            text_d, quelle_d = moderation_d
        else:
            text_d, quelle_d = "Moderation fehlgeschlagen oder leer", "Unbekannt"

        if "•" in text_d:
            liste_d, haupttext_d = [t.strip() for t in text_d.split("•", 1)]
        else:
            liste_d = None
            haupttext_d = text_d

        print("Liste:", liste_d)
        print("Haupttext:", haupttext_d)
        print("Quelle:", quelle_d)

        print(f" Moderator zu C: {haupttext_c}")
        print(f" Moderator zu D: {haupttext_d}")

        letzter_c = haupttext_c
        letzter_d = haupttext_d

        protokoll.append({
            "runde": runde,
            "antwort_c": antwort_c_clean,
            "antwort_d": antwort_d_clean,
            "entscheidung_c": entscheidung_c,
            "entscheidung_d": entscheidung_d,
            "moderation_c": haupttext_c,
            "moderation_d": haupttext_d,
            "quelle_c": quelle_c,
            "quelle_d": quelle_d
        })
    protokoll_df = pd.DataFrame(protokoll)
    # Finde einen freien Dateinamen mit Nummerierung (_0000, _0001, ...)
    base, ext = os.path.splitext(output_csv)
    i = 0
    # Speichere Ergebnisse im Unterordner 'with_mod'
    output_dir = os.path.join(drive_folder, "with_mod")
    os.makedirs(output_dir, exist_ok=True)
    while True:
        numbered_csv = f"{base}_{i:04d}{ext}"
        full_path = os.path.join(output_dir, numbered_csv)
        if not os.path.exists(full_path):
            break
        i += 1
    protokoll_df.to_csv(full_path, mode="w", header=True, index=False, encoding="utf-8")

    print("\n Simulation abgeschlossen!")

#%%

# Beispielaufrufe:

for i in range(16):
    run_no_mod()
    #run_with_mod(text_path="./data")
    for name in ("agent_c", "agent_d", "moderator", "agent_a", "agent_b"):
        if name in globals():
            del globals()[name]

    torch.cuda.empty_cache()


# run_with_mod(text_path="./data")




































