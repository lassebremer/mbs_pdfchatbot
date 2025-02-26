from dotenv import load_dotenv
import os
from transformers import AutoTokenizer, AutoModel
import requests
import torch
import numpy as np
import sqlite3
import re
import logging
import torch.cuda
from datetime import datetime
from app.utils import get_database_path

# Lade Umgebungsvariablen und setze TensorFlow Logging-Level
load_dotenv()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
PDF-Chatbot Kernfunktionalität
------------------------------

Implementiert die Hauptlogik des PDF-Chatbots:

- Verarbeitung von Benutzeranfragen
- Vektorbasierte Ähnlichkeitssuche in PDF-Dokumenten
- Integration mit Hugging Face Transformers für Embeddings
- Verwaltung des Chat-Verlaufs
- Intelligente Antwortgenerierung basierend auf PDF-Inhalten

Nutzt KI-Modelle für Textverständnis und semantische Suche.
"""

class VectorDBChatbot:
    def __init__(self, db_name='default'):
        # Lade das Embedding-Modell aus den Umgebungsvariablen
        EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
        if not EMBEDDING_MODEL:
            raise ValueError("Umgebungsvariable 'EMBEDDING_MODEL' ist nicht gesetzt.")
        
        # Lade Threshold-Werte für die Ähnlichkeitssuche
        self.initial_threshold = float(os.getenv("INITIAL_THRESHOLD", "0.25"))
        self.final_threshold = float(os.getenv("FINAL_THRESHOLD", "0.3"))
        
        try:
            # Initialisiere das KI-Modell mit CUDA-Unterstützung falls verfügbar
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
            self.model = AutoModel.from_pretrained(EMBEDDING_MODEL)
            self.model = self.model.to(self.device)
            
        except Exception as e:
            logging.error(f"Fehler beim Laden des Models: {str(e)}")
            raise

        # Konfigurationsvariablen
        self.k = 4  # Anzahl der zurückzugebenden ähnlichen Dokumente
        self.db_name = db_name
        self.chat_history = []

        # API-Konfiguration für Hugging Face
        self.api_token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not self.api_token:
            raise ValueError("Umgebungsvariable 'HUGGINGFACE_API_TOKEN' ist nicht gesetzt.")
        
        self.llm_model_name = os.getenv("LLM_MODEL_NAME")
        if not self.llm_model_name:
            raise ValueError("Umgebungsvariable 'LLM_MODEL_NAME' ist nicht gesetzt.")
        
        self.llm_api_url = f"https://api-inference.huggingface.co/models/{self.llm_model_name}"

    def get_db_connection(self):
        """Erstellt eine Verbindung zur SQLite-Datenbank"""
        return sqlite3.connect(get_database_path(self.db_name))

    def generate_embedding(self, texts):
        """Generiert Embeddings für die gegebenen Texte"""
        print("\n=== DEBUG: Starte Embedding-Generierung ===")
        print(f"Anzahl Texte: {len(texts) if isinstance(texts, list) else 1}")
        print(f"Device: {self.device}")
        
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # Tokenisiere die Eingabetexte
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generiere Embeddings ohne Gradient-Berechnung
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].squeeze()
                if self.device.type == 'cuda':
                    embeddings = embeddings.cpu()
                embeddings = embeddings.numpy()
                print("Embedding erfolgreich generiert")
                return embeddings if len(texts) > 1 else embeddings.reshape(1, -1)
        except Exception as e:
            print(f"FEHLER bei Embedding-Generierung: {str(e)}")
            raise

    def get_similar_chunks(self, query, similarity_threshold=0.5):
        """Findet ähnliche Textabschnitte basierend auf der Eingabefrage"""
        # Generiere Embedding für die Anfrage
        query_embedding = self.generate_embedding(query)
        query_embedding = np.nan_to_num(query_embedding, 0)

        # Hole alle Dokumente aus der Datenbank
        with self.get_db_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT id, content, embedding, file_path FROM documents')
            results = c.fetchall()

        if not results:
            return []

        # Berechne Ähnlichkeiten
        similarities = []
        for doc_id, content, embedding_blob, file_path in results:
            try:
                # Konvertiere das gespeicherte Embedding
                doc_embedding = np.frombuffer(embedding_blob, dtype=np.float32).copy()
                doc_embedding = np.nan_to_num(doc_embedding, 0)

                # Berechne Kosinus-Ähnlichkeit
                query_norm = np.linalg.norm(query_embedding)
                doc_norm = np.linalg.norm(doc_embedding)

                if query_norm == 0 or doc_norm == 0:
                    continue

                similarity = np.dot(query_embedding, doc_embedding) / (query_norm * doc_norm)

                if similarity >= similarity_threshold:
                    similarities.append((float(similarity), content, doc_id, file_path))
            except Exception as e:
                print(f"Fehler bei Dokument {doc_id}: {str(e)}")
                continue

        # Sortiere nach Ähnlichkeit und gebe die Top-k zurück
        similarities.sort(reverse=True)
        return similarities[:self.k]

    def get_response(self, query):
        """Generiert eine Antwort basierend auf der Eingabefrage"""
        print("\n=== DEBUG: Starte get_response ===")
        print(f"Eingabe-Query: {query}")
        print(f"Initial Threshold: {self.initial_threshold}")
        print(f"Final Threshold: {self.final_threshold}")

        # Hole ähnliche Textabschnitte
        similar_chunks = self.get_similar_chunks(query, similarity_threshold=self.initial_threshold)
        
        # Debug-Ausgabe der gefundenen Chunks
        print("\n=== DEBUG: Gefundene Chunks ===")
        for i, (similarity, content, doc_id, file_path) in enumerate(similar_chunks, 1):
            print(f"\nChunk {i}:")
            print(f"Datei: {file_path}")
            print(f"Ähnlichkeit: {similarity:.4f}")
            print(f"Doc ID: {doc_id}")
            print(f"Inhalt (erste 100 Zeichen): {content[:100]}...")

        # Filtere und sortiere relevante Chunks
        relevant_chunks = [(s, c, d, f) for s, c, d, f in similar_chunks if s >= self.final_threshold]
        relevant_chunks.sort(key=lambda x: x[0], reverse=True)
        used_chunks = relevant_chunks[:4]
        
        # Erstelle Kontext aus den relevanten Chunks
        context = "\n\n---\n\n".join([content for _, content, _, _ in used_chunks])
        
        print(f"\n=== Finaler Kontext ===")
        print(f"Anzahl verwendeter Chunks: {len(used_chunks)}")
        print(f"Verwendete Chunks nach Ähnlichkeit:")
        for i, (similarity, _, _, file_path) in enumerate(used_chunks, 1):
            print(f"  {i}. {os.path.basename(file_path)} (Ähnlichkeit: {similarity:.3f})")

        if not context.strip():
            return "Keine relevanten Informationen gefunden.", []

        # Erstelle den Prompt für das LLM
        prompt = f"""### Rolle und Kontext
            Du bist ein hilfreicher Assistent, der präzise und faktentreue Antworten basierend auf dem bereitgestellten Kontext gibt. Deine Antworten sollen direkt und ohne zusätzliche Ausgaben oder Verweise auf den Kontext erfolgen.

            Aufgabe
            Frage: {query}
            ###
            Verfügbarer Kontext:
            {context}
            ###

            Anweisungen
            Lies den bereitgestellten Kontext sorgfältig.
            Identifiziere die relevanten Informationen für die Frage.
            Formuliere eine Antwort, die ausschließlich die gesuchte Information enthält.
            
            Antwortformat
            Gib ausschließlich die benötigte Information aus.
            Vermeide jegliche Verweise auf den Kontext oder die Methodik.
            Bei fehlenden Informationen antworte mit: "Die benötigte Information ist im Kontext nicht vorhanden."
            
            Wichtige Richtlinien
            Keine Ausgabe von Kontextinhalten in der Antwort.
            Keine Spekulationen oder Interpretationen über den Kontext hinaus.
            Transparenz bei fehlenden Informationen.
        """
            
            
        # Hole und bereinige die Antwort vom LLM
        answer = self.get_llm_response(prompt)
        clean_answer = self.clean_answer(answer)

        # Aktualisiere Chat-Verlauf
        self.chat_history.append(("user", query))
        self.chat_history.append(("assistant", clean_answer))

        # Erstelle PDF-Links für die Quelldokumente
        used_files = set()
        pdf_links = []
        for similarity, _, doc_id, file_path in used_chunks:
            if os.path.exists(file_path) and file_path not in used_files:
                used_files.add(file_path)
                pdf_links.append({
                    "url": f"/pdf/{doc_id}",
                    "name": f"{os.path.basename(file_path)} (Relevanz: {similarity:.2f})"
                })
        
        return clean_answer, pdf_links

    def clean_answer(self, answer):
        """Bereinigt die Antwort von Systemprompts und Formatierungen"""
        patterns = [
            r'\[.*?\]',           # Alles in eckigen Klammern
            r'<.*?>',             # HTML-ähnliche Tags
            r'Antwort:',          # "Antwort:"
            r'Assistant:',        # "Assistant:"
            r'###.*?###',         # Alles zwischen ### Markierungen
            r'Rolle und Kontext:.*?(?=\n\n)', # Rolle und Kontext Block
            r'Verfügbarer Kontext:.*?(?=\n\n)', # Kontext Block
            r'Anweisungen:.*?(?=\n\n)',  # Anweisungen Block
            r'Wichtige Richtlinien:.*?(?=\n\n)' # Richtlinien Block
        ]
        
        # Wende alle Patterns nacheinander an
        cleaned = answer
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        # Normalisiere Whitespace
        cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned

    def get_llm_response(self, prompt):
        """Holt eine Antwort vom Language Model via API"""
        print("\n=== DEBUG: Sende Anfrage an LLM ===")
        print(f"Prompt-Länge: {len(prompt)} Zeichen")
        print(f"API URL: {self.llm_api_url}")
        
        # API-Konfiguration
        headers = {"Authorization": f"Bearer {self.api_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": 500,
                "temperature": 0.2,
                "top_p": 0.7,
                "do_sample": True
            }
        }
        
        try:
            # Sende Anfrage an API
            response = requests.post(self.llm_api_url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0]["generated_text"]
                
                # Bereinige die Antwort
                if prompt in generated_text:
                    generated_text = generated_text.replace(prompt, '').strip()
                
                query = prompt.split("Frage: ")[-1].split("\n")[0]
                if query in generated_text:
                    generated_text = generated_text.replace(query, '').strip()
                
                print(f"LLM Antwort erhalten: {len(generated_text)} Zeichen")
                return self.clean_answer(generated_text)
                
            return "Entschuldigung, es gab ein unerwartetes Antwortformat."
            
        except Exception as e:
            print(f"FEHLER bei LLM-Anfrage: {str(e)}")
            return "Entschuldigung, es gab einen Fehler bei der Verarbeitung Ihrer Anfrage."

    def log_interaction(self, prompt, model_response, chat_history):
        """Protokolliert die Chatbot-Interaktion"""
        # Erstelle logs Ordner
        log_dir = "chat_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Erstelle Logdatei mit Zeitstempel
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{log_dir}/chat_{timestamp}.txt"
        
        # Schreibe Interaktionsdaten
        with open(filename, "w", encoding="utf-8") as f:
            f.write("=== PROMPT ===\n")
            f.write(prompt + "\n\n")
            
            f.write("=== MODEL RESPONSE ===\n")
            f.write(model_response + "\n\n")
            
            f.write("=== CHAT HISTORY ===\n")
            for role, content in chat_history:
                f.write(f"{role}: {content}\n")

    def switch_database(self, new_db_name):
        """Wechselt zu einer anderen Datenbank"""
        self.db_name = new_db_name
        self.chat_history = []  # Setzt Chat-Verlauf zurück beim Datenbankwechsel


    

