"""
PDF-Chatbot Hilfsfunktionen
---------------------------

Sammlung von Utility-Funktionen für den PDF-Chatbot:

- PDF-Verarbeitung und Textextraktion
- OCR für nicht-durchsuchbare PDFs
- Datenbank-Verwaltung und Verbindungs-Pooling
- Cache-Management
- Asynchrone Verarbeitung von PDF-Batches
- Thread-sichere Datenbankoperationen

Enthält kritische Infrastruktur-Komponenten für die PDF-Verarbeitung.
"""

import os
import json
import sqlite3
import numpy as np
import fitz  # PyMuPDF
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import torch
from datetime import datetime
import uuid
import pdfplumber
import re
import logging
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import torch.cuda
import threading
import time
import mmap
from io import BytesIO
import pytesseract
from PIL import Image
import pdf2image
import tempfile
from dotenv import load_dotenv
import multiprocessing

# Lade Umgebungsvariablen
load_dotenv()

# Tesseract-Konfiguration für Windows
if os.name == 'nt':  # Prüft ob Windows
    tesseract_path = os.getenv('TESSERACT_PATH')
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    else:
        print("Warnung: TESSERACT_PATH nicht in .env konfiguriert")

def get_database_path(db_name):
    """Generiert den vollständigen Pfad zur Datenbank"""
    db_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'databases')
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, f"{db_name}.db")

class ThreadSafeDBConnection:
    """Thread-sichere Datenbankverbindung"""
    def __init__(self, db_name='default'):
        """Initialisiert eine threadsichere Datenbankverbindung"""
        db_path = get_database_path(db_name)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.Lock()
    
    def execute(self, query, params=None):
        """Führt eine SQL-Abfrage thread-sicher aus"""
        with self.lock:
            cursor = self.conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            self.conn.commit()
            return cursor
    
    def close(self):
        """Schließt die Datenbankverbindung"""
        self.conn.close()

def get_db_connection(db_name='default'):
    """Erstellt eine thread-sichere Datenbankverbindung"""
    return ThreadSafeDBConnection(db_name)

def create_memory_mapped_cache(cache_file, size):
    """Erstellt eine Memory-Mapped Datei für den Cache"""
    with open(cache_file, 'wb') as f:
        f.write(b'\0' * size)
    return mmap.mmap(f.fileno(), size)

async def process_single_pdf_async(pdf_path, chunk_size=1000, chunk_overlap=200):
    """Verarbeitet eine einzelne PDF-Datei asynchron"""
    try:
        print(f"\nVerarbeite PDF: {os.path.basename(pdf_path)}")
        start_time = time.time()
        chunks = []
        
        # Cache-Verwaltung
        cache_dir = os.path.join(os.path.dirname(pdf_path), '.cache')
        cache_file = os.path.join(cache_dir, f"{os.path.basename(pdf_path)}_chunks.pickle")
        
        # Prüfe Cache
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                return pickle.loads(mm.read())
        
        # PDF-Verarbeitung in ThreadPool
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            chunks = await loop.run_in_executor(pool, 
                lambda: process_pdf_in_thread(pdf_path, chunk_size, chunk_overlap))
        
        # Erfolgsmeldung und Cache-Erstellung
        duration = time.time() - start_time
        if chunks:
            print(f"PDF verarbeitet: {os.path.basename(pdf_path)} - {len(chunks)} Chunks erstellt in {duration:.2f} Sekunden")
            os.makedirs(cache_dir, exist_ok=True)
            async with aiofiles.open(cache_file, 'wb') as f:
                await f.write(pickle.dumps(chunks))
        else:
            print(f"Keine Chunks erstellt für: {os.path.basename(pdf_path)}")
        
        return chunks
        
    except Exception as e:
        print(f"Fehler bei der Verarbeitung von {os.path.basename(pdf_path)}: {str(e)}")
        return []

async def process_pdf_with_ocr(pdf_path):
    """Verarbeitet ein PDF mit OCR wenn normales Einlesen fehlschlägt"""
    try:
        print(f"Versuche OCR für: {os.path.basename(pdf_path)}")
        
        # Temporäres Verzeichnis für Bilder
        with tempfile.TemporaryDirectory() as temp_dir:
            # Konvertiere PDF zu Bildern
            images = pdf2image.convert_from_path(pdf_path)
            
            full_text = ""
            for i, image in enumerate(images):
                # Speichere und verarbeite jedes Bild
                image_path = os.path.join(temp_dir, f'page_{i}.png')
                image.save(image_path)
                
                # OCR mit Tesseract
                text = pytesseract.image_to_string(image, lang='deu')
                if text:
                    full_text += f"\n{text}"
            
            if not full_text.strip():
                print(f"OCR konnte keinen Text extrahieren aus: {os.path.basename(pdf_path)}")
                return None
                
            return full_text
            
    except Exception as e:
        print(f"Fehler bei OCR-Verarbeitung von {os.path.basename(pdf_path)}: {str(e)}")
        return None

def process_pdf_in_thread(pdf_path, chunk_size=2048, chunk_overlap=200):
    """Thread-sichere PDF-Verarbeitung"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            # Extrahiere Text aus jeder Seite
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    text = re.sub(r'\s+', ' ', text)  # Bereinige Whitespace
                    text = text.strip()
                    full_text += f"\n{text}"
            
            # Verwende OCR falls kein Text gefunden wurde
            if not full_text.strip():
                print(f"Kein Text gefunden in {os.path.basename(pdf_path)}, versuche OCR...")
                loop = asyncio.new_event_loop()
                full_text = loop.run_until_complete(process_pdf_with_ocr(pdf_path))
                loop.close()
                
                if not full_text:
                    return []
            
            # Teile Text in Chunks
            words = full_text.split()
            chunks = []
            
            for i in range(0, len(words), chunk_size - chunk_overlap):
                chunk_words = words[i:i + chunk_size]
                if chunk_words:
                    chunk_text = ' '.join(chunk_words)
                    chunks.append({
                        'content': chunk_text,
                        'metadata': {
                            'source': pdf_path,
                            'chunk_number': len(chunks) + 1,
                            'total_chunks': (len(words) + chunk_size - 1) // chunk_size
                        }
                    })
            
            print(f"Created {len(chunks)} chunks from {pdf_path}")
            return chunks
            
    except Exception as e:
        print(f"Fehler bei der Verarbeitung von {os.path.basename(pdf_path)}: {str(e)}")
        return []

def print_gpu_utilization():
    """Zeigt aktuelle GPU-Auslastung."""
    if torch.cuda.is_available():
        print(f"GPU-Speicher belegt: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
        print(f"GPU-Speicher reserviert: {torch.cuda.memory_reserved()/1024**3:.1f}GB")

async def generate_embeddings_async(chunks, tokenizer, model, batch_size=8):
    """Optimierte Embedding-Generierung."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    all_embeddings = []
    total_chunks = len(chunks)
    
    print(f"\nGeneriere Embeddings für {total_chunks} Chunks...")
    start_time = time.time()
    
    for i in range(0, len(chunks), batch_size):
        try:
            batch = chunks[i:i + batch_size]
            # Stelle sicher, dass wir die 'content' Felder korrekt extrahieren
            texts = []
            for chunk in batch:
                if isinstance(chunk, dict) and 'content' in chunk:
                    texts.append(chunk['content'])
                elif isinstance(chunk, str):
                    texts.append(chunk)
                else:
                    print(f"Warnung: Ungültiges Chunk-Format: {type(chunk)}")
                    continue
            
            if not texts:
                continue
                
            inputs = tokenizer(texts, return_tensors="pt", padding=True, 
                             truncation=True, max_length=512).to(device)
            
            with torch.no_grad(), torch.amp.autocast('cuda'):
                outputs = model(**inputs).last_hidden_state.mean(dim=1)
                all_embeddings.extend(outputs.cpu().numpy())
            
            if i % (batch_size * 10) == 0:
                progress = (i + len(batch)) / total_chunks * 100
                elapsed = time.time() - start_time
                chunks_per_second = (i + len(batch)) / elapsed
                remaining = (total_chunks - (i + len(batch))) / chunks_per_second
                print(f"\rFortschritt: {progress:.1f}% - "
                      f"Chunks/s: {chunks_per_second:.1f} - "
                      f"Verbleibend: {remaining/60:.1f}min", end="")
            
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                batch_size = max(1, batch_size // 2)
                print(f"\nReduziere Batch-Größe auf {batch_size}")
                continue
            raise e
    
    total_time = time.time() - start_time
    print(f"\nEmbedding-Generierung abgeschlossen in {total_time:.1f}s "
          f"({total_chunks/total_time:.1f} Chunks/s)")
    
    return all_embeddings

def calculate_optimal_batch_size(total_files, available_memory_gb=32): ## RAM Anpassen
    """
    Berechnet die optimale Batch-Größe basierend auf verfügbarem RAM.
    
    Args:
        total_files: Gesamtanzahl der zu verarbeitenden PDFs
        available_memory_gb: Verfügbarer RAM in GB (default: 32)
    """
    # Verfügbare CPU-Kerne
    cpu_cores = multiprocessing.cpu_count()
    num_processes = max(1, cpu_cores - 1)  # Ein Kern frei lassen
    
    # Bei 32GB RAM:
    # - System & andere Prozesse: ~8GB
    # - Python & Bibliotheken: ~4GB
    # - Verfügbar für PDFs: ~20GB
    usable_memory_gb = available_memory_gb * 0.6  # 60% des RAMs für PDFs
    
    # Schätzung: Ein PDF benötigt durchschnittlich 50MB RAM
    estimated_pdf_memory_mb = 50
    max_pdfs_by_memory = int((usable_memory_gb * 1024) / estimated_pdf_memory_mb)
    
    # Berechne optimale Batch-Größe
    suggested_batch_size = max(100, total_files // (num_processes * 2))
    batch_size = min(max_pdfs_by_memory, suggested_batch_size)
    
    # Für 32GB RAM empfohlene Maximalgrößen
    if batch_size > 200:  # Limitiere auf 200 PDFs pro Batch
        batch_size = 200
    
    print(f"RAM-Konfiguration:")
    print(f"  - Gesamt RAM: {available_memory_gb}GB")
    print(f"  - Nutzbar für PDFs: {usable_memory_gb:.1f}GB")
    print(f"  - Maximale PDFs im Speicher: {max_pdfs_by_memory}")
    
    return int(batch_size), num_processes

async def add_documents_to_db_async(chunks, tokenizer, model, conn, batch_size=None):
    if batch_size is None:
        batch_size = calculate_optimal_batch_size(len(chunks))[0]
    print(f"Verwende GPU Batch-Größe: {batch_size}")
    """Optimierte Datenbankoperationen mit Bulk-Insert."""
    embeddings = await generate_embeddings_async(chunks, tokenizer, model, batch_size)
    
    try:
        # Bereite Bulk-Insert vor
        for chunk, embedding in zip(chunks, embeddings):
            doc_id = f"doc_{uuid.uuid4().hex}"
            embedding_blob = embedding.astype('float32').tobytes()
            metadata = {
                **chunk['metadata'],
                'processed_date': datetime.now().isoformat()
            }
            
            # Einzelner Insert statt Bulk
            conn.execute('''
                INSERT INTO documents (id, content, embedding, metadata, file_path)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                doc_id,
                chunk['content'],
                embedding_blob,
                json.dumps(metadata),
                chunk['metadata']['source']
            ))
            
    except Exception as e:
        print(f"Fehler beim Datenbank-Insert: {str(e)}")
        raise e

def insert_batch_to_db(chunks, embeddings, conn):
    """
    Thread-sichere Batch-Insertion in die Datenbank.
    """
    for chunk, embedding in zip(chunks, embeddings):
        doc_id = f"doc_{uuid.uuid4().hex}"
        embedding_blob = embedding.astype('float32').tobytes()
        metadata = {
            **chunk['metadata'],
            'processed_date': datetime.now().isoformat()
        }
        
        conn.execute('''
            INSERT INTO documents (id, content, embedding, metadata, file_path) 
            VALUES (?, ?, ?, ?, ?)
        ''', (
            doc_id,
            chunk['content'],
            embedding_blob,
            json.dumps(metadata),
            chunk['metadata']['source']
        ))

def load_and_split_pdfs(file_path=None, pdf_directory=None, chunk_size=1000, chunk_overlap=200):
    """
    Lädt und verarbeitet PDFs entweder aus einem einzelnen file_path oder einem ganzen Verzeichnis.
    
    Args:
        file_path (str, optional): Pfad zu einer einzelnen PDF-Datei
        pdf_directory (str, optional): Pfad zum PDF-Verzeichnis
        chunk_size (int): Größe der Textabschnitte
        chunk_overlap (int): Überlappung zwischen Abschnitten
    """
    if file_path:
        # Verarbeite eine einzelne PDF
        if not os.path.exists(file_path):
            print(f"PDF nicht gefunden: {file_path}")
            return []
        # Verwende die Thread-basierte Verarbeitung
        return process_pdf_in_thread(file_path, chunk_size, chunk_overlap)
    
    elif pdf_directory:
        # Verarbeite alle PDFs in einem Verzeichnis
        all_chunks = []
        for root, _, files in os.walk(pdf_directory):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_path = os.path.join(root, file)
                    chunks = process_pdf_in_thread(pdf_path, chunk_size, chunk_overlap)
                    if chunks:
                        all_chunks.extend(chunks)
        return all_chunks
    
    return []

def init_database(db_name):
    """Initialisiert eine neue Datenbank mit der korrekten Struktur."""
    db_path = get_database_path(db_name)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding BLOB,
                metadata TEXT,
                file_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    c.execute('CREATE INDEX IF NOT EXISTS idx_filepath ON documents(file_path)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON documents(created_at)')
    
    conn.commit()
    conn.close()

def process_pdf(file_path):
    # Existierender Code...
    
    # Empfohlene Verbesserungen:
    try:
        text_chunks = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    # Bessere Textvorverarbeitung
                    text = re.sub(r'\s+', ' ', text)
                    text = text.strip()
                    
                    # Intelligentere Chunk-Erstellung
                    chunks = split_into_chunks(text, chunk_size=1000, overlap=100)
                    text_chunks.extend(chunks)
                    
        return text_chunks
    except Exception as e:
        logging.error(f"PDF Verarbeitungsfehler: {str(e)}")
        return []

def split_into_chunks(text, chunk_size=1000, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def add_documents_to_db(chunks, tokenizer, model, conn, batch_size=8):
    """
    Synchrone Version mit optimierter Speichernutzung.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    total_chunks = len(chunks)
    
    # Reduzierte Batch-Größe für GPU
    gpu_batch_size = 4
    
    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i:i + batch_size]
        texts = [chunk['content'] for chunk in batch_chunks]
        batch_embeddings = []
        
        # Verarbeite in kleineren Sub-Batches für GPU
        for j in range(0, len(texts), gpu_batch_size):
            sub_batch = texts[j:j + gpu_batch_size]
            
            try:
                inputs = tokenizer(sub_batch, return_tensors="pt", padding=True, 
                                 truncation=True, max_length=512).to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs).last_hidden_state.mean(dim=1)
                    sub_embeddings = outputs.cpu().numpy()
                    batch_embeddings.extend(sub_embeddings)
                    
                    # Explizit GPU-Speicher freigeben
                    del inputs, outputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"GPU OOM, versuche mit kleinerer Batch-Größe: {str(e)}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # Reduziere Batch-Größe weiter und versuche erneut
                    gpu_batch_size = max(1, gpu_batch_size // 2)
                    continue
                raise e
        
        # Insert chunks and embeddings into database
        insert_batch_to_db(batch_chunks, batch_embeddings, conn)
        
        print(f"Processed {min(i + batch_size, total_chunks)}/{total_chunks} chunks...")

async def process_pdf_streaming(pdf_path, chunk_size=1000, chunk_overlap=200):
    """Streaming-basierte PDF-Verarbeitung."""
    chunks = []
    buffer = ""
    
    async with aiofiles.open(pdf_path, 'rb') as file:
        pdf = pdfplumber.open(BytesIO(await file.read()))
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                buffer += text
                while len(buffer.split()) >= chunk_size:
                    words = buffer.split()
                    chunk_words = words[:chunk_size]
                    buffer = " ".join(words[chunk_size-chunk_overlap:])
                    chunks.append({
                        'content': " ".join(chunk_words),
                        'metadata': {'source': pdf_path}
                    })

class EmbeddingConfig:
    def __init__(self):
        self.use_mixed_precision = True
        self.precision_threshold = 0.8
        self.accumulation_steps = 4
        self.quality_check = True

    def adjust_for_quality(self):
        """Passt Einstellungen für höhere Qualität an."""
        self.use_mixed_precision = False
        self.accumulation_steps = 1

def process_pdf_with_params(args):
    """Hilfsfunktion für die Pool.map Verarbeitung"""
    pdf_path, chunk_size, chunk_overlap = args
    return process_pdf_in_thread(pdf_path, chunk_size, chunk_overlap)

async def process_batch_pdfs(batch, process_pool, chunk_size, chunk_overlap):
    """Verarbeitet einen Batch von PDFs parallel."""
    # Erstelle eine Liste von Argumenten für jede PDF
    args = [(pdf, chunk_size, chunk_overlap) for pdf in batch]
    
    # Verwende Pool.map mit der Hilfsfunktion
    results = process_pool.map(process_pdf_with_params, args)
    
    # Konvertiere die Ergebnisse in das gewünschte Format
    return [(pdf, chunks) for pdf, chunks in zip(batch, results)]

async def process_embeddings_batch(chunks_list, tokenizer, model, conn, batch_size):
    """Verarbeitet Embeddings für mehrere PDFs zusammen."""
    if not chunks_list:
        return
    
    print(f"\nVerarbeite {len(chunks_list)} Chunks...")
    
    try:
        # Generiere Embeddings
        embeddings = await generate_embeddings_async(chunks_list, tokenizer, model, batch_size)
        
        # Bereite Batch-Insert vor
        cursor = conn.cursor()
        for chunk, embedding in zip(chunks_list, embeddings):
            doc_id = chunk.get('id', f"doc_{uuid.uuid4().hex}")
            embedding_blob = embedding.astype('float32').tobytes()
            metadata = json.dumps(chunk.get('metadata', {}))
            file_path = chunk.get('file_path', '')
            content = chunk.get('content', '')
            
            if not all([doc_id, embedding_blob, content, file_path]):
                print(f"Warnung: Unvollständige Daten für Chunk {doc_id}")
                continue
                
            try:
                cursor.execute('''
                    INSERT INTO documents (id, content, embedding, metadata, file_path)
                    VALUES (?, ?, ?, ?, ?)
                ''', (doc_id, content, embedding_blob, metadata, file_path))
                
            except sqlite3.Error as e:
                print(f"Fehler beim Einfügen von Chunk {doc_id}: {str(e)}")
                continue
        
        conn.commit()
        print(f"\nErfolgreich {cursor.rowcount} Chunks in die Datenbank eingefügt")
        
    except Exception as e:
        print(f"Fehler beim Verarbeiten der Chunks: {str(e)}")
        conn.rollback()
        raise e

def get_available_databases():
    """Gibt eine Liste aller verfügbaren Datenbanken zurück."""
    db_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'databases')
    os.makedirs(db_dir, exist_ok=True)
    databases = []
    for file in os.listdir(db_dir):
        if file.endswith('.db'):
            name = os.path.splitext(file)[0]
            databases.append({
                'name': name,
                'path': os.path.join(db_dir, file),
                'size': os.path.getsize(os.path.join(db_dir, file)) / (1024 * 1024)  # Size in MB
            })
    return databases