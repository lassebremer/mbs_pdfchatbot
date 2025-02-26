"""
PDF-Chatbot Datenbank-Initialisierung
------------------------------------

Dieses Skript verarbeitet einen Ordner mit PDF-Dateien und erstellt
eine durchsuchbare Vektordatenbank:

- Rekursives Scannen von PDF-Ordnern
- Parallele Verarbeitung von PDF-Dokumenten
- Textextraktion und Chunk-Erstellung
- Generierung von Embeddings
- Speicherung in SQLite-Datenbank
- Fortschrittsüberwachung und Berichtserstellung


- Pfad für report_file anpassen
Muss vor der ersten Nutzung des Chatbots ausgeführt werden.
""" 
# Konfiguration des Eingabeordners

FOLDER_PATH = r"data_pdf\Support\_Messen"  # Passe diesen Pfad an

import os
import time
import asyncio
import multiprocessing
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
from app.utils import (
    get_database_path,
    process_batch_pdfs,
    process_embeddings_batch,
    calculate_optimal_batch_size
)
import sqlite3



def create_vector_database(db_name):
    """Erstellt und initialisiert die Vektordatenbank
    
    Args:
        db_name (str): Name der zu erstellenden Datenbank
    """
    db_path = get_database_path(db_name)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Lösche existierende Tabellen für Neustart
    c.execute('DROP TABLE IF EXISTS documents')
    
    # Erstelle Dokumententabelle mit TEXT ID
    c.execute('''CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding BLOB,
                    metadata TEXT,
                    file_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Erstelle Indizes für bessere Performance
    c.execute('CREATE INDEX IF NOT EXISTS idx_filepath ON documents(file_path)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON documents(created_at)')
    
    conn.commit()
    conn.close()
    print(f"Vektordatenbank '{db_name}' erfolgreich erstellt!")

async def process_folder(folder_path):
    """Verarbeitet einen Ordner mit PDFs und erstellt eine Vektordatenbank
    
    Args:
        folder_path (str): Pfad zum Ordner mit den PDF-Dateien
        
    Ablauf:
    1. Validierung des Eingabeordners
    2. Erstellen der Datenbank
    3. Rekursives Sammeln aller PDF-Dateien
    4. Laden der KI-Modelle
    5. Parallele Verarbeitung in Batches
    6. Speichern der Embeddings
    7. Erstellung eines Abschlussberichts
    """
    start_time = time.time()
    
    # Validiere den Eingabeordner
    folder_path = os.path.abspath(folder_path)
    if not os.path.exists(folder_path):
        raise ValueError(f"Ordner nicht gefunden: {folder_path}")
    
    # Erstelle Datenbank mit Ordnernamen
    db_name = os.path.basename(folder_path)
    print(f"\nVerarbeite Ordner: {folder_path}")
    print(f"Erstelle Datenbank: {db_name}")
    
    create_vector_database(db_name)
    
    # Sammle alle PDF-Dateien rekursiv
    pdf_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    if not pdf_files:
        print("Keine PDF-Dateien im angegebenen Ordner gefunden.")
        return
    
    total_files = len(pdf_files)
    print(f"\nGefundene PDF-Dateien: {total_files}")
    
    # Initialisiere KI-Modelle
    load_dotenv()
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    if not EMBEDDING_MODEL:
        raise ValueError("EMBEDDING_MODEL nicht in .env konfiguriert")
    
    print("\nLade Modelle...")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL)
    
    # Optimiere Verarbeitungsparameter
    batch_size, num_processes = calculate_optimal_batch_size(total_files, available_memory_gb=16)
    print(f"\nVerwendete Batch-Größe: {batch_size}")
    print(f"Verwendete Prozesse: {num_processes}")
    
    # Verarbeitungsstatistiken
    processed_files = 0
    failed_files = 0
    total_chunks = 0
    failed_pdfs = []
    
    try:
        # Parallele Verarbeitung mit Multiprocessing
        with multiprocessing.Pool(num_processes) as process_pool:
            for i in range(0, len(pdf_files), batch_size):
                batch = pdf_files[i:i + batch_size]
                current_batch = i // batch_size + 1
                total_batches = (total_files + batch_size - 1) // batch_size
                
                print(f"\nVerarbeite Batch {current_batch}/{total_batches}")
                
                # Verarbeite PDFs parallel
                results = await process_batch_pdfs(batch, process_pool, 
                                                 int(os.getenv("CHUNK_SIZE", 1000)),
                                                 int(os.getenv("CHUNK_OVERLAP", 200)))
                
                # Verarbeite erfolgreiche Ergebnisse
                successful_chunks = []
                for pdf_path, chunks in results:
                    if chunks:
                        for chunk in chunks:
                            if isinstance(chunk, dict):
                                chunk_id = f"{os.path.basename(pdf_path)}_{len(successful_chunks)}"
                                chunk.update({
                                    'id': chunk_id,
                                    'file_path': pdf_path,
                                    'metadata': {
                                        'source': pdf_path,
                                        'chunk_number': len(successful_chunks) + 1
                                    }
                                })
                                successful_chunks.append(chunk)
                        processed_files += 1
                        total_chunks += len(chunks)
                    else:
                        failed_files += 1
                        failed_pdfs.append(pdf_path)
                
                # Speichere Embeddings in Datenbank
                if successful_chunks:
                    print(f"\nSpeichere {len(successful_chunks)} Chunks in Datenbank...")
                    verify_chunks_structure(successful_chunks)
                    
                    conn = sqlite3.connect(get_database_path(db_name))
                    try:
                        await process_embeddings_batch(successful_chunks, tokenizer, model, 
                                                     conn, int(os.getenv("BATCH_SIZE", 32)))
                        conn.commit()  # Wichtig: Commit nach jedem Batch!
                    except Exception as e:
                        print(f"Fehler beim Speichern in DB: {str(e)}")
                        raise
                    finally:
                        conn.close()
                
                # Zeige Fortschritt und Statistiken
                progress = (i + len(batch)) / total_files * 100
                elapsed_time = time.time() - start_time
                speed = (i + len(batch)) / elapsed_time
                remaining = (total_files - (i + len(batch))) / speed if speed > 0 else 0
                
                print(f"\rFortschritt: {progress:.1f}% - "
                      f"Verarbeitet: {processed_files}/{total_files} - "
                      f"Fehler: {failed_files} - "
                      f"PDFs/s: {speed:.2f} - "
                      f"Verbleibend: {remaining/60:.1f}min")
    
    except Exception as e:
        print(f"\nFehler während der Verarbeitung: {str(e)}")
        raise
    
    finally:
        # Erstelle Abschlussbericht
        total_duration = time.time() - start_time
        report_data = {
            'total_duration': total_duration,
            'total_files': total_files,
            'processed_files': processed_files,
            'failed_files': failed_files,
            'total_chunks': total_chunks,
            'failed_pdfs': failed_pdfs
        }
        
        save_report(report_data, start_time)

        # Überprüfe Datenbank
        if verify_database_content(db_name):
            print("\nDatenbank wurde erfolgreich erstellt und gefüllt!")
        else:
            print("\nWarnung: Datenbank scheint leer zu sein!")

def save_report(report_data, start_time):
    """Speichert einen detaillierten Verarbeitungsbericht
    
    Args:
        report_data (dict): Dictionary mit Verarbeitungsstatistiken
        start_time (float): Zeitstempel des Verarbeitungsbeginns
        
    Gespeicherte Informationen:
    - Start- und Endzeit
    - Gesamtdauer
    - Anzahl verarbeiteter Dateien
    - Fehlerstatistiken
    - Liste fehlgeschlagener PDFs
    """
    timestamp = datetime.fromtimestamp(start_time).strftime('%Y%m%d_%H%M%S')
    report_file = f"processing_report_{timestamp}.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*50 + "\n")
        f.write("VERARBEITUNGSBERICHT\n")
        f.write("="*50 + "\n")
        f.write(f"Startzeit: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Endzeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Gesamtzeit: {report_data['total_duration']:.2f} Sekunden "
                f"({report_data['total_duration']/60:.2f} Minuten)\n")
        f.write(f"Gefundene PDF-Dateien: {report_data['total_files']}\n")
        f.write(f"Erfolgreich verarbeitet: {report_data['processed_files']}\n")
        f.write(f"Fehlgeschlagen: {report_data['failed_files']}\n")
        f.write(f"Generierte Chunks insgesamt: {report_data['total_chunks']}\n")
        
        if report_data['processed_files'] > 0:
            f.write(f"Durchschnittliche Zeit pro Datei: "
                    f"{report_data['total_duration']/report_data['processed_files']:.2f} Sekunden\n")
            f.write(f"Durchschnittliche Chunks pro Datei: "
                    f"{report_data['total_chunks']/report_data['processed_files']:.1f}\n")
        
        if report_data['failed_pdfs']:
            f.write("\nFehlgeschlagene PDFs:\n")
            for pdf in report_data['failed_pdfs']:
                f.write(f"- {pdf}\n")
        
        f.write("="*50 + "\n")
    
    print(f"\nBericht wurde gespeichert in: {report_file}")

def verify_chunks_structure(chunks):
    """Überprüft die Struktur und Vollständigkeit der Chunks
    
    Args:
        chunks (list): Liste von Chunk-Dictionaries
        
    Prüft:
    - Korrekte Dictionary-Struktur
    - Vorhandensein aller Pflichtfelder
    - Nicht-leere Werte in Pflichtfeldern
    """
    required_fields = ['id', 'content', 'file_path']
    
    print(f"\nÜberprüfe {len(chunks)} Chunks...")
    
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            print(f"FEHLER: Chunk {i} ist kein Dictionary!")
            continue
            
        missing_fields = [field for field in required_fields if field not in chunk]
        if missing_fields:
            print(f"FEHLER: Chunk {i} fehlen folgende Felder: {missing_fields}")
            continue
            
        if not chunk['id'] or not chunk['content'] or not chunk['file_path']:
            print(f"FEHLER: Chunk {i} hat leere Pflichtfelder!")
            continue
            
    print("Chunk-Struktur-Überprüfung abgeschlossen")

def verify_database_content(db_name):
    """Überprüft den Inhalt und die Struktur der erstellten Datenbank
    
    Args:
        db_name (str): Name der zu prüfenden Datenbank
        
    Returns:
        bool: True wenn Datenbank Einträge enthält, sonst False
        
    Prüft:
    - Tabellenstruktur
    - Anzahl der Einträge
    - Beispieleinträge (erste 3)
    """
    try:
        db_path = get_database_path(db_name)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Prüfe Tabellenstruktur
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='documents'")
        table_info = cursor.fetchone()
        print(f"\nTabellenstruktur:\n{table_info[0] if table_info else 'Keine Tabelle gefunden!'}")
        
        # Prüfe Datenbankinhalt
        cursor.execute("SELECT COUNT(*) FROM documents")
        count = cursor.fetchone()[0]
        print(f"\nAnzahl Dokumente in der Datenbank: {count}")
        
        # Zeige Beispieleinträge
        if count > 0:
            cursor.execute("""
                SELECT id, substr(content, 1, 100), length(embedding), file_path 
                FROM documents LIMIT 3
            """)
            print("\nBeispieleinträge:")
            for row in cursor.fetchall():
                print(f"ID: {row[0]}")
                print(f"Content: {row[1]}...")
                print(f"Embedding Größe: {row[2]} Bytes")
                print(f"Dateipfad: {row[3]}\n")
        
        conn.close()
        return count > 0
        
    except Exception as e:
        print(f"Fehler bei der Datenbankprüfung: {str(e)}")
        return False

if __name__ == "__main__":
    # Starte die Verarbeitung mit dem konfigurierten Ordnerpfad
    asyncio.run(process_folder(FOLDER_PATH))

