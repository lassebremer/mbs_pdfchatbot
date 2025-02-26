import sqlite3
import numpy as np
import json
import os

def analyze_vectorstore():
    # Überprüfen Sie zuerst, ob die Datei existiert
    db_path = 'databases/_Anleitungen.db'
    print(f"\n=== Datenbankstatus ===")
    print(f"Datenbank Pfad: {os.path.abspath(db_path)}")
    print(f"Datenbank existiert: {os.path.exists(db_path)}")
    print(f"Dateigröße: {os.path.getsize(db_path) if os.path.exists(db_path) else 'N/A'} Bytes")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Überprüfen Sie die Anzahl der Einträge in allen Tabellen
        print("\n=== Tabellenstatistiken ===")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cursor.fetchone()[0]
            print(f"Tabelle '{table_name}': {count} Einträge")
        
        # Tabellenstruktur anzeigen
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
        print("\n=== Tabellenstruktur ===")
        for table in cursor.fetchall():
            print(table[0])
        
        # Metadaten analysieren
        print("\n=== Metadaten Analyse ===")
        cursor.execute("SELECT id, metadata FROM documents LIMIT 5;")
        for doc_id, metadata in cursor.fetchall():
            print(f"\nDokument ID: {doc_id}")
            try:
                if metadata:
                    meta_dict = json.loads(metadata)
                    print(f"Metadata: {json.dumps(meta_dict, indent=2, ensure_ascii=False)}")
                else:
                    print("Keine Metadaten")
            except json.JSONDecodeError:
                print(f"Ungültige JSON Metadaten: {metadata}")
        
        # Detaillierte Inhaltsanalyse
        cursor.execute("SELECT id, content, file_path FROM documents LIMIT 3;")
        print("\n=== Detaillierte Inhaltsanalyse ===")
        for doc_id, content, file_path in cursor.fetchall():
            print(f"\nDokument ID: {doc_id}")
            print(f"Dateipfad: {file_path}")
            print(f"Encoding Info:")
            print(f"- UTF-8: {content.encode('utf-8')[:50]}")
            print(f"- ASCII: {content.encode('ascii', 'ignore')[:50]}")
            print(f"- Reverse Test: {content[:50][::-1]}")  # Test auf umgekehrten Text
            print(f"Erste 100 Zeichen: {content[:100]}")
            print(f"Letzte 100 Zeichen: {content[-100:]}")
            
            # Zeichenanalyse
            special_chars = set(char for char in content if not char.isalnum() and not char.isspace())
            print(f"Spezielle Zeichen: {special_chars}")
        
        # Statistiken über Inhaltslängen
        cursor.execute("SELECT LENGTH(content) as len FROM documents;")
        lengths = [row[0] for row in cursor.fetchall()]
        if lengths:
            print("\n=== Inhaltsstatistiken ===")
            print(f"Minimale Länge: {min(lengths)}")
            print(f"Maximale Länge: {max(lengths)}")
            print(f"Durchschnittliche Länge: {sum(lengths)/len(lengths):.2f}")
            
            # Verteilung der Längen
            print("\n=== Längenverteilung ===")
            ranges = [(0, 100), (101, 500), (501, 1000), (1001, 5000), (5001, float('inf'))]
            for start, end in ranges:
                count = len([l for l in lengths if start <= l <= end])
                print(f"{start}-{end if end != float('inf') else '∞'} Zeichen: {count} Dokumente")
        
    except sqlite3.Error as e:
        print(f"\nSQLite Fehler: {e}")
    except Exception as e:
        print(f"\nAllgemeiner Fehler: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

def inspect_document(doc_id):
    """Detaillierte Analyse eines einzelnen Dokuments"""
    db_path = 'databases/_Anleitungen.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
    doc = cursor.fetchone()
    
    if doc:
        print(f"\n=== Dokument {doc_id} ===")
        content = doc[1]  # content ist der zweite Wert
        print(f"Dateipfad: {doc[4]}")  # file_path ist der fünfte Wert
        print(f"Erstellungsdatum: {doc[5]}")  # created_at ist der sechste Wert
        
        # Encoding-Tests
        print("\nEncoding-Tests:")
        print(f"String Repräsentation: {repr(content[:100])}")
        print(f"UTF-8 Bytes: {content[:100].encode('utf-8')}")
        
        # Textanalyse
        words = content.split()
        print(f"\nWortanalyse:")
        print(f"Anzahl Wörter: {len(words)}")
        print(f"Durchschnittliche Wortlänge: {sum(len(w) for w in words)/len(words):.2f}")
        print(f"Erste 5 Wörter: {' '.join(words[:5])}")
        
        # Test auf umgekehrten Text
        print(f"\nTest auf umgekehrten Text:")
        print(f"Original: {content[:50]}")
        print(f"Rückwärts: {content[:50][::-1]}")
        
    else:
        print(f"Dokument {doc_id} nicht gefunden")
    
    conn.close()

def check_database_writes():
    """Überprüft die Schreibvorgänge in der Datenbank"""
    db_path = 'databases/_Anleitungen.db'
    
    print("\n=== Schreibtest ===")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Versuche einen Testdatensatz zu schreiben
        test_data = {
            'id': 'test_doc',
            'content': 'Test Inhalt',
            'metadata': json.dumps({'test': True}),
            'file_path': 'test/path.pdf'
        }
        
        cursor.execute('''
            INSERT OR REPLACE INTO documents 
            (id, content, metadata, file_path) 
            VALUES (?, ?, ?, ?)
        ''', (
            test_data['id'],
            test_data['content'],
            test_data['metadata'],
            test_data['file_path']
        ))
        
        conn.commit()
        print("Testdatensatz erfolgreich geschrieben")
        
        # Überprüfe ob der Datensatz gelesen werden kann
        cursor.execute("SELECT * FROM documents WHERE id = 'test_doc'")
        result = cursor.fetchone()
        if result:
            print("Testdatensatz erfolgreich gelesen")
            print(f"Inhalt: {result}")
        else:
            print("Testdatensatz konnte nicht gelesen werden!")
            
    except sqlite3.Error as e:
        print(f"SQLite Fehler: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    analyze_vectorstore()
    check_database_writes()  # Füge den Schreibtest hinzu
    # Um ein spezifisches Dokument zu analysieren:
    # inspect_document("ihr_dokument_id") 