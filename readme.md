
## Erste Schritte
1. Huggingface Pro-Account erstellen und API-Key in .env einfügen

2. PDF-Datenbank erstellen:
   - Passen sie den Dateipfad des Ordners den sie verarbeiten wollen im Skript "vectorize_folder.py" unter FOLDER_PATH an.
   - Führen Sie das Skript aus: vectorize_folder.py

3. Server starten: python wsgi.py

4. Öffnen Sie den Browser unter `http://localhost:5000`

## Projektstruktur

- `app/` - Hauptanwendungsverzeichnis
  - `__init__.py` - Flask-App-Initialisierung
  - `routes.py` - Web-Routen und API-Endpunkte
  - `chatbot.py` - Chatbot-Kernfunktionalität
  - `utils.py` - Hilfsfunktionen
  - `templates/` - HTML-Templates
- `databases/` - SQLite-Datenbanken
- `vectorize_folder.py` - Skript zur PDF-Verarbeitung
- `wsgi.py` - Server-Konfiguration

## Nutzung

1. **PDF-Verarbeitung**:
   - FOLDER_PATH im Skript "vectorize_folder.py" anpassen
   - `vectorize_folder.py` ausführen
   - Verarbeitungsbericht wird erstellt
   Hinweise: 
   - Ein Ordner sollte nicht mehr als 1000 Dateien enthalten.
   - Dieser Prozess kann einige Zeit in Anspruch nehmen, je nach Größe der PDF-Dateien und der Leistung des Systems.

2. **Chatbot-Nutzung**:
   - Server starten
   - Im Browser öffnen
   - Datenbank aus Dropdown wählen
   - Fragen eingeben
   - Auf PDF-Quellen klicken für Details

## Fehlerbehebung

- **OCR-Fehler**: Prüfen Sie die Tesseract-Installation und den Pfad in `.env`
- **Speicherprobleme**: Batch-Größe in `vectorize_folder.py` anpassen
- **GPU-Nutzung**: CUDA-Installation und PyTorch-Version prüfen


## Support

Bei Fragen oder Problemen:
- Verarbeitungsberichte analysieren
- System-Ressourcen überwachen
- debug_vectorstore ausführen

# Weiteres

- RAM Anpassen in calculate_optimal_batch_size in utils.py
- HUGGINGFACEAPI KEY in .env einfügen

