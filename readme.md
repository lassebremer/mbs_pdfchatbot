
## Erste Schritte

1. PDF-Datenbank erstellen:
   - Legen Sie Ihre PDFs im Ordner `data_pdf` ab
   - Passen Sie den `FOLDER_PATH` in `vectorize_folder.py` an
   - Führen Sie das Skript aus: vectorize_folder.py

2. Server starten: python wsgi.py
3. Öffnen Sie den Browser unter `http://localhost:5000`

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
   - PDFs im `data_pdf` Ordner ablegen
   - `vectorize_folder.py` ausführen
   - Verarbeitungsbericht wird erstellt

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

## Performance-Optimierung

- GPU-Beschleunigung aktivieren
- Batch-Größen anpassen
- Connection-Pool-Einstellungen in `wsgi.py` optimieren
- Cache-Nutzung für häufige Anfragen aktivieren


## Wartung

- Log-Dateien regelmäßig prüfen
- Datenbank-Indizes optimieren
- Cache regelmäßig bereinigen
- PDF-Verarbeitung überwachen

## Support

Bei Fragen oder Problemen:
- Verarbeitungsberichte analysieren
- System-Ressourcen überwachen
- debug_vectorstore ausführen

# Weiteres

- RAM Anpassen in calculate_optimal_batch_size in utils.py
- HUGGINGFACEAPI KEY in .env einfügen

