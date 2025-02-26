from flask import Blueprint, request, jsonify, render_template, session
from .utils import get_available_databases, get_database_path
from .chatbot import VectorDBChatbot
from flask import Blueprint, send_file
import os
import sqlite3
import json
from dotenv import load_dotenv

# Erstelle Blueprint für die Routen
main = Blueprint('main', __name__)
chatbots = {}  # Speichere Chatbot-Instanzen für verschiedene Sessions

# Lade Umgebungsvariablen
load_dotenv()

"""
PDF-Chatbot Routen-Handler
--------------------------

Dieses Modul definiert alle Web-Routen des PDF-Chatbots:

- / : Hauptseite mit Chat-Interface
- /chat : API-Endpunkt für Chat-Anfragen
- /switch_database : Endpunkt zum Wechseln der aktiven Datenbank
- /pdf/<doc_id> : Endpunkt zum Herunterladen von PDF-Dokumenten
- /get_databases : API zum Abrufen verfügbarer Datenbanken

Verwaltet Session-basierte Chatbot-Instanzen und Datenbankverbindungen.
"""

@main.route('/')
def index():
    """Hauptseite der Anwendung"""
    # Hole verfügbare Datenbanken
    databases = get_available_databases()
    if not databases:
        return render_template('index.html', 
                             error="Keine Datenbanken verfügbar. Bitte zuerst Datenbank mit vectorize_folder.py erstellen.", 
                             databases=[], 
                             current_db=None)
    
    # Prüfe und setze aktuelle Datenbank
    current_db = session.get('current_db')
    if not current_db or current_db not in [db['name'] for db in databases]:
        # Setze die erste verfügbare Datenbank als aktuelle DB
        current_db = databases[0]['name']
        session['current_db'] = current_db
        
    return render_template('index.html', databases=databases, current_db=current_db)

@main.route('/switch_database', methods=['POST'])
def switch_database():
    """Wechselt zu einer anderen Datenbank"""
    try:
        # Hole Datenbanknamen aus der Anfrage
        db_name = request.json.get('database')
        if not db_name:
            return jsonify({"error": "Kein Datenbankname angegeben"}), 400
            
        # Prüfe ob Datenbank verfügbar ist
        available_dbs = [db['name'] for db in get_available_databases()]
        if db_name not in available_dbs:
            return jsonify({"error": f"Datenbank {db_name} nicht gefunden"}), 404
            
        # Verwalte Session-ID
        session_id = session.get('session_id')
        if not session_id:
            session_id = os.urandom(16).hex()
            session['session_id'] = session_id
        
        # Aktualisiere oder erstelle Chatbot-Instanz
        if session_id in chatbots:
            chatbots[session_id].switch_database(db_name)
        else:
            chatbots[session_id] = VectorDBChatbot(db_name)
            
        session['current_db'] = db_name
        
        return jsonify({
            "success": True, 
            "message": f"Datenbank zu {db_name} gewechselt",
            "reload": True
        })
        
    except Exception as e:
        print(f"Fehler beim Datenbankwechsel: {str(e)}")
        return jsonify({"error": f"Fehler beim Datenbankwechsel: {str(e)}"}), 500

@main.route('/chat', methods=['POST'])
def chat():
    """Verarbeitet Chat-Anfragen"""
    # Verwalte Session-ID
    session_id = session.get('session_id')
    if not session_id:
        session_id = os.urandom(16).hex()
        session['session_id'] = session_id
        
    # Erstelle bei Bedarf neue Chatbot-Instanz
    if session_id not in chatbots:
        current_db = session.get('current_db', 'default')
        chatbots[session_id] = VectorDBChatbot(current_db)
    
    # Prüfe Nachrichteninhalt
    message = request.json.get("message")
    if not message:
        return jsonify({"error": "Keine Nachricht übermittelt"}), 400

    try:
        # Hole Antwort vom Chatbot
        response, pdf_links = chatbots[session_id].get_response(message)
        return jsonify({"response": response, "pdfLinks": pdf_links}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@main.route('/get_databases', methods=['GET'])
def get_databases():
    """API-Endpunkt zum Abrufen der verfügbaren Datenbanken"""
    databases = get_available_databases()
    return jsonify(databases)

@main.route('/pdf/<doc_id>')
def serve_pdf(doc_id):
    """Stellt PDF-Dokumente zum Download bereit"""
    # Hole aktuelle Datenbank aus der Session
    current_db = session.get('current_db')
    if not current_db:
        return "Keine aktive Datenbank ausgewählt", 400
        
    # Verbinde zur Datenbank und hole Dateipfad
    db_path = get_database_path(current_db)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT file_path FROM documents WHERE id = ?', (doc_id,))
    result = c.fetchone()
    conn.close()

    if result:
        # Konstruiere den vollständigen Dateipfad
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        stored_path = result[0]
        relative_path = stored_path.split('data_pdf')[-1].lstrip('\\')
        file_path = os.path.join(base_dir, 'data_pdf', relative_path)
        file_path = os.path.normpath(file_path)
        
        # Debug-Ausgaben für Pfadkonstruktion
        print(f"Base directory: {base_dir}")
        print(f"Stored path: {stored_path}")
        print(f"Relative path: {relative_path}")
        print(f"Final path: {file_path}")
        
        # Sende PDF-Datei
        if os.path.exists(file_path):
            try:
                return send_file(file_path, mimetype='application/pdf')
            except Exception as e:
                print(f"Fehler beim Senden der Datei: {str(e)}")
                return f"Fehler beim Öffnen der PDF: {str(e)}", 500
        else:
            print(f"Datei nicht gefunden: {file_path}")
            return f"PDF nicht gefunden: {file_path}", 404
    else:
        return "Dokument-ID nicht gefunden", 404


