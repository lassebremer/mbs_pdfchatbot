"""
PDF-Chatbot Flask-Anwendung Initialisierung
------------------------------------------

Dieses Modul initialisiert die Flask-Anwendung für den PDF-Chatbot.
Es konfiguriert:

- Die grundlegende Flask-Anwendungsstruktur
- Cross-Origin Resource Sharing (CORS)
- Umgebungsvariablen aus .env
- Session-Sicherheit
- Blueprint-Registrierung für die Routen

Die Anwendung wird als Factory-Pattern implementiert, was Flexibilität
bei Tests und verschiedenen Konfigurationen ermöglicht.
"""
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
import os

def create_app():
    # Erstelle eine neue Flask-Anwendung
    app = Flask(__name__)
    
    # Aktiviere Cross-Origin Resource Sharing (CORS)
    # Dies erlaubt Anfragen von anderen Domains
    CORS(app)
    
    # Lade Umgebungsvariablen aus der .env-Datei
    load_dotenv()
    
    # Setze den Secret Key für die Session-Verschlüsselung
    # Verwende entweder den Wert aus den Umgebungsvariablen oder generiere einen zufälligen Key
    app.secret_key = os.getenv('FLASK_SECRET_KEY') or os.urandom(24)
    
    # Importiere und registriere die Routen aus der routes.py
    from .routes import main
    app.register_blueprint(main)

    return app