"""
PDF-Chatbot Server-Konfiguration
--------------------------------

Dieses Skript ist der Einstiegspunkt für den Flask-Server des PDF-Chatbots.
Es konfiguriert die grundlegenden Server-Einstellungen für den Produktivbetrieb:

- Sicherheitseinstellungen (Secret Key für Sessions)
- Session-Verwaltung (Lebensdauer und Persistenz)
- Datenbank-Connection-Pool für optimale Performance
- Server-Konfiguration (Port, Threading, externe Zugriffe)

Der Server wird im Produktionsmodus gestartet und ist für externe Verbindungen 
über Port 5000 zugänglich.

"""

from app import create_app
import secrets
from datetime import timedelta

# Erstelle Flask-Anwendung
app = create_app()

# Sicherheitseinstellungen
app.secret_key = secrets.token_hex(32)  # Generiere sicheren, zufälligen Schlüssel

# Session-Konfiguration
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)  # Session läuft nach einem Tag ab
app.config['SESSION_PERMANENT'] = True  # Sessions bleiben über Browser-Neustart erhalten

# Datenbank-Pool-Konfiguration für bessere Performance und Stabilität
app.config['SQLALCHEMY_POOL_SIZE'] = 20          # Maximale Anzahl gleichzeitiger Datenbankverbindungen
app.config['SQLALCHEMY_POOL_TIMEOUT'] = 30       # Timeout für Verbindungsversuche in Sekunden
app.config['SQLALCHEMY_MAX_OVERFLOW'] = 5        # Zusätzliche Verbindungen bei hoher Last
app.config['SQLALCHEMY_POOL_RECYCLE'] = 1800     # Verbindungen nach 30 Minuten erneuern

# Server-Start-Konfiguration
if __name__ == "__main__":
    app.run(
        debug=False,          # Produktionsmodus
        host='0.0.0.0',      # Erlaube externe Verbindungen
        port=5000,           # Standard-Port
        threaded=True        # Aktiviere Multi-Threading
    )