# config.py
import os
from datetime import datetime
from pathlib import Path

# === Configuration API ===
API_KEY = "3fdb5f73f24ccda1ee902717c594a85d"  # À remplacer par votre vraie clé

# === Villes marocaines ===
CITIES = {
    "Casablanca": {"lat": 33.5731, "lon": -7.5898},
    "Rabat": {"lat": 34.0209, "lon": -6.8416},
    "Marrakech": {"lat": 31.6295, "lon": -7.9811},
    "Fes": {"lat": 34.0181, "lon": -5.0078},
    "Tanger": {"lat": 35.7595, "lon": -5.8340},  # Optionnel
    "Agadir": {"lat": 30.4278, "lon": -9.5981},  # Optionnel
}

# === Paramètres de collecte ===
INTERVAL_SECONDS = 3600  # 1 heure entre chaque collecte
MAX_REQUESTS = 168  # Nombre total de collectes (168 = 7 jours complets)
REQUEST_TIMEOUT = 30  # Timeout en secondes pour les requêtes API

# === Structure des dossiers ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "logs"

# Création automatique des dossiers
for dir_path in [RAW_DIR, CHECKPOINT_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# === Fichiers ===
CHECKPOINT_FILE = CHECKPOINT_DIR / "collection_progress.json"
LOG_FILE = LOG_DIR / f"collection_{datetime.now().strftime('%Y%m%d')}.log"

# === Limites API (pour éviter le blocage) ===
RATE_LIMIT_DELAY = 2  # Secondes entre chaque requête (limite: 60/min max)
MAX_RETRIES = 3  # Nombre de tentatives en cas d'échec
RETRY_DELAY = 10  # Secondes avant de réessayer