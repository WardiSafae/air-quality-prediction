# collector.py
import requests
import pandas as pd
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Import de la configuration
from config import *

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AirQualityCollector:
    """Collecteur professionnel avec gestion de reprise"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.progress = self.load_progress()
        self.data = self.progress.get("collected_data", [])
        self.last_request_time = 0
        
    def load_progress(self) -> Dict:
        """Charge la progression sauvegardée"""
        if CHECKPOINT_FILE.exists():
            try:
                with open(CHECKPOINT_FILE, 'r') as f:
                    progress = json.load(f)
                logger.info(f" Progression chargée : {progress['last_index']}/{progress.get('total_planned', '?')} collectes effectuées")
                return progress
            except Exception as e:
                logger.error(f" Erreur chargement progression : {e}")
                return self.init_progress()
        else:
            logger.info(" Aucune progression trouvée, démarrage depuis zéro")
            return self.init_progress()
    
    def init_progress(self) -> Dict:
        """Initialise une nouvelle progression"""
        return {
            "last_index": 0,
            "collected_data": [],
            "total_planned": MAX_REQUESTS,
            "start_time": datetime.now().isoformat(),
            "failed_attempts": [],
            "cities_completed": {city: 0 for city in CITIES.keys()}
        }
    
    def save_progress(self):
        """Sauvegarde la progression actuelle"""
        try:
            self.progress["collected_data"] = self.data
            self.progress["last_save"] = datetime.now().isoformat()
            # Sauvegarde temporaire avant écriture finale
            temp_file = CHECKPOINT_FILE.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(self.progress, f, indent=2, default=str)
            temp_file.replace(CHECKPOINT_FILE)
            logger.debug(f" Progression sauvegardée : {len(self.data)} données")
        except Exception as e:
            logger.error(f" Erreur sauvegarde progression : {e}")
    
    def rate_limit_wait(self):
        """Gère les limites de taux de l'API"""
        elapsed = time.time() - self.last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
    
    def fetch_city_data(self, city_name: str, lat: float, lon: float) -> Optional[Dict]:
        """Récupère les données météo et qualité de l'air pour une ville"""
        
        # Rate limiting
        self.rate_limit_wait()
        
        # URLs API
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={self.api_key}&units=metric"
        air_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={self.api_key}"
        
        # Tentatives avec retry
        for attempt in range(MAX_RETRIES):
            try:
                # Requête météo
                weather_response = self.session.get(weather_url, timeout=REQUEST_TIMEOUT)
                weather_response.raise_for_status()
                weather_data = weather_response.json()
                
                # Petite pause entre les deux requêtes
                time.sleep(1)
                
                # Requête qualité de l'air
                air_response = self.session.get(air_url, timeout=REQUEST_TIMEOUT)
                air_response.raise_for_status()
                air_data = air_response.json()
                
                self.last_request_time = time.time()
                
                # Construction de l'enregistrement
                record = {
                    "timestamp": datetime.now().isoformat(),
                    "city": city_name,
                    "lat": lat,
                    "lon": lon,
                    "temp": weather_data["main"]["temp"],
                    "feels_like": weather_data["main"]["feels_like"],
                    "humidity": weather_data["main"]["humidity"],
                    "pressure": weather_data["main"]["pressure"],
                    "wind_speed": weather_data["wind"]["speed"],
                    "wind_deg": weather_data["wind"].get("deg", 0),
                    "clouds": weather_data["clouds"]["all"],
                    "weather_main": weather_data["weather"][0]["main"],
                    "weather_description": weather_data["weather"][0]["description"],
                    "aqi": air_data["list"][0]["main"]["aqi"],
                    "pm2_5": air_data["list"][0]["components"]["pm2_5"],
                    "pm10": air_data["list"][0]["components"]["pm10"],
                    "no2": air_data["list"][0]["components"]["no2"],
                    "o3": air_data["list"][0]["components"]["o3"],
                    "so2": air_data["list"][0]["components"].get("so2", 0),
                    "co": air_data["list"][0]["components"].get("co", 0)
                }
                
                logger.info(f" {city_name} : Temp={record['temp']}°C, AQI={record['aqi']}, PM2.5={record['pm2_5']}")
                return record
                
            except requests.exceptions.RequestException as e:
                logger.warning(f" Tentative {attempt+1}/{MAX_RETRIES} échouée pour {city_name} : {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f" Échec définitif pour {city_name} après {MAX_RETRIES} tentatives")
                    return None
            except KeyError as e:
                logger.error(f" Structure de réponse inattendue pour {city_name} : {e}")
                return None
        
        return None
    
    def run_collection_cycle(self, cycle_index: int) -> int:
        """Exécute un cycle complet de collecte (toutes les villes)"""
        collected_in_cycle = 0
        
        for city_name, coords in CITIES.items():
            record = self.fetch_city_data(city_name, coords["lat"], coords["lon"])
            if record:
                record["cycle_index"] = cycle_index
                self.data.append(record)
                collected_in_cycle += 1
                self.progress["cities_completed"][city_name] += 1
            else:
                self.progress["failed_attempts"].append({
                    "city": city_name,
                    "cycle": cycle_index,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Sauvegarde après chaque ville (au cas où)
            self.save_progress()
        
        return collected_in_cycle
    
    def start_collection(self):
        """Démarre la collecte avec reprise automatique"""
        logger.info("=" * 60)
        logger.info(" DÉMARRAGE DE LA COLLECTE AUTOMATIQUE")
        logger.info(f" Planifié : {MAX_REQUESTS} cycles")
        logger.info(f" Villes : {', '.join(CITIES.keys())}")
        logger.info(f"⏱ Intervalle : {INTERVAL_SECONDS//3600} heure(s)")
        logger.info("=" * 60)
        
        start_cycle = self.progress["last_index"]
        
        try:
            for cycle in range(start_cycle, MAX_REQUESTS):
                logger.info(f"\n CYCLE {cycle + 1}/{MAX_REQUESTS}")
                logger.info(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                collected = self.run_collection_cycle(cycle)
                logger.info(f" Collecté : {collected}/{len(CITIES)} villes")
                
                # Mise à jour progression
                self.progress["last_index"] = cycle + 1
                self.save_progress()
                
                # Statistiques intermédiaires
                total_records = len(self.data)
                logger.info(f" Total collecté jusqu'ici : {total_records} enregistrements")
                
                # Attente avant le prochain cycle (sauf si dernier cycle)
                if cycle < MAX_REQUESTS - 1:
                    logger.info(f" Attente de {INTERVAL_SECONDS//60} minutes... (Ctrl+C pour interrompre et reprendre plus tard)")
                    for remaining in range(INTERVAL_SECONDS, 0, -60):
                        if remaining % 300 == 0 or remaining <= 60:  # Log toutes les 5 minutes
                            logger.debug(f" Prochain cycle dans {remaining//60} minutes")
                        time.sleep(60)
                
        except KeyboardInterrupt:
            logger.info("\n INTERRUPTION DÉTECTÉE - Sauvegarde en cours...")
            self.save_progress()
            logger.info(f" Sauvegarde terminée. {len(self.data)} enregistrements préservés.")
            logger.info(" Relancez le script pour continuer là où vous vous êtes arrêté.")
            
        except Exception as e:
            logger.error(f" Erreur fatale : {e}")
            self.save_progress()
            raise
        
        finally:
            self.finalize()
    
    def finalize(self):
        """Finalise la collecte et génère les rapports"""
        logger.info("\n" + "=" * 60)
        logger.info(" RAPPORT FINAL DE COLLECTE")
        logger.info("=" * 60)
        logger.info(f" Total enregistrements : {len(self.data)}")
        logger.info(f" Par ville :")
        for city, count in self.progress["cities_completed"].items():
            logger.info(f"   - {city}: {count} enregistrements")
        logger.info(f" Échecs : {len(self.progress['failed_attempts'])}")
        
        # Sauvegarde finale en CSV
        if self.data:
            df = pd.DataFrame(self.data)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = RAW_DIR / f"air_quality_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f" Données sauvegardées dans : {csv_path}")
            
            # Création aussi d'un fichier consolidé
            consolidated_path = RAW_DIR / "air_quality_latest.csv"
            df.to_csv(consolidated_path, index=False)
            logger.info(f" Version consolidée : {consolidated_path}")
        
        logger.info("=" * 60)
        logger.info("Collecte terminée !")

def main():
    """Point d'entrée principal"""
    # Validation de la clé API
    if API_KEY == "votre_clé_api_ici":
        logger.error("Veuillez configurer votre clé API dans config.py")
        logger.info(" 1. Créez un compte sur OpenWeatherMap")
        logger.info(" 2. Obtenez votre clé API")
        logger.info(" 3. Remplacez 'votre_clé_api_ici' dans config.py")
        sys.exit(1)
    
    # Lancement du collecteur
    collector = AirQualityCollector(API_KEY)
    collector.start_collection()

if __name__ == "__main__":
    main()