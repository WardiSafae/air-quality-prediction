# task.py 
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
import os

warnings.filterwarnings('ignore')

# Configuration globale
FEATURES = ['temp', 'humidity', 'pressure', 'wind_speed', 
            'pm10', 'no2', 'o3', 'so2', 'co',
            'hour', 'day_of_week', 'city_encoded']

# Déterminer le chemin absolu du projet (remonter jusqu'à la racine)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Ou plus simplement :
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def load_data():
    """Charge les données préparées"""
    # Chemin absolu vers le fichier
    file_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'air_quality_scaled.csv')
    print(f"📂 Chargement du fichier : {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier non trouvé : {file_path}")
    
    df = pd.read_csv(file_path)
    return df

def get_city_data(city_name: str):
    """Retourne les données d'une ville spécifique"""
    df = load_data()
    city_df = df[df['city'] == city_name]
    
    X = city_df[FEATURES].values
    y = city_df['pm2_5'].values
    
    return X, y

def get_all_cities():
    """Retourne la liste de toutes les villes"""
    df = load_data()
    return df['city'].unique().tolist()

def get_train_test_split(city_name: str, test_size: float = 0.2, random_state: int = 42):
    """Retourne les données train/test pour une ville"""
    X, y = get_city_data(city_name)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def create_model():
    """Crée un nouveau modèle Linear Regression"""
    return LinearRegression()

def evaluate_model(model, X_test, y_test):
    """Évalue le modèle sur les données de test"""
    y_pred = model.predict(X_test)
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, r2