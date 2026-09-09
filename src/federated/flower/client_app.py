# client_app.py
import flwr as fl
import warnings
import numpy as np
from flwr.common import Context
from task import get_train_test_split, create_model, evaluate_model

warnings.filterwarnings('ignore')

class AirQualityClient(fl.client.NumPyClient):
    def __init__(self, city_name: str):
        self.city_name = city_name
        X_train, X_test, y_train, y_test = get_train_test_split(city_name)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = create_model()
        
        # ✅ INITIALISATION DES COEFFICIENTS (important pour get_parameters)
        n_features = X_train.shape[1]
        self.model.coef_ = np.zeros(n_features)
        self.model.intercept_ = 0.0
        
        print(f"🏙️ Client {city_name} initialisé : {len(y_train)} train, {len(y_test)} test, {n_features} features")
    
    def get_parameters(self, config):
        """Retourne les paramètres du modèle local"""
        # Maintenant coef_ existe toujours (initialisé dans __init__)
        coef = self.model.coef_.astype(np.float32)
        intercept = np.array([self.model.intercept_]).astype(np.float32)
        return [coef, intercept]
    
    def set_parameters(self, parameters):
        """Définit les paramètres du modèle local"""
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1][0]
    
    def fit(self, parameters, config):
        """Entraîne le modèle sur les données locales"""
        self.set_parameters(parameters)
        self.model.fit(self.X_train, self.y_train)
        
        print(f"   ✅ {self.city_name} - Entraînement terminé")
        
        return self.get_parameters(config), len(self.X_train), {}
    
    def evaluate(self, parameters, config):
        """Évalue le modèle sur les données de test locales"""
        self.set_parameters(parameters)
        y_pred = self.model.predict(self.X_test)
        
        from sklearn.metrics import mean_absolute_error, r2_score
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        print(f"   📊 {self.city_name} - MAE={mae:.3f}, R²={r2:.3f}")
        
        return float(mae), len(self.X_test), {"r2": r2}

def client_fn(context: Context):
    """Flower crée un client pour chaque partition"""
    from task import get_all_cities
    
    # Récupérer l'ID du client
    cid = context.node_config.get("partition-id")
    if cid is None:
        cid = context.node_config.get("cid", "0")
    
    cid_int = int(cid) if isinstance(cid, str) else cid
    
    # Liste des villes
    cities = get_all_cities()
    num_cities = len(cities)
    
    print(f"📌 Client ID: {cid_int}, Nombre de villes: {num_cities}")
    
    # Protection contre les IDs hors limites
    if cid_int >= num_cities:
        print(f"   ⚠️ ID {cid_int} hors limites, utilisation modulo {num_cities}")
        cid_int = cid_int % num_cities
    
    city_name = cities[cid_int]
    return AirQualityClient(city_name).to_client()

# Création de l'app client
app = fl.client.ClientApp(client_fn=client_fn)