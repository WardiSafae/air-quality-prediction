# server_app.py
import flwr as fl
from flwr.common import Context
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig, ServerAppComponents

def server_fn(context: Context):
    """Retourne les composants du serveur pour Flower"""
    
    # Configuration des rounds
    num_rounds = context.run_config.get("num-server-rounds", 5)
    server_config = ServerConfig(num_rounds=num_rounds)
    
    # Stratégie FedAvg standard
    strategy = FedAvg(
        fraction_fit=1.0,      # Utiliser tous les clients
        fraction_evaluate=1.0,  # Évaluer tous les clients
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=6,  # 6 villes
    )
    
    print(f"🚀 Serveur configuré : {num_rounds} rounds")
    
    # Retourner les composants (Nouvelle API)
    return ServerAppComponents(
        strategy=strategy,
        config=server_config,
    )

# Création de l'app serveur avec la nouvelle API
app = fl.server.ServerApp(server_fn=server_fn)

