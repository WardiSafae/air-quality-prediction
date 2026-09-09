# model_building.py 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE  # Pour équilibrer les classes
import warnings
import os
import joblib

warnings.filterwarnings('ignore')

# Création des dossiers
os.makedirs('models', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

# Configuration des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 60)
print("🤖 ÉTAPE 3 : CONSTRUCTION DES MODÈLES")
print("=" * 60)

# 1. CHARGEMENT DES DONNÉES PRÉPARÉES
print("\n📂 Chargement des données...")
df = pd.read_csv('data/processed/air_quality_scaled.csv')
print(f"   ✓ {len(df)} enregistrements chargés")

# 2. PRÉPARATION POUR LA RÉGRESSION (PM2.5)
print("\n" + "=" * 60)
print("📈 PARTIE 1 : RÉGRESSION - Prédiction de PM2.5")
print("=" * 60)

# Sélection des features et target
features = ['temp', 'humidity', 'pressure', 'wind_speed', 
            'pm10', 'no2', 'o3', 'so2', 'co',
            'hour', 'day_of_week', 'city_encoded']

X = df[features]
y = df['pm2_5']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n📊 Split des données :")
print(f"   - Training : {len(X_train)} échantillons")
print(f"   - Test : {len(X_test)} échantillons")

# 3. MODÈLE 1 : RÉGRESSION LINÉAIRE
print("\n" + "-" * 40)
print("🔹 Modèle 1 : Régression Linéaire")
print("-" * 40)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print(f"   ✓ MAE  : {mae_lr:.3f} µg/m³")
print(f"   ✓ RMSE : {rmse_lr:.3f} µg/m³")
print(f"   ✓ R²   : {r2_lr:.3f}")

# 4. MODÈLE 2 : RANDOM FOREST
print("\n" + "-" * 40)
print("🔹 Modèle 2 : Random Forest")
print("-" * 40)

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print(f"   ✓ MAE  : {mae_rf:.3f} µg/m³")
print(f"   ✓ RMSE : {rmse_rf:.3f} µg/m³")
print(f"   ✓ R²   : {r2_rf:.3f}")

# 5. MODÈLE 3 : ARBRE DE DÉCISION
print("\n" + "-" * 40)
print("🔹 Modèle 3 : Arbre de Décision")
print("-" * 40)

dt = DecisionTreeRegressor(max_depth=10, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

mae_dt = mean_absolute_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
r2_dt = r2_score(y_test, y_pred_dt)

print(f"   ✓ MAE  : {mae_dt:.3f} µg/m³")
print(f"   ✓ RMSE : {rmse_dt:.3f} µg/m³")
print(f"   ✓ R²   : {r2_dt:.3f}")

# 6. COMPARAISON DES MODÈLES DE RÉGRESSION
print("\n" + "-" * 40)
print("📊 COMPARAISON DES MODÈLES (Régression)")
print("-" * 40)

comparison = pd.DataFrame({
    'Modèle': ['Linear Regression', 'Random Forest', 'Decision Tree'],
    'MAE': [mae_lr, mae_rf, mae_dt],
    'RMSE': [rmse_lr, rmse_rf, rmse_dt],
    'R²': [r2_lr, r2_rf, r2_dt]
})
print(comparison.to_string(index=False))

# Identifier le meilleur modèle
best_regressor = 'Random Forest' if r2_rf > r2_lr and r2_rf > r2_dt else \
                 'Linear Regression' if r2_lr > r2_rf and r2_lr > r2_dt else \
                 'Decision Tree'
print(f"\n🏆 Meilleur modèle pour la régression : {best_regressor}")

# 7. PARTIE 2 : CLASSIFICATION (AQI)
print("\n" + "=" * 60)
print("📊 PARTIE 2 : CLASSIFICATION - Prédiction de l'AQI")
print("=" * 60)

# Préparation pour classification
y_clf = df['aqi']

# Vérifier le déséquilibre des classes
print("\n📊 Distribution des classes AQI :")
class_dist = y_clf.value_counts().sort_index()
for cls, count in class_dist.items():
    print(f"   - Classe {cls}: {count} ({count/len(y_clf)*100:.1f}%)")

# Split avec stratification (important pour classes déséquilibrées)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

print(f"\n📊 Split avec stratification :")
print(f"   - Training : {len(X_train_c)} échantillons")
print(f"   - Test : {len(X_test_c)} échantillons")

# Option: Application de SMOTE pour équilibrer (décommentez si besoin)
# print("\n⚖️ Application de SMOTE pour équilibrer les classes...")
# smote = SMOTE(random_state=42)
# X_train_c, y_train_c = smote.fit_resample(X_train_c, y_train_c)
# print(f"   - Après SMOTE : {len(X_train_c)} échantillons")

# Modèle Random Forest Classifier
print("\n🔹 Random Forest Classifier")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_clf.fit(X_train_c, y_train_c)
y_pred_c = rf_clf.predict(X_test_c)

accuracy = accuracy_score(y_test_c, y_pred_c)
f1 = f1_score(y_test_c, y_pred_c, average='weighted')

print(f"   ✓ Accuracy : {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"   ✓ F1-Score (weighted) : {f1:.3f}")

print(f"\n   ✓ Classification Report :")
print(classification_report(y_test_c, y_pred_c))

# 8. SAUVEGARDE DES MODÈLES
print("\n💾 SAUVEGARDE DES MODÈLES...")

joblib.dump(rf, 'models/centralized/random_forest_regressor.pkl')
joblib.dump(rf_clf, 'models/centralized/random_forest_classifier.pkl')
joblib.dump(lr, 'models/centralized/linear_regression.pkl')

print(f"   ✓ Modèles sauvegardés dans : models/")

# 9. GÉNÉRATION DES VISUALISATIONS
print("\n📈 GÉNÉRATION DES VISUALISATIONS...")

# Figure 1 : Comparaison des modèles
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

models_names = ['LR', 'RF', 'DT']
mae_values = [mae_lr, mae_rf, mae_dt]
rmse_values = [rmse_lr, rmse_rf, rmse_dt]
r2_values = [r2_lr, r2_rf, r2_dt]

x = np.arange(len(models_names))
width = 0.35

axes[0].bar(x - width/2, mae_values, width, label='MAE', color='skyblue')
axes[0].bar(x + width/2, rmse_values, width, label='RMSE', color='lightcoral')
axes[0].set_xlabel('Modèles')
axes[0].set_ylabel('Erreur')
axes[0].set_title('Comparaison des Erreurs (MAE vs RMSE)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models_names)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].bar(models_names, r2_values, color='lightgreen')
axes[1].set_xlabel('Modèles')
axes[1].set_ylabel('R²')
axes[1].set_title('Coefficient de Détermination (R²)')
axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/model_comparison.png', dpi=150)
plt.close()
print(f"   ✓ visualizations/model_comparison.png")

# Figure 2 : Matrice de confusion
from sklearn.metrics import ConfusionMatrixDisplay

fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_estimator(rf_clf, X_test_c, y_test_c, ax=ax)
plt.title('Matrice de Confusion - Classification AQI')
plt.savefig('visualizations/confusion_matrix.png', dpi=150)
plt.close()
print(f"   ✓ visualizations/confusion_matrix.png")

# Figure 3 : Importance des features
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature', palette='viridis')
plt.title('Top 10 Features les plus importantes (Random Forest)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('visualizations/feature_importance.png', dpi=150)
plt.close()
print(f"   ✓ visualizations/feature_importance.png")

# Figure 4 : Distribution des erreurs
fig, ax = plt.subplots(figsize=(10, 5))
errors = y_test - y_pred_rf
sns.histplot(errors, bins=30, kde=True, ax=ax)
ax.axvline(x=0, color='red', linestyle='--', label='Erreur zéro')
ax.set_xlabel('Erreur de prédiction (PM2.5)')
ax.set_ylabel('Fréquence')
ax.set_title('Distribution des erreurs - Random Forest')
ax.legend()
plt.tight_layout()
plt.savefig('visualizations/error_distribution.png', dpi=150)
plt.close()
print(f"   ✓ visualizations/error_distribution.png")

# 10. RAPPORT FINAL
print("\n" + "=" * 60)
print("✅ ÉTAPE 3 TERMINÉE : Modèles entraînés et évalués !")
print("=" * 60)

print("\n📋 RÉSUMÉ DES PERFORMANCES :")
print(f"\n🔹 RÉGRESSION (Prédiction PM2.5)")
print(f"   → Linear Regression : R² = {r2_lr:.3f}")
print(f"   → Random Forest     : R² = {r2_rf:.3f}")
print(f"   → Decision Tree     : R² = {r2_dt:.3f}")
print(f"   → Meilleur modèle   : {best_regressor}")

print(f"\n🔹 CLASSIFICATION (Prédiction AQI)")
print(f"   → Accuracy  : {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"   → F1-Score  : {f1:.3f}")

print("\n📁 Fichiers générés :")
print("   - models/random_forest_regressor.pkl")
print("   - models/random_forest_classifier.pkl")
print("   - models/linear_regression.pkl")
print("   - visualizations/*.png (4 graphiques)")

print("\n" + "=" * 60)



