# data_preparation.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from datetime import datetime
import warnings
import os
import joblib

warnings.filterwarnings('ignore')

# CRÉATION DES DOSSIERS NÉCESSAIRES
os.makedirs('models/federated', exist_ok=True)
os.makedirs('models/centralized', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

print("=" * 60)
print("📊 ÉTAPE 2 : PRÉPARATION DES DONNÉES")
print("=" * 60)

# 1. CHARGEMENT DES DONNÉES
print("\n📂 Chargement des données...")
df = pd.read_csv('data/raw/air_quality_latest.csv')
print(f"✅ Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# 2. INSPECTION INITIALE
print("\n📋 APERÇU DES DONNÉES :")
print(df.head(3))

# 3. NETTOYAGE
print("\n🧹 NETTOYAGE DES DONNÉES...")

# Convertir timestamp en datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Supprimer les doublons (garder le plus récent en cas de doublon)
initial_count = len(df)
df = df.sort_values('timestamp').drop_duplicates(subset=['cycle_index', 'city'], keep='last')
print(f"   - Doublons supprimés : {initial_count - len(df)}")

# Vérifier les valeurs manquantes
print(f"   - Valeurs manquantes avant interpolation :")
print(df.isnull().sum())

# 4. INTERPOLATION DES DONNÉES MANQUANTES
print("\n📊 INTERPOLATION DES DONNÉES MANQUANTES...")

# Identifier les cycles incomplets
cycle_counts = df.groupby('cycle_index')['city'].count()
missing_cycles = cycle_counts[cycle_counts < 6]

if len(missing_cycles) > 0:
    print(f"   ⚠️ {len(missing_cycles)} cycles incomplets détectés")
    
    for cycle, count in missing_cycles.items():
        cities_present = df[df['cycle_index'] == cycle]['city'].unique()
        cities_missing = [c for c in df['city'].unique() if c not in cities_present]
        print(f"      - Cycle {cycle}: {count}/6 villes (manquent: {cities_missing})")
    
    # Créer un DataFrame avec tous les cycles/villes attendus
    all_cycles = sorted(df['cycle_index'].unique())
    all_cities = sorted(df['city'].unique())
    
    print(f"\n   📊 Génération de la grille complète : {len(all_cycles)} cycles × {len(all_cities)} villes")
    
    # Créer un index complet
    full_index = pd.MultiIndex.from_product(
        [all_cycles, all_cities],
        names=['cycle_index', 'city']
    )
    
    # Reindexer le DataFrame avec gestion des doublons
    df_unique = df.drop_duplicates(subset=['cycle_index', 'city'], keep='last')
    df_full = df_unique.set_index(['cycle_index', 'city']).reindex(full_index)
    
    print(f"   📊 DataFrame réindexé : {len(df_full)} lignes")
    
    # Colonnes numériques à interpoler
    numeric_cols = ['temp', 'feels_like', 'humidity', 'pressure', 'wind_speed', 
                    'wind_deg', 'clouds', 'pm2_5', 'pm10', 'no2', 'o3', 'so2', 'co']
    
    # Interpolation par ville
    interpolated_count = 0
    for city in all_cities:
        city_mask = df_full.index.get_level_values('city') == city
        
        for col in numeric_cols:
            if col in df_full.columns:
                missing_count = df_full.loc[city_mask, col].isnull().sum()
                if missing_count > 0:
                    df_full.loc[city_mask, col] = df_full.loc[city_mask, col].interpolate(
                        method='linear',
                        limit_direction='both',
                        limit_area='inside'
                    )
                    interpolated_count += missing_count
    
    # Pour les colonnes catégorielles
    categorical_cols = ['weather_main', 'weather_description', 'aqi']
    for city in all_cities:
        city_mask = df_full.index.get_level_values('city') == city
        for col in categorical_cols:
            if col in df_full.columns:
                df_full.loc[city_mask, col] = df_full.loc[city_mask, col].fillna(method='ffill')
                df_full.loc[city_mask, col] = df_full.loc[city_mask, col].fillna(method='bfill')
    
    # Pour lat/lon (constantes par ville)
    for city in all_cities:
        city_mask = df_full.index.get_level_values('city') == city
        city_data = df[df['city'] == city]
        if len(city_data) > 0:
            df_full.loc[city_mask, 'lat'] = city_data.iloc[0]['lat']
            df_full.loc[city_mask, 'lon'] = city_data.iloc[0]['lon']
    
    # Réinitialiser l'index
    df = df_full.reset_index()
    
    print(f"   ✅ {interpolated_count} valeurs manquantes interpolées")
    
    # Vérifier après interpolation
    cycle_counts_after = df.groupby('cycle_index')['city'].count()
    complete_cycles = (cycle_counts_after == 6).sum()
    print(f"   ✅ Cycles complets après interpolation : {complete_cycles}/{len(all_cycles)}")
    print(f"   📊 Total enregistrements après interpolation : {len(df)}")
else:
    print("   ✅ Tous les cycles sont déjà complets !")

# 5. RECALCUL DES FEATURES TEMPORELLES
print("\n🔄 RECALCUL DES FEATURES TEMPORELLES...")

# Supprimer les colonnes temporelles existantes
df = df.drop(columns=['hour', 'day_of_week', 'day_of_month', 'month'], errors='ignore')

# Recalculer à partir du timestamp
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['day_of_month'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month

print(f"   ✅ Features temporelles recalculées")

# 6. TRAITEMENT DES DERNIÈRES VALEURS MANQUANTES (CORRIGÉ)
print("\n🧹 TRAITEMENT DES DERNIÈRES VALEURS MANQUANTES...")

# Identifier les colonnes numériques avec des NaN (exclure timestamp et colonnes non-numériques)
numeric_cols_for_impute = ['temp', 'feels_like', 'humidity', 'pressure', 'wind_speed', 
                           'wind_deg', 'clouds', 'pm2_5', 'pm10', 'no2', 'o3', 'so2', 'co',
                           'hour', 'day_of_week', 'day_of_month', 'month']

# Vérifier quelles colonnes ont des NaN
cols_with_nan = [col for col in numeric_cols_for_impute if col in df.columns and df[col].isnull().any()]
print(f"   Colonnes avec NaN avant imputation : {cols_with_nan if cols_with_nan else 'Aucune'}")

if cols_with_nan:
    # Utiliser SimpleImputer sur les colonnes numériques uniquement
    imputer = SimpleImputer(strategy='mean')
    df[cols_with_nan] = imputer.fit_transform(df[cols_with_nan])
    print(f"   ✅ {len(cols_with_nan)} colonnes imputées avec la moyenne")
    
    # Vérification après imputation
    remaining_nan = df[cols_with_nan].isnull().sum().sum()
    print(f"   ✅ NaN restants après imputation : {remaining_nan}")
else:
    print("   ✅ Aucune valeur manquante à traiter")

# 7. VÉRIFICATION FINALE
print("\n🔍 VÉRIFICATION FINALE DES VALEURS MANQUANTES...")

# Vérifier toutes les colonnes (sauf timestamp qui peut être ignoré)
check_cols = [col for col in df.columns if col != 'timestamp']
missing_check = df[check_cols].isnull().sum()
total_nan = missing_check.sum()

if total_nan > 0:
    print(f"   ⚠️ Il reste des valeurs manquantes :")
    print(missing_check[missing_check > 0])
    # Supprimer les dernières valeurs manquantes si nécessaire
    df = df.dropna(subset=check_cols)
    print(f"   - Lignes après suppression des NaN restants : {len(df)}")
else:
    print("   ✅ Aucune valeur manquante restante !")

# 8. ENCODAGE DES VARIABLES CATÉGORIELLES
print("\n🏷️ ENCODAGE DES VILLES...")

le_city = LabelEncoder()
df['city_encoded'] = le_city.fit_transform(df['city'])
city_mapping = dict(zip(le_city.classes_, le_city.transform(le_city.classes_)))
print(f"   - Mapping des villes : {city_mapping}")

# 9. SÉLECTION DES FEATURES
print("\n🔧 SÉLECTION DES FEATURES...")

features_regression = [
    'temp', 'humidity', 'pressure', 'wind_speed',
    'pm10', 'no2', 'o3', 'so2', 'co',
    'hour', 'day_of_week', 'city_encoded'
]

print(f"   - {len(features_regression)} features sélectionnées")

# 10. NORMALISATION
print("\n📏 NORMALISATION DES FEATURES NUMÉRIQUES...")

numeric_features = ['temp', 'humidity', 'pressure', 'wind_speed', 
                    'pm10', 'no2', 'o3', 'so2', 'co']

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numeric_features] = scaler.fit_transform(df[numeric_features])

print(f"   - Normalisation terminée sur {len(numeric_features)} features")

# 11. PRÉPARATION DES TARGETS
print("\n🎯 PRÉPARATION DES VARIABLES CIBLES...")

y_regression = df_scaled['pm2_5']
y_classification = df_scaled['aqi']

print(f"   - Régression (PM2.5) : {y_regression.nunique()} valeurs uniques")
print(f"   - Classification (AQI) : {y_classification.nunique()} classes")

print(f"\n   📊 Distribution des classes AQI :")
aqi_counts = df['aqi'].value_counts().sort_index()
for aqi_class, count in aqi_counts.items():
    aqi_names = {1: 'Bon', 2: 'Moyen', 3: 'Dégradé', 4: 'Mauvais', 5: 'Très mauvais'}
    name = aqi_names.get(aqi_class, 'Inconnu')
    print(f"      - Classe {aqi_class} ({name}) : {count} enregistrements")

# 12. SAUVEGARDE DES DONNÉES PRÉPARÉES
print("\n💾 SAUVEGARDE DES DONNÉES PRÉPARÉES...")

df_clean = df.copy()
df_clean.to_csv('data/processed/air_quality_clean.csv', index=False)
print(f"   ✓ Dataset nettoyé : data/processed/air_quality_clean.csv")

df_scaled.to_csv('data/processed/air_quality_scaled.csv', index=False)
print(f"   ✓ Dataset normalisé : data/processed/air_quality_scaled.csv")

joblib.dump(le_city, 'models/federated/city_encoder.pkl')
joblib.dump(scaler, 'models/federated/feature_scaler.pkl')
print(f"   ✓ Encoders sauvegardés dans : models/federated/")

print("\n" + "=" * 60)
print("✅ ÉTAPE 2 TERMINÉE : Données prêtes pour le modelling !")
print("=" * 60)

# 13. STATISTIQUES FINALES
print("\n📊 RÉSUMÉ FINAL :")
print(f"   ✓ Total enregistrements : {len(df)}")
print(f"   ✓ Villes : {list(df['city'].unique())}")
print(f"   ✓ Période : du {df['timestamp'].min()} au {df['timestamp'].max()}")
print(f"   ✓ Features disponibles : {len(features_regression)}")

total_expected = df['cycle_index'].nunique() * 6
print(f"   ✓ Taux de collecte : {len(df)/total_expected*100:.1f}% ({len(df)}/{total_expected})")
print(f"   ✓ Cycles complets : {df.groupby('cycle_index')['city'].nunique().eq(6).sum()} / {df['cycle_index'].nunique()}")

print("\n📋 APERÇU DES DONNÉES PRÉPARÉES :")
print(df_scaled[['city', 'temp', 'humidity', 'pm2_5', 'aqi', 'city_encoded']].head())

print("\n📊 RAPPORT SUR LES DONNÉES MANQUANTES :")
print("=" * 60)
total_expected = df['cycle_index'].nunique() * 6
print(f"   Enregistrements attendus : {total_expected}")
print(f"   Enregistrements collectés : {len(df)}")
print(f"   Taux de complétude : {len(df)/total_expected*100:.1f}%")

# Vérification finale des NaN
check_cols = [col for col in df.columns if col != 'timestamp']
nan_count = df[check_cols].isnull().sum().sum()
print(f"   NaN restants : {nan_count}")
if nan_count == 0:
    print("   ✅ Jeu de données complet et prêt pour la modélisation !")
else:
    print(f"   ⚠️ Attention : {nan_count} valeurs NaN restantes")