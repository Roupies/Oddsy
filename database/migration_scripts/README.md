# 🔄 Migration CSV vers PostgreSQL

## Pourquoi COPY au lieu d'INSERT ?

**COPY est 10-100x plus rapide** que les INSERT individuels :

- ✅ **Performance:** Traite des millions de lignes en secondes
- ✅ **Atomique:** Une seule transaction pour tout le fichier  
- ✅ **Optimisé:** PostgreSQL désactive les index temporairement
- ✅ **Memory-efficient:** Stream les données sans tout charger en RAM

## 🚀 Utilisation

### Démarrage rapide
```bash
# 1. S'assurer que PostgreSQL tourne
docker-compose up -d

# 2. Lancer la migration
cd database/migration_scripts
python csv_to_postgres.py
```

### Migration personnalisée
```python
from csv_to_postgres import CSVToPostgresqlMigrator
from python_connector import OddsyDatabase

# Connexion
db = OddsyDatabase()
migrator = CSVToPostgresqlMigrator(db)

# Migrer fichier spécifique
migrator.migrate_matches_from_csv("../../data/processed/v_auto_update_20250922_093416.csv")
```

## 📊 Performance Comparison

| Méthode | 10K lignes | 100K lignes | 1M lignes |
|---------|------------|-------------|-----------|
| INSERT individuels | ~45s | ~8min | ~1h30 |
| INSERT batch (1000) | ~8s | ~1min | ~12min |
| **COPY** | **~0.5s** | **~3s** | **~25s** |

## 🔧 Mapping Colonnes CSV → PostgreSQL

Le script mappe automatiquement :

```python
column_mapping = {
    'Date': 'match_date',
    'HomeTeam': 'home_team', 
    'AwayTeam': 'away_team',
    'FTHG': 'home_goals',
    'FTAG': 'away_goals',
    'FTR': 'full_time_result',
    'B365H': 'home_odds',
    'B365D': 'draw_odds', 
    'B365A': 'away_odds'
    # ... etc
}
```

## ✅ Validation des Données

Le script valide automatiquement :

- **Team IDs:** Lookup depuis table `teams`
- **Résultats:** Seuls H/D/A acceptés
- **Types numériques:** Conversion automatique
- **Dates:** Format ISO standard
- **Contraintes:** Respect schéma PostgreSQL

## 🛠️ Gestion d'Erreurs

- **Transaction atomique:** Rollback si erreur
- **Logging détaillé:** Suivi des étapes
- **Équipes manquantes:** Warning + skip
- **Données invalides:** Nettoyage automatique

## 📈 Post-Migration

Après migration, vérifier :

```sql
-- Stats générales
SELECT COUNT(*) FROM matches;
SELECT COUNT(*) FROM teams;

-- Distribution par saison  
SELECT season, COUNT(*) 
FROM matches 
GROUP BY season 
ORDER BY season;

-- Vérifier données récentes
SELECT * FROM match_results 
WHERE season = '2025-2026' 
ORDER BY match_date DESC 
LIMIT 10;
```

## 🔄 Re-migration

Pour nettoyer et re-migrer :

```sql
-- Vider tables (attention: supprime tout!)
TRUNCATE TABLE matches RESTART IDENTITY CASCADE;
TRUNCATE TABLE teams RESTART IDENTITY CASCADE;
```

Puis relancer le script Python.

## ⚡ Optimisations COPY

Le script utilise :

- **Format CSV optimisé** avec délimiteur TAB
- **NULL handling** avec `\\N` 
- **Single transaction** pour atomicité
- **Column ordering** pour performance maximale
- **Memory streaming** via io.StringIO