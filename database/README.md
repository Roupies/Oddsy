# 🗄️ Oddsy Database Setup

## PostgreSQL avec Docker

### 🚀 Démarrage rapide

```bash
# Démarrer la base de données
docker-compose up -d

# Vérifier le statut
docker ps

# Se connecter à la base
docker exec -it oddsy_postgres psql -U oddsy_user -d oddsy_football
```

### 🔧 Configuration

**Connexion PostgreSQL:**
- Host: `localhost`
- Port: `5432` 
- Database: `oddsy_football`
- User: `oddsy_user`
- Password: `oddsy_password`

**Administration pgAdmin:**
- URL: http://localhost:8080
- Email: `admin@oddsy.local`
- Password: `admin_password`

### 📊 Structure de la base

#### Tables principales:
- **`teams`** - Équipes Premier League
- **`matches`** - Matchs historiques et résultats
- **`predictions`** - Prédictions des modèles ML
- **`model_performance`** - Métriques de performance

#### Vues utiles:
- **`match_results`** - Résultats avec noms d'équipes
- **`model_performance_summary`** - Résumé performance modèles
- **`prediction_accuracy`** - Précision par modèle

### 💻 Commandes utiles

```bash
# Arrêter les services
docker-compose down

# Redémarrer avec reconstruction
docker-compose up -d --build

# Voir les logs
docker-compose logs -f postgres

# Backup de la base
docker exec oddsy_postgres pg_dump -U oddsy_user oddsy_football > backup.sql

# Restaurer un backup
cat backup.sql | docker exec -i oddsy_postgres psql -U oddsy_user -d oddsy_football
```

### 🔄 Intégration Python

```python
import psycopg2
import pandas as pd

# Connexion
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="oddsy_football",
    user="oddsy_user", 
    password="oddsy_password"
)

# Exemple: Charger les matchs récents
df = pd.read_sql("""
    SELECT * FROM match_results 
    WHERE season = '2025-2026' 
    ORDER BY match_date DESC
""", conn)
```

### 📈 Requêtes d'exemple

```sql
-- Performance des modèles
SELECT * FROM prediction_accuracy;

-- Matchs récents avec prédictions
SELECT mr.*, p.predicted_result, p.probability_home, p.probability_draw, p.probability_away
FROM match_results mr
LEFT JOIN predictions p ON mr.match_id = p.match_id
WHERE mr.season = '2025-2026'
ORDER BY mr.match_date DESC;

-- Statistiques par équipe  
SELECT 
    home_team,
    COUNT(*) as matches_played,
    SUM(CASE WHEN full_time_result = 'H' THEN 1 ELSE 0 END) as wins,
    AVG(home_goals) as avg_goals_scored
FROM match_results
WHERE season = '2025-2026'
GROUP BY home_team
ORDER BY wins DESC;
```