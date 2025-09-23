#!/bin/bash

# =====================================================
# 🚀 QUICK MIGRATION SCRIPT - CSV to PostgreSQL
# =====================================================
# Script simple pour migration rapide des CSVs

set -e  # Arrêt si erreur

echo "🔄 Quick Migration CSV → PostgreSQL"
echo "=================================="

# Vérifier que Docker tourne
echo "🐳 Vérification Docker..."
if ! docker ps | grep -q oddsy_postgres; then
    echo "⚠️  PostgreSQL non démarré, lancement..."
    docker-compose up -d
    echo "⏳ Attente démarrage (10s)..."
    sleep 10
else
    echo "✅ PostgreSQL déjà actif"
fi

# Vérifier connexion base
echo "🔗 Test connexion base..."
if docker exec oddsy_postgres psql -U oddsy_user -d oddsy_football -c "SELECT 1;" > /dev/null 2>&1; then
    echo "✅ Connexion PostgreSQL OK"
else
    echo "❌ Impossible de se connecter à PostgreSQL"
    exit 1
fi

# Lancer migration Python
echo "📊 Lancement migration..."
cd "$(dirname "$0")"

if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python non trouvé"
    exit 1
fi

echo "🐍 Utilisation de: $PYTHON_CMD"

# Installer dépendances si nécessaire
if ! $PYTHON_CMD -c "import psycopg2" 2>/dev/null; then
    echo "📦 Installation psycopg2..."
    pip install psycopg2-binary
fi

# Migration
echo "🚀 Migration en cours..."
$PYTHON_CMD csv_to_postgres.py

# Vérification post-migration
echo "📊 Vérification données..."
MATCH_COUNT=$(docker exec oddsy_postgres psql -U oddsy_user -d oddsy_football -t -c "SELECT COUNT(*) FROM matches;")
TEAM_COUNT=$(docker exec oddsy_postgres psql -U oddsy_user -d oddsy_football -t -c "SELECT COUNT(*) FROM teams;")

echo "✅ Migration terminée!"
echo "📈 Résultats:"
echo "   - Équipes: $TEAM_COUNT"
echo "   - Matchs: $MATCH_COUNT"

echo ""
echo "🎯 Prochaines étapes:"
echo "   - Accès base: docker exec -it oddsy_postgres psql -U oddsy_user -d oddsy_football"
echo "   - Interface web: http://localhost:8080"
echo "   - Tester requêtes: SELECT * FROM match_results LIMIT 5;"