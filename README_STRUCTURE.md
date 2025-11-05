# Oddsy - AI Premier League Predictions

## 📁 Project Structure

Cette organisation facilite la maintenance et la compréhension du projet en séparant clairement les environnements.

### `/prod/` - Environment de Production
Version finale prête pour présentation et déploiement.

**Contenu :**
- `frontend/` - Application Next.js optimisée
- `backend/` - API FastAPI avec modèle Enhanced Baseline v3.0
- `data/` - Données de prédictions validées
- `config/` - Configuration production

**Usage :**
```bash
cd prod/backend && python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
cd prod/frontend && npm run dev
```

### `/rest/` - Archive & Legacy
Code hérité et versions antérieures du projet.

**Contenu :**
- Anciennes versions
- Code expérimental archivé
- Prototypes abandonnés

### `/dev/` - Environment de Développement
Code source, scripts d'entraînement et outils de développement.

**Contenu :**
- `models/` - Modèles entraînés et métadonnées
- `backend/` & `frontend/` - Versions de développement
- `scripts/` - Scripts d'entraînement et pipeline
- `validation/` - Tests et validation
- `data/` - Datasets et données d'entraînement

## 🚀 Quick Start

**Pour la présentation :**
```bash
# Lancer l'environnement de production
cd prod/
# Suivre les instructions dans prod/README.md
```

**Pour le développement :**
```bash
# Travailler sur les modèles et features
cd dev/
# Scripts d'entraînement, validation, etc.
```

## 📊 Performance

- **Enhanced Baseline v3.0** : 51.3% d'accuracy
- **Pipeline v1.0** : Validation temporelle stricte
- **Validé sur** : 100+ matchs EPL réels

---