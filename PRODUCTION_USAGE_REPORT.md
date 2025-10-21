# Production Usage Mapping Report
**Date**: 2025-10-21  
**Objectif**: Nettoyer le repo pour ne garder que les fichiers production

## Structure Production Identifiée

### ✅ Frontend (Next.js 14)
- **Localisation**: `frontend/`
- **Type**: Next.js App Router avec TypeScript
- **Pages critiques**:
  - `/` (home avec HeroSplitPremium + FortressAnalysis)
  - `/matchday/[round]` (prédictions J8 etc.)
  - `/models` (performance modèles)
  - `/pipeline` (status pipeline)
- **Fichiers essentiels**:
  - `app/` - App Router pages
  - `components/` - Composants UI réutilisables  
  - `lib/` - API client, types, utilitaires
  - `hooks/` - Hooks React personnalisés
  - `public/` - Assets statiques (vidéos, logos)
  - Config: `next.config.js`, `tailwind.config.ts`, `package.json`
- **Total**: ~1165 fichiers TypeScript/React

### ✅ Backend (FastAPI)
- **Localisation**: `backend/`
- **Type**: FastAPI structuré avec API v1 et v5
- **Endpoints critiques**:
  - `GET /api/v5/gameweeks/available`
  - `GET /api/v5/gameweeks/{gameweek}/predictions`
  - `GET /api/v1/health/metrics`
- **Structure**:
  - `main.py` - Point d'entrée FastAPI
  - `api/` - Routes API v1 et v5
  - `core/` - Configuration et exceptions
  - `schemas/` - Modèles Pydantic
  - `services/` - Logique métier
- **Total**: 22 fichiers Python
- **Backend temporaire**: `simple_j8_server.py` (actuellement utilisé)

## Structure Expérimentale/Dev à Archiver

### 📦 Scripts ML/Data Science (518 fichiers Python racine)
- Modèles ML: `enhanced_*.py`, `cascade_*.py`, `pipeline_*.py`  
- Extraction données: `extract_*.py`, `understat_*.py`
- Tests/expérimentations: `test_*.py`, `debug_*.py`
- Monitoring: `monitoring/`, alertes

### 📦 Dossiers d'expérimentation
- `archive/` - Archives existantes
- `data/` - Datasets locaux  
- `models/production/` - Modèles ML entraînés
- `scripts/` - Scripts d'analyse/maintenance
- `experiments/` - Code expérimental
- `notebooks/` - Jupyter notebooks
- `outputs/` - Résultats d'analyses
- `reports/` - Rapports techniques
- `temp/` - Fichiers temporaires

### 📦 Configuration développement
- `.venv/` - Environnement Python
- `database/` - Setup PostgreSQL
- `.env.template`, configs diverses

## Validation Production

### Pages testées ✅
- [x] Home (/) - HeroSplit + vidéos
- [x] Predictions (/matchday) - Redirection vers J8
- [x] Matchday (/matchday/8) - Cartes prédictions
- [x] Models (/models) - Performance metrics  
- [x] Pipeline (/pipeline) - Status système

### APIs testées ✅  
- [x] Backend FastAPI sur port 8000
- [x] Endpoints v5 fonctionnels
- [x] CORS configuré pour localhost:3000

### Assets critiques ✅
- [x] Vidéos: `/videos/oddsy-bg-1080p.webm`, `/videos/oddsy-hero-poster.svg`
- [x] Images stades: Arsenal.avif, Chelsea.webp, Liverpool.webp, Manchester_City.jpg
- [x] Logos/icônes utilisés par les composants

## Plan de Nettoyage

### Étape 1: Sauvegarde Git
```bash
git tag pre-cleanup-20251021
git branch archive/experimental-2025-10
```

### Étape 2: Fichiers à Conserver (Production)
```
frontend/           # Next.js app complète
backend/            # FastAPI structuré  
simple_j8_server.py # Backend temporaire actuel
.gitignore
README.md           # Si documentation projet
package.json        # Root level si nécessaire
```

### Étape 3: Fichiers à Archiver
```
archive/ archives/ data/ database/ experiments/
models/ notebooks/ outputs/ reports/ scripts/ temp/
*.py (518 scripts racine sauf simple_j8_server.py)
.venv/ .env.template
monitoring/ dashboards/ (si non utilisés en prod)
```

### Étape 4: Validation Post-Nettoyage
- [ ] `cd frontend && npm run build` ✅
- [ ] `python simple_j8_server.py` ✅ 
- [ ] Test pages critiques navigateur
- [ ] Vérification assets manquants

## État Final Souhaité

```
oddsy/
├── frontend/           # Next.js production app
├── backend/            # FastAPI production server  
├── simple_j8_server.py # Current backend
├── README.md           # Production layout doc
└── .gitignore          # Clean ignores
```

**Réduction**: ~600 fichiers → ~50 fichiers essentiels  
**Historique**: Tout accessible via branche `archive/experimental-2025-10`