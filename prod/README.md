# Oddsy - AI Premier League Predictions Platform

> Advanced machine learning platform delivering validated Premier League match predictions with **51.3% accuracy** using Enhanced Baseline v3.0 and Pipeline Durci v1.0.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │───▶│    Backend      │───▶│   JSON Files    │
│   Next.js 14   │    │   FastAPI       │    │  (Atomic I/O)   │
│   TypeScript    │    │   Python 3.11  │    │   + ISR Cache   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                               ┌───────▼─────────┐
                                               │   PostgreSQL    │
                                               │   (Future/Off)  │
                                               └─────────────────┘
```

## 🛠️ Technology Stack

### Primary Architecture
- **Backend**: FastAPI (Python 3.11+), atomic file operations
- **Frontend**: Next.js 14 (React 19, TypeScript), ISR performance  
- **ML Pipeline**: Enhanced Baseline v3.0 (51.3% validated accuracy)
- **Storage**: JSON files with atomic writes (production-ready)
- **Infrastructure**: Docker Compose, structured logging, rate limiting

### Future-Ready Components (Dormant)
- **Database**: PostgreSQL 16 (configured, inactive, ready for scaling)
- **Admin**: pgAdmin 4 (optional, for future DB management)

## ✨ Recent Improvements (Technical Maturity)

- ✅ **English Documentation**: All code commented professionally throughout codebase
- ✅ **Dual API Architecture**: v1 (health/operations) + v5 (predictions) for clear separation  
- ✅ **Atomic Operations**: Crash-safe file handling with atomic writes
- ✅ **Format Evolution**: v3 predictions with full backward compatibility
- ✅ **Codebase Cleanup**: Removed empty directories, optimized project structure
- ✅ **Production Readiness**: Rate limiting, structured logging, correlation IDs
- ✅ **Type Safety**: Zod validation schemas with runtime type checking
- ✅ **Performance**: ISR caching, 5-minute revalidation, efficient data flow

## 🚀 Quick Start

### Minimal Setup (Recommended for Demo)
```bash
# 1. Backend (File-based, no DB required)
cd backend && python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# 2. Frontend  
cd frontend && npm install && npm run dev

# 3. Access
# ✅ Frontend: http://localhost:3000
# ✅ Backend API: http://localhost:8000/api/system/docs
# ✅ Health Check: http://localhost:8000/api/system/health/live
```

### Advanced Setup (Optional PostgreSQL Testing)
```bash
# Optional: Future database features
docker-compose up -d postgres pgadmin

# Note: Not required for core functionality
# pgAdmin: http://localhost:8080 (credentials in .env.example)
```

## 📋 Prerequisites

### Required
- **Python 3.11+** (for FastAPI backend)
- **Node.js 18+** (for Next.js frontend)

### Optional (Future Features)
- **Docker & Docker Compose** (for PostgreSQL when scaling)
- **PostgreSQL client** (for direct DB access when activated)

## 🔧 Detailed Setup

### 1. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings (optional for development)
nano .env
```

### 2. Database Setup

```bash
# Start PostgreSQL and pgAdmin
docker-compose up -d postgres pgadmin

# Wait for services to be ready (healthchecks will ensure proper boot order)
docker-compose logs postgres

# Run migrations
./scripts/migrate.sh
```

### 3. Backend Setup

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# For production build
npm run build && npm start
```

## 💾 Data Storage Architecture

### Current: File-Based JSON (Production-Ready)
- **Location**: `data/predictions/j{round}_predictions_v3_*.json`
- **Features**: Atomic writes, crash-safe, ISR-optimized
- **Performance**: 5min cache, real-time updates
- **Format**: v3 prediction schema with v5 API compatibility
- **Calendar**: `data/EPL_25_26_Full_Calendar.csv` (fixture data)

### Future: PostgreSQL Integration (Ready but Inactive)
- **Status**: Schema designed, migrations ready in `database/` directory
- **Purpose**: Horizontal scaling, complex queries, analytics
- **Activation**: Available when scaling requirements emerge
- **Connection**: Pre-configured in `docker-compose.yml`

### pgAdmin Access (When PostgreSQL Active)
- URL: http://localhost:8080
- Credentials: See `.env.example` for default values
- Database connection details in environment configuration

## 🔌 API Documentation

### Dual API Architecture

**Clean separation for evaluation:**
- **v1 API** (`/api/v1` & `/api/system`) - Health, metrics, pipeline operations
- **v5 API** (`/api/v5` & `/api/gameweeks`) - Predictions, fixtures, ML data

### Core Endpoints

#### Predictions & Data (v5 API)
```bash
# Get latest gameweek
GET /api/v5/gameweeks/latest

# Get gameweek predictions (v3 format)
GET /api/v5/gameweeks/{gameweek}/predictions

# Get fixture data with real kickoff times
GET /api/v5/gameweeks/{gameweek}/fixtures

# Available gameweeks
GET /api/v5/gameweeks/available
```

#### System & Health (v1 API)
```bash
# Health check
GET /api/v1/health/live

# Readiness probe (checks file access)
GET /api/v1/health/ready

# System metrics
GET /api/v1/health/metrics

# Pipeline status
GET /api/v1/pipeline/status
```

### Response Format
```json
{
  "api_version": "5.0.0",
  "generated_at": "2025-11-05T10:00:00Z",
  "data": {
    "gameweek": 11,
    "predictions": {...}
  }
}
```

## 🤖 AI Prediction Pipeline

### Enhanced Baseline v3.0
- **Accuracy**: 51.3% on real EPL data
- **Validation**: Cross-validated on 2280 matches (2019-2025)
- **Features**: Team form, venue advantage, tactical patterns
- **Anti-leakage**: Strict temporal validation

### Pipeline Durci v1.0
- **Hardened production pipeline**
- **Temporal validation**: No future data contamination
- **Ensemble methods**: Multiple model aggregation
- **Real-time updates**: 5-minute cache revalidation

### Model Performance
```
Cross-Validation Accuracy: 53.5%
Real EPL Performance: 51.3%
Dataset: 2280 matches (2019-2025)
Confidence Intervals: 51.3% ± 5.7%
```

## 🛠️ Services Architecture

### File-Based Storage Service
- **Atomic writes**: Crash-safe JSON operations
- **Prediction files**: `j{round}_predictions_v3_*.json` format
- **Calendar data**: `EPL_25_26_Full_Calendar.csv` for fixtures
- **Performance**: Direct file access, no DB overhead

### Fixture Service
- Reads EPL calendar CSV with real kickoff times
- Maps team name variations for consistency
- Converts UTC to local time for display
- Endpoint: `/api/v5/gameweeks/{gw}/fixtures`

### Prediction Service
- Loads v3 prediction files with Enhanced Baseline v3.0
- Probability validation (sum ≈ 1.0)
- Confidence scoring and market analysis
- Endpoint: `/api/v5/gameweeks/{gw}/predictions`

### Cache Service
- **ISR**: 5-minute revalidation for predictions
- **Static**: 1-hour cache for historical data
- **Intelligent**: File modification-based invalidation
- **Next.js**: Integrated with React Query

### Validation Services
- **Coverage**: 10 matches per gameweek validation
- **Integrity**: Atomic file operations, data consistency
- **Format**: v3 schema validation with Zod
- **Anti-leakage**: Temporal validation in ML pipeline

## 🐳 Docker Deployment

### Full Stack via Docker

```bash
# Start everything via Docker
docker-compose up -d

# Or start selectively
docker-compose up -d postgres pgadmin
docker-compose up -d backend frontend
```

### Docker Services
- **postgres**: PostgreSQL 16 with healthchecks
- **pgadmin**: Database administration interface
- **backend**: FastAPI application (optional)
- **frontend**: Next.js application (optional)

### Health Checks
- PostgreSQL: `pg_isready` with 30s start period
- Backend: Depends on healthy PostgreSQL
- Frontend: Depends on backend availability

## 📊 Performance Monitoring

### Metrics Available
- API response times
- Database query performance
- Prediction accuracy tracking
- Cache hit rates
- Error rates and alerts

### Monitoring Endpoints
```bash
# System health
curl http://localhost:8000/api/v1/health/ready

# Performance metrics
curl http://localhost:8000/api/v1/health/metrics

# Pipeline status
curl http://localhost:8000/api/v1/pipeline/status
```

## 🐛 Troubleshooting

### Common Issues

#### Backend won't start
```bash
# Check file permissions
ls -la data/predictions/

# Check backend logs
cd backend && uvicorn main:app --reload --log-level debug

# Verify data directory exists
mkdir -p data/predictions data/fixtures
```

#### Frontend can't reach API
```bash
# Verify backend is running
curl http://localhost:8000/api/v1/health/live

# Check environment variables
cat frontend/.env.local | grep NEXT_PUBLIC_API_URL

# Clear Next.js cache
cd frontend && rm -rf .next && npm run dev
```

#### File-based storage issues
```bash
# Check data directory permissions
chmod 755 data/
chmod 644 data/predictions/*.json

# Verify prediction files exist
ls -la data/predictions/j*_predictions_v3_*.json

# Check atomic write operations
tail -f backend/logs/structured.log | grep "atomic"
```

#### Optional: PostgreSQL (when activated)
```bash
# Check PostgreSQL status (if using)
docker-compose ps postgres

# View PostgreSQL logs
docker-compose logs postgres

# Test connection (use credentials from .env)
psql -h localhost -U oddsy_user -d oddsy_football -c '\l'
```

#### Ports already in use
```bash
# Check what's using ports
lsof -i :3000,:8000,:5432,:8080

# Stop conflicting services
sudo kill -9 $(lsof -t -i:3000)
```

### Migration Issues

```bash
# Reset database (destructive!)
docker-compose down -v
docker-compose up -d postgres

# Re-run migrations
./scripts/migrate.sh

# Check migration status
docker-compose exec postgres psql -U oddsy_user -d oddsy_football -c '\dt'
```

### Cache Issues

```bash
# Clear all caches
cd frontend && rm -rf .next
docker-compose restart backend

# Invalidate specific endpoints
curl -X POST http://localhost:8000/api/revalidate?secret=dev_secret_key
```

## 🧪 Testing & Validation

### Backend Tests
```bash
cd backend
python -m pytest tests/ -v
```

### Frontend Tests
```bash
cd frontend
npm test
npm run type-check
```

### Production Build Test
```bash
# Test production build
cd frontend
npm run build
npm start

# Test backend in production mode
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### Smoke Tests
```bash
# API smoke test
curl http://localhost:8000/api/v1/health/live

# Frontend smoke test
curl http://localhost:3000 | grep "Oddsy"

# Database smoke test
PGPASSWORD=oddsy_password psql -h localhost -U oddsy_user -d oddsy_football -c 'SELECT NOW();'
```

## 📁 Project Structure (Post-Cleanup)

```
/prod/
├── backend/                    # FastAPI application (Python 3.11+)
│   ├── api/v1/                # Health, pipeline operations
│   ├── api/v5/                # Predictions, gameweeks
│   ├── services/              # Business logic services
│   ├── core/                  # Configuration, exceptions
│   ├── middleware/            # Rate limiting, logging
│   ├── requirements.txt       # Python dependencies
│   └── main.py               # FastAPI entry point
├── frontend/                   # Next.js 14 application
│   ├── app/                  # App Router pages
│   ├── components/           # React components
│   ├── hooks/                # Custom React hooks
│   ├── lib/                  # API clients, utilities
│   ├── package.json          # Node.js dependencies
│   └── tailwind.config.js    # Styling configuration
├── data/                       # File-based storage (current)
│   ├── EPL_25_26_Full_Calendar.csv
│   └── j*_predictions_v3_*.json
├── config/                     # ML pipeline configuration
│   ├── features.json
│   └── team_mappings.json
├── database/                   # PostgreSQL schemas (future)
├── scripts/                    # Automation tools
├── ARCHITECTURE.md             # 📊 Detailed technical architecture
├── docker-compose.yml          # Optional PostgreSQL
├── .env.example               # Environment template
└── README.md                  # This file (current state)
```

### 📊 For Detailed Architecture
See **[ARCHITECTURE.md](./ARCHITECTURE.md)** for comprehensive Mermaid diagrams and technical specifications.

## 🎯 Key Features

### Real-time Predictions
- Live predictions for current gameweek
- Real kickoff times from EPL calendar
- Dynamic confidence scoring
- Match-by-match analysis

### Advanced Analytics
- Team fortress analysis (home advantage)
- Historical performance tracking
- Model disagreement indicators
- Probability distributions

### Production Ready
- Comprehensive error handling
- Health checks and monitoring
- Graceful degradation
- Cache optimization

### User Experience
- Responsive design (mobile-first)
- Real-time updates
- Interactive match cards
- Performance dashboard

## 📈 Model Validation

### Cross-Validation Results
```
Enhanced Baseline v3.0 Performance:
✅ Cross-validation accuracy: 53.5%
✅ Real EPL accuracy: 51.3%
✅ Validated on 2280 matches
✅ Confidence intervals: ±5.7%
✅ Anti-data leakage verified
```

### Validation Process
1. **Temporal splits**: No future data contamination
2. **Rolling validation**: Time-series cross-validation
3. **Real EPL testing**: Live match validation
4. **Performance tracking**: Continuous monitoring

## 🔒 Security Considerations

- Environment variables for secrets
- CORS configuration
- Rate limiting (production)
- SQL injection prevention
- Input validation and sanitization

## 📝 License

This project is proprietary software for demonstration purposes.

## 🤝 Contributing

This is a demonstration project. For evaluation purposes only.

---

**For support or questions during evaluation, please refer to this README or check the troubleshooting section above.**