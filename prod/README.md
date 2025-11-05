# Oddsy - AI Premier League Predictions Platform

> Advanced machine learning platform delivering validated Premier League match predictions with **51.3% accuracy** using Enhanced Baseline v3.0 and Pipeline Durci v1.0.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │───▶│    Backend      │───▶│   PostgreSQL    │
│   Next.js 14   │    │   FastAPI       │    │   Database      │
│   TypeScript    │    │   Python 3.11  │    │   + pgAdmin     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Technology Stack
- **Frontend**: Next.js 14, React, TypeScript, Tailwind CSS
- **Backend**: FastAPI, Python 3.11, async/await
- **Database**: PostgreSQL 16 with pgAdmin 4
- **AI Model**: Enhanced Baseline v3.0 (51.3% accuracy)
- **Pipeline**: Pipeline Durci v1.0 with anti-data leakage
- **Infrastructure**: Docker Compose, healthchecks

## 🚀 Quick Start (TL;DR)

```bash
# 1. Database
docker-compose up -d postgres pgadmin

# 2. Backend  
cd backend && python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# 3. Frontend
cd frontend && npm install && npm run dev
```

**URLs:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/api/system/docs
- Health Check: http://localhost:8000/api/system/health/ready
- pgAdmin: http://localhost:8080 (admin@oddsy.local / admin_password)

## 📋 Prerequisites

- **Docker & Docker Compose** (for PostgreSQL)
- **Python 3.11+** (for backend)
- **Node.js 18+** (for frontend)
- **PostgreSQL client** (optional, for direct DB access)

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

## 🗄️ Database Schema

### Core Tables
- **predictions**: Model predictions with probabilities
- **fixtures**: EPL fixtures with real kickoff times
- **teams**: Team information and statistics

### pgAdmin Access
- URL: http://localhost:8080
- Email: admin@oddsy.local
- Password: admin_password

**Connect to Database:**
- Host: postgres (or localhost if connecting externally)
- Port: 5432
- Database: oddsy_football
- Username: oddsy_user
- Password: oddsy_password

## 🔌 API Documentation

### API Architecture

**Deux surfaces API claires pour le jury :**
- `/api/system` - Health, metrics, pipeline operations
- `/api/gameweeks` - Fixtures, predictions, latest data

*Note: Anciens noms `/api/v1` et `/api/v5` supportés en parallèle pour compatibilité.*

### Core Endpoints

#### Gameweeks & Predictions
```bash
# Get latest gameweek
GET /api/gameweeks/latest

# Get gameweek predictions  
GET /api/gameweeks/{gameweek}/predictions

# Get fixture data with real kickoff times
GET /api/gameweeks/{gameweek}/fixtures
```

#### System & Health
```bash
# Health check
GET /api/system/health/live

# Readiness probe (vérifie DB + répertoires runtime)
GET /api/system/health/ready

# System metrics
GET /api/system/health/metrics

# Pipeline status
GET /api/system/pipeline/status
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

### Fixture Service
- Reads EPL_25_26_Full_Calendar.csv
- Provides real kickoff times in UTC
- Maps team name variations
- Endpoint: `/api/v5/gameweeks/{gw}/fixtures`

### Prediction Service
- Model inference with Enhanced Baseline v3.0
- Probability validation (sum ≈ 1.0)
- Confidence scoring (high/medium/low)
- Endpoint: `/api/v5/gameweeks/{gw}/predictions`

### Cache Service
- 5-minute revalidation for predictions
- 1-hour cache for historical data
- Intelligent cache invalidation
- Next.js ISR integration

### Validation Services
- Coverage validation (10 matches per gameweek)
- Probability validation (probabilities sum to 1.0)
- Data integrity checks
- Anti-leakage verification

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
# Check database connection
docker-compose logs postgres

# Test database connectivity
PGPASSWORD=oddsy_password psql -h localhost -U oddsy_user -d oddsy_football -c '\l'

# Check backend logs
cd backend && uvicorn main:app --reload --log-level debug
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

#### Database connection issues
```bash
# Check PostgreSQL status
docker-compose ps postgres

# View PostgreSQL logs
docker-compose logs postgres

# Restart database
docker-compose restart postgres
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

## 📁 Project Structure

```
/prod/
├── backend/              # FastAPI application
│   ├── api/             # API routes (v1, v5)
│   ├── services/        # Business logic services
│   ├── core/            # Configuration, exceptions
│   ├── requirements.txt # Python dependencies
│   └── Dockerfile       # Backend container
├── frontend/            # Next.js application
│   ├── app/            # App Router pages
│   ├── components/     # React components
│   ├── hooks/          # Custom React hooks
│   ├── lib/            # Utilities and API clients
│   ├── package.json    # Node.js dependencies
│   └── Dockerfile      # Frontend container
├── database/           # Database scripts and migrations
│   └── migration_scripts/
├── data/               # Critical data files
│   ├── EPL_25_26_Full_Calendar.csv
│   └── j11_predictions_*.json
├── scripts/            # Deployment and maintenance
│   ├── setup.sh        # Complete setup automation
│   ├── migrate.sh      # Database migrations
│   └── deploy.sh       # Production deployment
├── docker-compose.yml  # Full stack orchestration
├── .env.example       # Environment template
└── README.md          # This file
```

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