# 🏆 Oddsy - AI Premier League Predictions

**Enhanced Baseline v3.0 with 51.3% accuracy on real EPL data**

## 🚀 Quick Demo Start

```bash
# Clone and run the production demo
git clone [repo-url]
cd oddsy

# Start production services (see SETUP_INSTRUCTIONS.md in /prod for configuration)
cd prod/backend && python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
cd prod/frontend && npm install && npm run dev

# Access: http://localhost:3000
```

## 📁 Project Structure

```
oddsy/                           # Clean 3-directory organization
├── 📦 prod/                     # 🎯 PRODUCTION DEMO (GitHub focus)
│   ├── backend/                 # FastAPI + Enhanced Baseline v3.0
│   ├── frontend/                # Next.js 14 + Tailwind CSS
│   ├── data/                    # Validated prediction data (J11)
│   └── README.md                # Production setup guide
├── 🔧 dev/                      # Development & training (excluded from GitHub)
│   ├── models/                  # ML models and training scripts
│   ├── scripts/                 # Pipeline and validation tools
│   └── data/                    # Raw datasets and experiments
└── 📚 rest/                     # Legacy code archive (excluded from GitHub)
```

## 🎯 Performance Metrics

- **Model:** Enhanced Baseline v3.0 (Random Forest)
- **Accuracy:** 51.3% validated on real EPL matches
- **Pipeline:** v1.0 with strict temporal validation
- **Data:** 2280 matches (2019-2025) + real-time APIs

## 🏗️ Production Stack

- **Frontend:** Next.js 14 + TypeScript + Tailwind CSS
- **Backend:** FastAPI + Enhanced Baseline v3.0 
- **ML Model:** Random Forest with strict temporal validation
- **APIs:** Football Data + API-Football + The Odds API

## 📊 Key Features

✅ **Enhanced Baseline v3.0:** 51.3% accuracy on real EPL data  
✅ **Pipeline v1.0:** Strict temporal validation, anti-data leakage  
✅ **Real-time APIs:** Live fixtures, odds, and team statistics  
✅ **Production UI:** Modern design with stadium backgrounds  
✅ **Clean Architecture:** Organized structure for demo and development  

## 🚀 Demo Instructions

### For Jury Presentation
```bash
# Quick demo setup (see /prod/SETUP_INSTRUCTIONS.md for full configuration)
cd prod/backend && python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
cd prod/frontend && npm run dev

# Access the demo: http://localhost:3000
```

### Key Demo Pages
- **Homepage** - Performance metrics and model overview
- **Latest Predictions** - Current gameweek predictions (J11)
- **Models** - Enhanced Baseline v3.0 performance details
- **Pipeline** - Real-time system status and health checks

## 📈 Project Organization Benefits

### 🎯 `/prod` - Clean Demo Environment
- Zero setup friction for presentation
- Production-ready code only
- All dependencies isolated
- Performance metrics visible

### 🔧 `/dev` - Development Workspace  
- Model training and experimentation
- Historical data and research
- Pipeline development and testing
- Validation and performance analysis

### 📚 `/rest` - Legacy Archive
- Previous iterations and experiments
- Research prototypes
- Historical codebase versions

## 🏆 Technical Achievements

- **51.3% accuracy** validated on real Premier League matches
- **Temporal validation** preventing data leakage
- **Production pipeline** with automated gameweek predictions  
- **Modern UI/UX** with responsive design
- **Clean architecture** separating demo from development

---

**🎯 Focus on `/prod` directory for GitHub demo - complete, tested, presentation-ready**