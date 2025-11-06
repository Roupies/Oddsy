# Oddsy Production Architecture - High Level Overview

```mermaid
graph TB
    %% === USER LAYER ===
    subgraph "👥 User Layer"
        USER[🧑‍💻 Users<br/>Web Browser]
        MOBILE[📱 Mobile Users<br/>Responsive UI]
    end

    %% === FRONTEND LAYER ===
    subgraph "🎨 Frontend Layer (Next.js 14)"
        direction TB
        
        subgraph "📱 Client Application"
            NEXTJS[⚛️ Next.js 14 App Router<br/>React 18 + TypeScript]
            COMPONENTS[🧩 Component Library<br/>• Hero Sections<br/>• Match Cards<br/>• Prediction UI<br/>• Stadium Backgrounds]
            HOOKS[🪝 Custom Hooks<br/>• useSmartHeroVideo<br/>• usePredictions<br/>• useCurrentGameweek]
        end
        
        subgraph "🎯 State Management"
            REACTQUERY[📊 React Query v5<br/>Data Fetching & Caching]
            APICLIENT[🌐 API Client<br/>Type-safe with Zod validation]
        end
        
        subgraph "🎨 Styling System"
            TAILWIND[🎨 Tailwind CSS<br/>EPL Brand Colors + Animations]
            POSTCSS[⚙️ PostCSS<br/>Processing Pipeline]
        end
    end

    %% === API GATEWAY ===
    subgraph "🌉 API Layer (FastAPI)"
        direction TB
        
        subgraph "🛡️ Middleware Stack"
            CORS[🔀 CORS Middleware]
            RATEHIMIT[⏱️ Rate Limiting<br/>Production Grade]
            LOGGING[📝 Structured Logging<br/>JSON + Correlation IDs]
            GZIP[📦 GZip Compression]
        end
        
        subgraph "🚀 API Endpoints"
            V1API[📍 API v1<br/>• /predictions/j{round}<br/>• /pipeline/*<br/>• /health/*<br/>• /operations/*]
            V5API[📍 API v5<br/>• /gameweeks/*<br/>• Enhanced metadata]
        end
        
        subgraph "🔧 Core Services"
            PIPELINE_INT[⚙️ Pipeline Interface<br/>Data access layer]
            ATOMIC_WRITER[💾 Atomic Writer<br/>Crash-safe file ops]
            CACHE_SVC[⚡ Cache Service<br/>Redis-compatible]
            METRICS[📊 Production Metrics<br/>Performance monitoring]
        end
    end

    %% === ML PIPELINE LAYER ===
    subgraph "🤖 ML Pipeline Layer"
        direction TB
        
        subgraph "⚽ Data Ingestion"
            UNDERSTAT[📈 Understat API<br/>Match statistics]
            FOOTBALL_DATA[⚽ Football-data.co.uk<br/>E0 league data]
            REAL_ODDS[💰 Real Odds Integration<br/>Betting market data]
        end
        
        subgraph "🧠 ML Models"
            ENSEMBLE[🎯 Ensemble System<br/>• Enhanced Baseline v2.4<br/>• Cascade v2.1 Optimized<br/>• XGBoost + RandomForest]
            MODEL_VALIDATION[✅ Model Validation<br/>• Probability validation<br/>• Coverage validation<br/>• Performance tracking]
        end
        
        subgraph "⚙️ Pipeline Orchestration"
            MAIN_PIPELINE[🏭 Production Pipeline<br/>run_pipeline.py]
            GAMEWEEK_GEN[📅 Gameweek Generator<br/>gameweek_predictions_production.py]
            JOB_MANAGER[👷 Job Manager<br/>Background processing]
        end
    end

    %% === DATA LAYER ===
    subgraph "💾 Data Layer"
        direction TB
        
        subgraph "📁 File Storage"
            PREDICTIONS_JSON[📄 Predictions JSON<br/>j{round}_predictions_v3_*.json]
            CONFIG_FILES[⚙️ Configuration<br/>• features.json<br/>• team_mappings.json<br/>• target_mapping.json]
            CALENDAR_CSV[📅 EPL Calendar<br/>EPL_25_26_Full_Calendar.csv]
        end
        
        subgraph "🗃️ Processed Data"
            ML_READY[🧮 ML Ready Dataset<br/>premier_league_ml_ready.csv]
            FEATURE_STORE[🏪 Feature Store<br/>Enhanced features v12+]
        end
    end

    %% === INFRASTRUCTURE ===
    subgraph "☁️ Infrastructure & DevOps"
        direction TB
        
        subgraph "📦 Container Layer"
            DOCKER_FRONTEND[🐳 Frontend Container<br/>Next.js Production Build]
            DOCKER_BACKEND[🐳 Backend Container<br/>FastAPI + Uvicorn]
            DOCKER_ML[🐳 ML Container<br/>Pipeline + Models]
        end
        
        subgraph "🔧 Production Services"
            NGINX[🌐 Nginx Reverse Proxy<br/>Load balancing + SSL]
            REDIS[⚡ Redis Cache<br/>Session & API cache]
            MONITORING[📊 Monitoring Stack<br/>Logs + Metrics + Alerts]
        end
    end

    %% === EXTERNAL SERVICES ===
    subgraph "🌍 External Services"
        EPL_API[⚽ Premier League API<br/>Official fixture data]
        WEATHER_API[🌤️ Weather Services<br/>Match conditions]
        BETTING_APIS[💰 Betting APIs<br/>Market odds comparison]
    end

    %% === FLOW CONNECTIONS ===
    USER --> NEXTJS
    MOBILE --> NEXTJS
    
    NEXTJS --> COMPONENTS
    NEXTJS --> HOOKS
    HOOKS --> REACTQUERY
    REACTQUERY --> APICLIENT
    APICLIENT --> CORS
    
    CORS --> RATEHIMIT
    RATEHIMIT --> LOGGING  
    LOGGING --> V1API
    LOGGING --> V5API
    
    V1API --> PIPELINE_INT
    V5API --> PIPELINE_INT
    PIPELINE_INT --> ATOMIC_WRITER
    PIPELINE_INT --> CACHE_SVC
    
    MAIN_PIPELINE --> GAMEWEEK_GEN
    GAMEWEEK_GEN --> ENSEMBLE
    ENSEMBLE --> MODEL_VALIDATION
    MODEL_VALIDATION --> PREDICTIONS_JSON
    
    UNDERSTAT --> MAIN_PIPELINE
    FOOTBALL_DATA --> MAIN_PIPELINE
    REAL_ODDS --> MAIN_PIPELINE
    
    PREDICTIONS_JSON --> PIPELINE_INT
    CONFIG_FILES --> ENSEMBLE
    ML_READY --> ENSEMBLE
    
    NEXTJS --> DOCKER_FRONTEND
    V1API --> DOCKER_BACKEND  
    MAIN_PIPELINE --> DOCKER_ML
    
    DOCKER_FRONTEND --> NGINX
    DOCKER_BACKEND --> NGINX
    
    EPL_API --> UNDERSTAT
    WEATHER_API --> MAIN_PIPELINE
    BETTING_APIS --> REAL_ODDS

    %% === STYLING ===
    classDef userLayer fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef frontendLayer fill:#f3e5f5,stroke:#4a148c,stroke-width:2px  
    classDef apiLayer fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef mlLayer fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef dataLayer fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef infraLayer fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    classDef externalLayer fill:#fff8e1,stroke:#ff6f00,stroke-width:2px

    class USER,MOBILE userLayer
    class NEXTJS,COMPONENTS,HOOKS,REACTQUERY,APICLIENT,TAILWIND,POSTCSS frontendLayer
    class CORS,RATEHIMIT,LOGGING,GZIP,V1API,V5API,PIPELINE_INT,ATOMIC_WRITER,CACHE_SVC,METRICS apiLayer
    class UNDERSTAT,FOOTBALL_DATA,REAL_ODDS,ENSEMBLE,MODEL_VALIDATION,MAIN_PIPELINE,GAMEWEEK_GEN,JOB_MANAGER mlLayer
    class PREDICTIONS_JSON,CONFIG_FILES,CALENDAR_CSV,ML_READY,FEATURE_STORE dataLayer
    class DOCKER_FRONTEND,DOCKER_BACKEND,DOCKER_ML,NGINX,REDIS,MONITORING infraLayer
    class EPL_API,WEATHER_API,BETTING_APIS externalLayer
```

## 📋 Architecture Summary

### 🎯 **Core Technology Stack**
- **Frontend**: Next.js 14 + React 18 + TypeScript + Tailwind CSS
- **Backend**: FastAPI + Python 3.9+ + Pydantic v2
- **ML Pipeline**: scikit-learn + XGBoost + LightGBM + Custom ensemble
- **Database**: File-based JSON + CSV (production-ready atomic writes)
- **Caching**: Redis-compatible service layer
- **Infrastructure**: Docker containers + Nginx + Production monitoring

### 🌊 **Data Flow**
1. **External APIs** → **ML Pipeline** → **Predictions JSON**
2. **API Layer** → **File Storage** → **Frontend Cache** → **User Interface**
3. **Real-time updates** via **polling service** + **frontend revalidation**

### 🔒 **Production Features**
- **Atomic file operations** for crash safety
- **Structured JSON logging** with correlation IDs  
- **Rate limiting** with client IP tracking
- **Type-safe API** with Zod validation
- **Performance monitoring** and metrics collection
- **Graceful error handling** and fallback strategies

### 🏗️ **Scalability Design**
- **Stateless API** design for horizontal scaling
- **Caching layers** at multiple levels (browser, API, file system)
- **Container-ready** for cloud deployment
- **Background job processing** for heavy ML operations
- **CDN-friendly** static assets and API responses

### ⚽ **Football-Specific Features**
- **Premier League** data integration (official + community sources)
- **Multi-model ensemble** for prediction accuracy
- **Gameweek-based** prediction lifecycle
- **Stadium visuals** and team branding integration
- **Real betting odds** comparison and market analysis