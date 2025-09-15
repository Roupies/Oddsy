# Oddsy - Project Structure

This document describes the reorganized project structure for better development workflow and maintainability.

## 📁 Directory Structure

```
Oddsy/
├── src/                        # Production source code
│   ├── core/                   # Core utilities and infrastructure
│   │   ├── audit_pipeline.py   # Model validation and audit
│   │   ├── metrics_tracker.py  # Performance monitoring
│   │   ├── run_tests.py        # Test orchestration
│   │   └── utils.py            # Common utilities
│   ├── data/                   # Data processing and management
│   │   ├── prepare_ml_data.py  # Main data preparation pipeline
│   │   ├── initialize_season_state.py  # Season initialization
│   │   ├── promoted_teams_analyzer.py  # Championship integration
│   │   └── team_initialization.py      # Team setup utilities
│   ├── models/                 # Model-related code (future)
│   ├── evaluation/             # Model evaluation and validation
│   │   ├── multi_season_backtest_oddsy.py    # Backtesting
│   │   ├── rolling_epl_2025_26_validator.py  # Season validation
│   │   ├── rolling_simulation_oddsy.py       # Simulation engine
│   │   ├── run_rolling_analysis.sh           # Analysis scripts
│   │   ├── business/           # ROI and business logic
│   │   ├── dynamic_validation/ # Validation results
│   │   ├── rolling_validation_2025_26/      # 2025-26 validation
│   │   └── v23_*.md|json       # v23 model reports
│   └── api/                    # API and web interfaces
│       └── live_predictions_pipeline.py     # Prediction API
├── scripts/                    # Development scripts (organized)
│   ├── analysis/               # Data analysis tools
│   ├── data_acquisition/       # Data collection scripts
│   ├── evaluation/             # Model evaluation scripts  
│   ├── modeling/               # Core modeling scripts
│   ├── preprocessing/          # Feature engineering
│   └── validation/             # Validation and testing
├── data/                       # Data storage
│   ├── raw/                    # Original datasets
│   ├── cleaned/                # Processed datasets
│   ├── processed/              # ML-ready datasets
│   ├── external/               # External data sources
│   └── calendars/              # Season calendars
├── models/                     # Production models only
│   ├── v23_retrained_2025_09_11_154613.joblib         # Production model
│   └── v23_retrained_2025_09_11_154613_metadata.json  # Model metadata
├── tests/                      # Test suite
│   ├── test_data_quality.py    # Data validation tests
│   ├── test_no_leakage.py      # Data leakage detection
│   └── scripts/                # Test utilities
├── config/                     # Configuration files
├── evaluation/                 # Historical evaluation reports
├── logs/                       # Application logs
├── predictions/                # Prediction outputs
├── archive/                    # Archived/deprecated code
│   ├── experimental/           # Research experiments
│   │   ├── v24_v30_attempts/   # Failed model versions
│   │   ├── optimization_experiments/  # Optimization trials
│   │   └── feature_experiments/       # Feature engineering tests
│   └── deprecated/             # Old/obsolete code
│       ├── old_models/         # Non-production models
│       ├── old_reports/        # Historical reports
│       ├── old_scripts/        # Deprecated scripts
│       ├── temp_analysis/      # Temporary analysis
│       ├── test_results/       # Old test outputs
│       └── tree_visualizations/ # Model visualizations
└── Documentation Files
    ├── CLAUDE.md               # Project instructions
    ├── Project_Charter_Oddsy.md # Project charter (Stage 2)
    ├── README.md               # Main project readme
    ├── README_PROJECT_STRUCTURE.md # This file
    ├── RAPPORT_INTEGRATION_EPL_2025_26.md  # EPL integration report
    ├── ROLLING_ANALYSIS_FINAL_REPORT.md    # Rolling analysis report
    └── v23_audit_complet_rapport.md        # v23 audit report
```

## 🎯 Key Principles

### Production vs Development
- **`src/`**: Production-ready code, well-tested and documented
- **`scripts/`**: Development tools and utilities  
- **`archive/`**: Experimental or deprecated code

### Clean Separation
- **Core Infrastructure** (`src/core/`): Audit, testing, utilities
- **Data Pipeline** (`src/data/`): Data processing and preparation
- **Model Evaluation** (`src/evaluation/`): Validation and business logic
- **API Layer** (`src/api/`): Web interfaces and prediction services

### Archive Organization
- **Experimental**: Research attempts that didn't make production
- **Deprecated**: Old code replaced by better implementations
- **Clear Categorization**: By type (models, scripts, reports)

## 🚀 Production Model Status

**Current Production Model: v2.3**
- **File**: `models/v23_retrained_2025_09_11_154613.joblib`
- **Metadata**: `models/v23_retrained_2025_09_11_154613_metadata.json`
- **Performance**: 51.06% ± 3.02% (with EPL 2025-26 integration)
- **Features**: 10 validated features
- **Status**: ✅ Production ready, comprehensive audit passed

## 📋 Development Workflow

### For New Features
1. Develop in `scripts/` appropriate subdirectory
2. Test thoroughly with existing test suite
3. Move to `src/` when production-ready
4. Update documentation

### For Model Changes
1. Use `src/core/audit_pipeline.py` for validation
2. Follow temporal validation principles
3. Update model metadata
4. Archive previous versions

### For Data Updates
1. Use `src/data/prepare_ml_data.py` pipeline
2. Validate with `tests/test_data_quality.py`
3. Update feature documentation
4. Verify no data leakage

## 🔧 Key Commands

```bash
# Core validation
python src/core/audit_pipeline.py --data data/processed/v15_final_enhanced.csv --model models/v23_retrained_2025_09_11_154613.joblib

# Data preparation  
python src/data/prepare_ml_data.py

# Testing
python src/core/run_tests.py

# Live predictions
python src/api/live_predictions_pipeline.py

# Evaluation
bash src/evaluation/run_rolling_analysis.sh
```

## 📚 Documentation

- **Project Instructions**: `CLAUDE.md` (comprehensive development guide)
- **Project Charter**: `Project_Charter_Oddsy.md` (objectives and planning)
- **Integration Report**: `RAPPORT_INTEGRATION_EPL_2025_26.md` (EPL 2025-26 integration)
- **Model Audit**: `v23_audit_complet_rapport.md` (production model validation)

---

*This structure supports the transition from research project to production-ready system while maintaining clear separation between experimental and validated code.*