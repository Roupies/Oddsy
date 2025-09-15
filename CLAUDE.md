# CLAUDE.md

This file provides guidance to Claude Code when working with the **Oddsy** repository.

## Project Overview

**Oddsy** is a Premier League football match prediction project that uses machine learning to predict Home/Draw/Away results. The project focuses on sophisticated feature engineering and data analysis of Premier League matches from 2019-2024.

## Project Objectives

**Primary Goal:** Build a machine learning model that predicts football match outcomes (H/D/A) better than naive baselines.

**Performance Targets:**
- **Minimum Acceptable:** > 43.6% accuracy (beat majority class baseline)
- **Good Model:** > 50% accuracy  
- **Excellent Model:** > 55% accuracy (industry-competitive)

**Key Baselines to Beat:**
- Random prediction (33/33/33): 33.3%
- Always predict Home: 43.6% 
- Weighted random by distribution: 35.4%

**Business Context:** Real Premier League distribution is naturally imbalanced (H: 43.6%, A: 33.4%, D: 23.0%) - this reflects actual football dynamics, not a data quality issue.

## Architecture & Data Pipeline

The project follows a clear data processing pipeline:

1. **Raw Data** (`data/raw/`) - Original Premier League CSV files
2. **Cleaned Data** (`data/cleaned/`) - Processed and cleaned datasets
3. **Processed Data** (`data/processed/`) - Feature-engineered datasets ready for ML
4. **Models** (`models/`) - Production model with metadata
5. **Evaluation** (`evaluation/`) - Performance reports & metrics history

## Current Production Status - v15 EPL 2025-26 INTEGRATED (September 14, 2025)

### **🎯 PRODUCTION MODEL: v2.3 with EPL 2025-26 - 51.06% VALIDATED ACCURACY**

**✅ EPL 2025-26 INTEGRATION COMPLETE** - Real-world season data successfully integrated with promoted teams

**Production Model (v2.3 + EPL 2025-26):**
- **Algorithm:** RandomForest with calibration (10 optimized features)
- **Current Performance:** 51.06% ± 3.02% (cross-validation with EPL 2025-26 included)
- **Base Performance:** 52.11% ± 3.46% (original v2.3 on historical data)
- **Audit Status:** ✅ PRODUCTION READY (comprehensive audit passed)
- **Model File:** `models/v23_retrained_2025_09_11_154613.joblib`
- **Dataset:** `data/processed/v15_final_enhanced.csv` (2310 matches)
- **EPL 2025-26:** 30 matches with 100% real xG coverage
- **Promoted Teams:** Leeds (Elo 1591), Sunderland (Elo 1398) intelligently initialized

### **🔬 v2.3 Production Feature Set (10 Validated Features)**

**Core Features (Production Validated):**
1. **`elo_diff_normalized`** - Team strength difference (core predictor)
2. **`market_entropy_norm`** - Market uncertainty signal  
3. **`shots_diff_normalized`** - Shot differential proxy
4. **`corners_diff_normalized`** - Pressure/possession proxy
5. **`form_diff_normalized`** - Recent form difference
6. **`h2h_score`** - Head-to-head historical performance
7. **`matchday_normalized`** - Season progression context
8. **`home_xg_eff_10`** - Home team xG efficiency (10-match window)
9. **`away_xg_eff_10`** - Away team xG efficiency (10-match window)
10. **`away_goals_sum_5`** - Away team scoring form (5-match sum)

**Feature Balance:**
- **Traditional Features (70%):** Elo, form, H2H, shots, corners, matchday
- **Market Intelligence (10%):** Betting market entropy  
- **xG Efficiency (20%):** Clean temporal xG efficiency metrics

### **📊 v2.3 Comprehensive Audit Results**

**Audit Pipeline Results (September 12, 2025):**
- **Cross-Validation Accuracy:** 52.11% ± 3.46% (temporal splits, realistic performance)
- **Robustness Check:** Perfect stability (0.0000% variance across seeds)
- **Baseline Comparisons:** Beats all naive baselines significantly
- **Data Validation:** Clean temporal features, no data leakage detected
- **Calibration:** Well-calibrated probability outputs
- **Production Readiness:** ✅ All criteria satisfied

**Performance vs Targets:**
- ✅ **Random Baseline (33.3%):** +18.8pp  
- ✅ **Majority Class (43.6%):** +8.5pp
- ✅ **Good Target (50%):** +2.1pp (EXCEEDED!)
- 🎯 **Excellent Target (55%):** -2.9pp (close, but not achieved)

## Strategic Decision: Global Accuracy Over Draw Specialization

**CRITICAL STRATEGIC CHOICE MADE:** The project prioritized **global accuracy (52.11%)** over specialized draw prediction capabilities.

**Trade-off Analysis:**
- **Option A:** Optimize for overall match prediction accuracy → **CHOSEN**
- **Option B:** Specialize in draw prediction with cascade models → **REJECTED**

**Rationale:**
1. **Business Value:** Consistent 52% accuracy across all match types provides stable foundation
2. **Simplicity:** Single model easier to maintain and deploy than complex cascades  
3. **Reliability:** Proven stability and validation across rigorous audit
4. **Market Reality:** Draw prediction remains inherently difficult (~23% frequency)

## Development History & Research Archive

### **✅ PRODUCTION PATH (What Worked):**

**v1.0-v1.3: Foundation Building**
- v1.0: 50.0% (basic features, Elo bug discovered)
- v1.1: 51.2% (cleaned redundant features)
- v1.2: 52.2% (hyperparameter optimization)
- v1.3: 53.05% (market intelligence integration)

**v2.1: Clean xG Integration**
- Clean temporal xG features (54.2% accuracy)
- Critical data leakage detection and resolution
- Established temporal safety principles

**v2.3: Production Optimization** 
- **52.11% validated performance** (realistic CV measurement)
- Feature selection refinement (27 → 10 features)
- **Comprehensive audit validation**

**v15: EPL 2025-26 Integration (September 2025)**
- **51.06% ± 3.02% performance** (30 matches EPL 2025-26 added)
- **100% real xG coverage** for EPL 2025-26 via UnderstatAPI
- **Smart promoted teams initialization** (Leeds, Sunderland from Championship)
- **60 future matches available** for predictions via TheSportsDB API

### **📁 EXPERIMENTAL ATTEMPTS (Archived):**

**Post-v2.3 Research (September 2025):**
- **v3.x Efficiency Features:** Marginal/no improvement over v2.3
- **v4.1 Referee Features:** Failed to deliver claimed performance (54.21% vs claimed 58.30%)
- **v2.2 27-Features:** Good test performance but validation gaps
- **Cascade Models:** Complex architectures without validated improvement
- **Various Experimental Features:** Player data, fatigue, momentum (archived as research)

**Key Research Insights:**
- **Feature Quality > Quantity:** 10 optimized features outperformed 27+ feature sets
- **Validation Critical:** Many promising results failed rigorous temporal validation
- **Simplicity Wins:** Complex architectures couldn't reliably beat simple RandomForest

## Professional Audit Infrastructure

### **🔬 audit_pipeline.py - Production Validation Tool**

**Comprehensive Model Auditing:**
```bash
python audit_pipeline.py --data data/processed/v15_final_enhanced.csv \
                        --model models/v23_retrained_2025_09_11_154613.joblib \
                        --target FullTimeResult \
                        --features elo_diff_normalized market_entropy_norm [...]
```

**Audit Capabilities:**
1. **Reproducibility:** Dataset hashing, library versions, system info
2. **Feature Validation:** Data leakage detection, missing values, consistency
3. **Temporal Validation:** TimeSeriesSplit cross-validation (5 folds)
4. **Performance Metrics:** Accuracy, balanced accuracy, log-loss, calibration
5. **Robustness Testing:** Multi-seed retraining (variance analysis)
6. **Baseline Comparisons:** Majority class, random baseline benchmarks
7. **Professional Reporting:** JSON reports + summary + visualizations

**Audit Requirements for Production:**
- Cross-validation accuracy > 50%
- Robustness variance < 1%
- Beats all naive baselines
- No data leakage detected
- Proper temporal validation

## Essential Production Commands

### **Model Validation & Audit:**
```bash
# Run comprehensive model audit
python audit_pipeline.py --data [dataset] --model [model] --target FullTimeResult

# Core infrastructure
python run_tests.py                    # Quality assurance
python metrics_tracker.py --report    # Performance tracking
python utils.py                       # Utility functions
```

### **Data Processing:**
```bash
python prepare_ml_data.py             # Versioned data preparation
```

## Data Integrity & Production Standards

### **Critical Validation Principles:**
- **No Data Leakage:** All features calculated using ONLY historical data
- **Temporal Integrity:** Proper time-series cross-validation (TimeSeriesSplit)
- **Feature Safety:** All features use shift(1) for rolling averages
- **Audit Required:** All models must pass `audit_pipeline.py` validation
- **Version Control:** All datasets and models versioned with metadata

### **Production Dataset Standards:**
- **Current Production Dataset:** `data/processed/v15_final_enhanced.csv`
- **Temporal Coverage:** 2019-2020 through 2025-26 (2,310 matches)
- **EPL 2025-26 Integration:** 30 matches with 100% real xG coverage
- **Feature Count:** 15 available features (10 used in production)
- **Target Encoding:** H→0, D→1, A→2
- **Validation:** Hash-verified, temporally clean, leakage-free
- **Promoted Teams:** Leeds, Sunderland intelligently initialized from Championship data

### **Model Standards:**
- **Architecture:** RandomForest with CalibratedClassifierCV
- **Features:** Exactly 10 validated features (see production list above)
- **Performance:** Cross-validated accuracy > 50%
- **Stability:** Seed variance < 1%
- **Documentation:** Complete metadata JSON with audit results

## Archive Organization

```
archive/
├── deprecated_claude_versions/     # Previous documentation versions
├── models_invalid_validation/      # Models that failed rigorous audit
├── scripts_experimental/          # Post-v2.3 research attempts  
├── root_cleanup_2025_09_12/       # Organized analysis files
└── failed_experiments/           # Feature experiments (referee, efficiency, etc.)
```

## Target Encoding & Evaluation

**Target Variable:** `FullTimeResult` → `target`
- `0` = Home win (H) - 43.6% of matches
- `1` = Draw (D) - 23.0% of matches  
- `2` = Away win (A) - 33.4% of matches

**Evaluation Strategy:**
- **Primary Metric:** Cross-validated accuracy (TimeSeriesSplit)
- **Validation:** Temporal integrity maintained
- **Audit Required:** All models must pass comprehensive audit

## Dependencies & Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

**Core Libraries:**
- pandas, numpy - Data manipulation
- scikit-learn - ML algorithms and validation  
- matplotlib - Visualizations and calibration plots
- joblib - Model serialization

## Production Status Summary

**Current State:** 
- ✅ **Production Model:** v2.3 (52.11% validated accuracy)
- ✅ **Audit Infrastructure:** Comprehensive `audit_pipeline.py`
- ✅ **Clean Organization:** Experimental code archived
- ✅ **Professional Documentation:** Honest, validated performance claims
- ✅ **Strategic Clarity:** Global accuracy prioritized over draw specialization

**Key Achievement:** Transformed from research project with scattered experiments into production-ready system with rigorous validation and honest performance documentation.

## EPL 2025-26 Season Integration & Prediction System

### **📅 Complete EPL 2025-26 Calendar**

**✅ FULL SEASON COVERAGE ACHIEVED**
- **Total Matches:** 380 complete matches (full EPL season)
- **Calendar Source:** Official EPL calendar (`data/raw/epl-2025-2026_GMTStandardTime.csv`)
- **Season Timeline:** August 15, 2025 → May 24, 2026
- **Current Status:** 30 matches played, 350 matches remaining for predictions

**Promoted Teams Successfully Integrated:**
- **Leeds United** - Initialized from Championship performance data
- **Sunderland** - Initialized from Championship performance data  
- **Burnley** - Returning to EPL (previously relegated)

### **🔮 Live Prediction System**

**Production-Ready Match Predictions:**
```bash
# Generate predictions for upcoming EPL 2025-26 matches
python scripts/modeling/v23_live_predictions.py --calendar data/raw/epl-2025-2026_GMTStandardTime.csv \
                                                 --model models/v23_retrained_2025_09_11_154613.joblib \
                                                 --features data/processed/v15_final_enhanced.csv
```

**Prediction Capabilities:**
- **Next Match Predictions:** Real-time predictions for upcoming gameweek
- **Full Season Outlook:** 350+ future match predictions available
- **Confidence Intervals:** Calibrated probabilities for H/D/A outcomes
- **Feature Coverage:** 100% feature coverage for all EPL 2025-26 matches
- **Update Frequency:** After each gameweek with latest results

**Prediction Output Format:**
```python
{
    "match_id": "2267115",
    "date": "2025-09-20",
    "home_team": "Brighton",
    "away_team": "Tottenham",
    "predictions": {
        "home_win": 0.42,
        "draw": 0.26, 
        "away_win": 0.32
    },
    "predicted_outcome": "H",
    "confidence": 0.42
}
```

### **📊 EPL 2025-26 Integration Statistics**

**Data Quality Metrics:**
- **Real xG Coverage:** 100% (via UnderstatAPI)
- **Market Data:** 100% entropy signals available
- **Historical H2H:** Complete coverage for all team pairings
- **Elo Ratings:** Updated with all 2025-26 results
- **Feature Completeness:** No missing values across all 10 production features

**Performance Validation:**
- **v15 Performance:** 51.06% ± 3.02% (includes 30 EPL 2025-26 matches)
- **Performance Drop:** -1.05pp vs v2.3 baseline (within expected variance)
- **Temporal Validation:** Proper TimeSeriesSplit maintained
- **Production Status:** ✅ Validated and ready for live predictions

### **🚀 Next Steps & Future Development**

**Immediate Roadmap:**
1. **Live Dashboard:** Real-time match predictions interface
2. **Performance Tracking:** Continuous model performance monitoring
3. **Feature Updates:** Incorporate new EPL 2025-26 insights
4. **Model Retraining:** Periodic retraining with expanding dataset

**Business Applications:**
- **Match Predictions:** Pre-match outcome forecasting
- **Betting Intelligence:** Market opportunity identification
- **Performance Analytics:** Team strength assessment over time
- **Strategic Insights:** League dynamics and trends analysis

---

*Oddsy v2.3 Production - Validated 52.11% accuracy with comprehensive audit infrastructure and professional model management. EPL 2025-26 integration complete with full 380-match calendar and live prediction system ready (September 14, 2025)*