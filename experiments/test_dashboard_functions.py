#!/usr/bin/env python3
"""
🧪 Test script pour vérifier les fonctions du dashboard
Vérifie que toutes les fonctions principales fonctionnent sans erreur
"""

import sys
import os
from pathlib import Path

# Add dashboard path
dashboard_path = Path(__file__).parent / "dashboards"
sys.path.insert(0, str(dashboard_path))

try:
    print("🧪 Test des imports du dashboard...")
    
    # Test robust_data_loader
    print("📊 Test robust_data_loader...")
    from core.robust_data_loader import (
        load_unified_metrics, 
        get_production_predictions,
        get_model_comparison_data,
        get_epl_2025_26_matches
    )
    print("✅ robust_data_loader: OK")
    
    # Test fonctions
    print("\n🔧 Test des fonctions...")
    
    # Test unified metrics
    metrics = load_unified_metrics()
    print(f"📈 Metrics status: {metrics.get('data_status', 'unknown')}")
    
    # Test predictions
    predictions = get_production_predictions()
    print(f"🔮 Predictions: {len(predictions)} items")
    
    # Test model comparison
    comparison = get_model_comparison_data()
    print(f"⚖️ Model comparison: {len(comparison)} models")
    
    # Test EPL matches
    epl_matches = get_epl_2025_26_matches()
    print(f"⚽ EPL 2025-26: {len(epl_matches)} matches")
    
    print("\n✅ Tous les tests passent! Le dashboard devrait fonctionner.")
    
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
except Exception as e:
    print(f"❌ Erreur: {e}")
    import traceback
    traceback.print_exc()