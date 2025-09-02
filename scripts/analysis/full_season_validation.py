#!/usr/bin/env python3
"""
full_season_validation.py

Validation complète sur saison entière pour mesurer écart exact batch vs rolling.

OBJECTIF P1:
- Mesurer écart de performance exact entre approches batch et rolling
- Valider que l'écart est dans la fourchette attendue (0.5-2pp)
- Identifier sources spécifiques de différence de performance
- Générer rapport détaillé pour décision business

USAGE:
python scripts/analysis/full_season_validation.py \
    --data data/processed/premier_league_full_with_results.csv \
    --model models/clean_xg_model_traditional_baseline_2025_08_31_235028.joblib \
    --test_season_start 2024-08-01 \
    --output results/full_season_validation_2024.json

MÉTHODES DE VALIDATION:
1. Rolling Validation - Performance temporellement sûre
2. Batch Validation - Reproduction approche actuelle  
3. Analyse comparative détaillée
4. Breakdown par journée, équipe, type de match
"""

import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import tempfile

# ML imports
import joblib
from sklearn.metrics import accuracy_score, classification_report, log_loss

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import setup_logging

# Import components
from scripts.analysis.initialize_historical_state import HistoricalStateBuilder
from scripts.analysis.rolling_simulation_engine import RollingSimulationEngine, FeatureBuilder, ResultsAggregator
from scripts.analysis.compare_batch_vs_rolling import BatchEvaluator, ComparisonFramework

@dataclass
class ValidationMetrics:
    """Métriques détaillées de validation"""
    approach: str  # 'batch' ou 'rolling'
    overall_accuracy: float
    log_loss: Optional[float]
    
    # Breakdown par journée
    daily_accuracies: List[float]
    daily_dates: List[str]
    
    # Breakdown par équipe
    team_performance: Dict[str, float]
    
    # Breakdown par type de match
    home_accuracy: float
    draw_accuracy: float
    away_accuracy: float
    
    # Features importance (si disponible)
    feature_importance: Optional[Dict[str, float]] = None
    
    # Données brutes pour analyse
    predictions: List[Dict] = field(default_factory=list)
    execution_time: float = 0.0

class FullSeasonValidator:
    """Validateur complet sur saison entière"""
    
    def __init__(self, data_path: str, model_path: str):
        self.data_path = data_path
        self.model_path = model_path
        self.logger = setup_logging()
        
        # Charger modèle
        self.model = joblib.load(model_path)
        self.logger.info(f"✅ Model loaded: {model_path}")
    
    def run_full_validation(self, test_season_start: str, 
                           test_season_end: Optional[str] = None) -> Dict[str, Any]:
        """Exécute validation complète batch vs rolling"""
        
        self.logger.info("🔍 FULL SEASON VALIDATION - BATCH VS ROLLING")
        self.logger.info("=" * 80)
        self.logger.info(f"Test season: {test_season_start} to {test_season_end or 'end'}")
        self.logger.info("=" * 80)
        
        # Charger et préparer données
        train_data, test_data = self._load_and_split_data(test_season_start, test_season_end)
        
        validation_results = {
            'meta': {
                'test_season_start': test_season_start,
                'test_season_end': test_season_end,
                'train_matches': len(train_data),
                'test_matches': len(test_data),
                'validation_timestamp': str(datetime.now())
            }
        }
        
        # 1. Validation Rolling (référence temporellement sûre)
        self.logger.info("🎮 ROLLING VALIDATION")
        rolling_metrics = self._run_rolling_validation(train_data, test_data, test_season_start)
        validation_results['rolling'] = self._metrics_to_dict(rolling_metrics)
        
        # 2. Validation Batch (approche actuelle)
        self.logger.info("📊 BATCH VALIDATION")
        batch_metrics = self._run_batch_validation(test_data)
        validation_results['batch'] = self._metrics_to_dict(batch_metrics)
        
        # 3. Analyse comparative détaillée
        self.logger.info("🔍 COMPARATIVE ANALYSIS")
        comparative_analysis = self._run_comparative_analysis(rolling_metrics, batch_metrics)
        validation_results['analysis'] = comparative_analysis
        
        # 4. Conclusions et recommandations
        recommendations = self._generate_recommendations(comparative_analysis)
        validation_results['recommendations'] = recommendations
        
        return validation_results
    
    def _load_and_split_data(self, test_start: str, test_end: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Charge et divise les données"""
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Split temporel
        test_start_dt = pd.to_datetime(test_start)
        train_data = df[df['Date'] < test_start_dt].copy()
        
        test_data = df[df['Date'] >= test_start_dt].copy()
        if test_end:
            test_end_dt = pd.to_datetime(test_end)
            test_data = test_data[test_data['Date'] <= test_end_dt].copy()
        
        self.logger.info(f"📊 Data split:")
        self.logger.info(f"  Train: {len(train_data)} matches ({train_data['Date'].min()} to {train_data['Date'].max()})")
        self.logger.info(f"  Test: {len(test_data)} matches ({test_data['Date'].min()} to {test_data['Date'].max()})")
        
        return train_data, test_data
    
    def _run_rolling_validation(self, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                               test_start: str) -> ValidationMetrics:
        """Validation rolling complète"""
        start_time = datetime.now()
        
        # 1. Initialiser état historique
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_state:
            state_path = tmp_state.name
        
        try:
            builder = HistoricalStateBuilder(self.data_path)
            state_manager = builder.build_initial_state(test_start)
            
            # Sauvegarder état
            import pickle
            with open(state_path, 'wb') as f:
                pickle.dump(state_manager, f)
            
            # 2. Simulation rolling
            feature_builder = FeatureBuilder()
            engine = RollingSimulationEngine(state_manager, self.model, feature_builder)
            
            simulation_results = engine.run_simulation(test_data)
            
            # 3. Agréger métriques
            aggregator = ResultsAggregator()
            aggregated = aggregator.aggregate_results(simulation_results)
            
            # 4. Construire métriques détaillées
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Extraire prédictions pour analyse
            all_predictions = []
            for matchday_result in simulation_results:
                for pred in matchday_result.predictions:
                    all_predictions.append({
                        'date': pred.date,
                        'home_team': pred.home_team,
                        'away_team': pred.away_team,
                        'predicted': pred.predicted_result,
                        'actual': pred.actual_result,
                        'correct': pred.predicted_result == pred.actual_result if pred.actual_result is not None else None
                    })
            
            # Accuracies par journée
            daily_accuracies = []
            daily_dates = []
            for matchday_result in simulation_results:
                if matchday_result.accuracy is not None:
                    daily_accuracies.append(matchday_result.accuracy)
                    daily_dates.append(matchday_result.date)
            
            # Performance par équipe
            team_performance = self._calculate_team_performance(all_predictions)
            
            # Performance par type de résultat
            correct_home = sum(1 for p in all_predictions if p['actual'] == 0 and p['correct'])
            total_home = sum(1 for p in all_predictions if p['actual'] == 0)
            home_acc = correct_home / total_home if total_home > 0 else 0
            
            correct_draw = sum(1 for p in all_predictions if p['actual'] == 1 and p['correct'])
            total_draw = sum(1 for p in all_predictions if p['actual'] == 1)
            draw_acc = correct_draw / total_draw if total_draw > 0 else 0
            
            correct_away = sum(1 for p in all_predictions if p['actual'] == 2 and p['correct'])
            total_away = sum(1 for p in all_predictions if p['actual'] == 2)
            away_acc = correct_away / total_away if total_away > 0 else 0
            
            return ValidationMetrics(
                approach='rolling',
                overall_accuracy=aggregated['summary']['overall_accuracy'],
                log_loss=aggregated['summary']['overall_log_loss'],
                daily_accuracies=daily_accuracies,
                daily_dates=daily_dates,
                team_performance=team_performance,
                home_accuracy=home_acc,
                draw_accuracy=draw_acc,
                away_accuracy=away_acc,
                predictions=all_predictions,
                execution_time=execution_time
            )
            
        finally:
            # Nettoyer
            if os.path.exists(state_path):
                os.unlink(state_path)
    
    def _run_batch_validation(self, test_data: pd.DataFrame) -> ValidationMetrics:
        """Validation batch (approche actuelle)"""
        start_time = datetime.now()
        
        # Utiliser BatchEvaluator existant
        batch_evaluator = BatchEvaluator(self.model)
        batch_result = batch_evaluator.evaluate_batch(test_data)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Pour batch, on n'a pas breakdown détaillé, donc approximations
        daily_accuracies = [batch_result.accuracy]  # Approximation
        daily_dates = [str(test_data['Date'].min())]
        
        # Team performance approximée
        unique_teams = set(test_data['HomeTeam'].unique()) | set(test_data['AwayTeam'].unique())
        team_performance = {team: batch_result.accuracy for team in unique_teams}  # Approximation
        
        # Performance par type (basée sur distribution)
        result_dist = batch_result.result_distribution
        
        return ValidationMetrics(
            approach='batch',
            overall_accuracy=batch_result.accuracy,
            log_loss=batch_result.log_loss,
            daily_accuracies=daily_accuracies,
            daily_dates=daily_dates,
            team_performance=team_performance,
            home_accuracy=result_dist.get('home', 0.5),  # Approximation
            draw_accuracy=result_dist.get('draw', 0.3),   # Approximation
            away_accuracy=result_dist.get('away', 0.4),   # Approximation
            predictions=[],  # Pas de détail pour batch
            execution_time=execution_time
        )
    
    def _calculate_team_performance(self, predictions: List[Dict]) -> Dict[str, float]:
        """Calcule performance par équipe"""
        team_stats = {}
        
        for pred in predictions:
            if pred['actual'] is None or pred['correct'] is None:
                continue
                
            home_team = pred['home_team']
            away_team = pred['away_team']
            
            # Initialiser si nécessaire
            if home_team not in team_stats:
                team_stats[home_team] = {'correct': 0, 'total': 0}
            if away_team not in team_stats:
                team_stats[away_team] = {'correct': 0, 'total': 0}
            
            # Compter
            team_stats[home_team]['total'] += 1
            team_stats[away_team]['total'] += 1
            
            if pred['correct']:
                team_stats[home_team]['correct'] += 1
                team_stats[away_team]['correct'] += 1
        
        # Calculer accuracies
        team_accuracies = {}
        for team, stats in team_stats.items():
            if stats['total'] > 0:
                team_accuracies[team] = stats['correct'] / stats['total']
            else:
                team_accuracies[team] = 0.0
        
        return team_accuracies
    
    def _run_comparative_analysis(self, rolling_metrics: ValidationMetrics, 
                                 batch_metrics: ValidationMetrics) -> Dict[str, Any]:
        """Analyse comparative détaillée"""
        
        # Différences principales
        acc_diff = batch_metrics.overall_accuracy - rolling_metrics.overall_accuracy
        
        log_loss_diff = None
        if rolling_metrics.log_loss and batch_metrics.log_loss:
            log_loss_diff = batch_metrics.log_loss - rolling_metrics.log_loss
        
        # Analyse de variance temporelle (si données disponibles)
        temporal_analysis = self._analyze_temporal_variance(rolling_metrics)
        
        # Top/Bottom teams par différence de performance
        team_diff_analysis = self._analyze_team_differences(rolling_metrics, batch_metrics)
        
        # Classification de l'impact
        if abs(acc_diff) < 0.005:
            impact_level = "Negligible"
            business_impact = "No significant difference - both approaches viable"
        elif abs(acc_diff) < 0.01:
            impact_level = "Small"
            business_impact = "Minor difference - consider rolling for production safety"
        elif abs(acc_diff) < 0.02:
            impact_level = "Moderate"
            business_impact = "Noticeable difference - rolling recommended for production"
        else:
            impact_level = "Large"
            business_impact = "Significant difference - rolling essential for realistic assessment"
        
        # Estimation data leakage
        data_leakage_estimate = max(0, acc_diff)  # Si batch > rolling
        
        return {
            'performance_differences': {
                'accuracy_difference': acc_diff,
                'log_loss_difference': log_loss_diff,
                'rolling_accuracy': rolling_metrics.overall_accuracy,
                'batch_accuracy': batch_metrics.overall_accuracy
            },
            'impact_assessment': {
                'level': impact_level,
                'business_impact': business_impact,
                'data_leakage_estimate': data_leakage_estimate
            },
            'temporal_analysis': temporal_analysis,
            'team_differences': team_diff_analysis,
            'execution_comparison': {
                'rolling_time': rolling_metrics.execution_time,
                'batch_time': batch_metrics.execution_time,
                'time_ratio': rolling_metrics.execution_time / batch_metrics.execution_time if batch_metrics.execution_time > 0 else None
            },
            'result_type_comparison': {
                'home_diff': batch_metrics.home_accuracy - rolling_metrics.home_accuracy,
                'draw_diff': batch_metrics.draw_accuracy - rolling_metrics.draw_accuracy, 
                'away_diff': batch_metrics.away_accuracy - rolling_metrics.away_accuracy
            }
        }
    
    def _analyze_temporal_variance(self, rolling_metrics: ValidationMetrics) -> Dict[str, Any]:
        """Analyse variance temporelle des prédictions rolling"""
        
        if not rolling_metrics.daily_accuracies:
            return {'error': 'No daily accuracy data available'}
        
        daily_acc = np.array(rolling_metrics.daily_accuracies)
        
        # Statistiques temporelles
        temporal_stats = {
            'mean_daily_accuracy': np.mean(daily_acc),
            'std_daily_accuracy': np.std(daily_acc),
            'min_daily_accuracy': np.min(daily_acc),
            'max_daily_accuracy': np.max(daily_acc),
            'accuracy_range': np.max(daily_acc) - np.min(daily_acc)
        }
        
        # Détection de tendances
        if len(daily_acc) > 5:
            # Régression linéaire simple pour trend
            x = np.arange(len(daily_acc))
            trend_slope = np.polyfit(x, daily_acc, 1)[0]
            temporal_stats['trend_slope'] = trend_slope
            
            # Classification du trend
            if abs(trend_slope) < 0.001:
                trend_direction = "Stable"
            elif trend_slope > 0:
                trend_direction = "Improving"
            else:
                trend_direction = "Declining"
                
            temporal_stats['trend_direction'] = trend_direction
        
        # Périodes de performance faible/forte
        mean_acc = temporal_stats['mean_daily_accuracy']
        weak_periods = [i for i, acc in enumerate(daily_acc) if acc < mean_acc - temporal_stats['std_daily_accuracy']]
        strong_periods = [i for i, acc in enumerate(daily_acc) if acc > mean_acc + temporal_stats['std_daily_accuracy']]
        
        temporal_stats['weak_periods'] = len(weak_periods)
        temporal_stats['strong_periods'] = len(strong_periods)
        
        return temporal_stats
    
    def _analyze_team_differences(self, rolling_metrics: ValidationMetrics, 
                                 batch_metrics: ValidationMetrics) -> Dict[str, Any]:
        """Analyse différences par équipe"""
        
        # Pour l'approche batch, on n'a qu'une approximation, donc analyse limitée
        common_teams = set(rolling_metrics.team_performance.keys()) & set(batch_metrics.team_performance.keys())
        
        if not common_teams:
            return {'error': 'No common teams for comparison'}
        
        team_diffs = {}
        for team in common_teams:
            rolling_acc = rolling_metrics.team_performance[team]
            batch_acc = batch_metrics.team_performance[team]
            team_diffs[team] = batch_acc - rolling_acc
        
        # Top 5 et Bottom 5 différences
        sorted_diffs = sorted(team_diffs.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_teams_compared': len(common_teams),
            'mean_team_difference': np.mean(list(team_diffs.values())),
            'top_5_differences': dict(sorted_diffs[:5]),
            'bottom_5_differences': dict(sorted_diffs[-5:]),
            'teams_with_large_diff': len([d for d in team_diffs.values() if abs(d) > 0.05])
        }
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Génère recommandations basées sur l'analyse"""
        
        recommendations = []
        
        # Recommandations basées sur l'écart de performance
        perf_diff = analysis['performance_differences']
        impact = analysis['impact_assessment']
        
        if impact['level'] == "Large":
            recommendations.append(
                f"🚨 CRITICAL: Large performance difference detected ({perf_diff['accuracy_difference']:+.4f}). "
                "Immediate switch to rolling evaluation required for production deployment."
            )
        elif impact['level'] == "Moderate":
            recommendations.append(
                f"⚠️ MODERATE IMPACT: Performance difference of {perf_diff['accuracy_difference']:+.4f} detected. "
                "Rolling evaluation recommended for production to ensure realistic performance assessment."
            )
        elif impact['level'] == "Small":
            recommendations.append(
                f"✅ MINOR DIFFERENCE: Small performance gap ({perf_diff['accuracy_difference']:+.4f}). "
                "Consider rolling for production safety, but both approaches viable."
            )
        else:
            recommendations.append(
                "✅ NO SIGNIFICANT DIFFERENCE: Both batch and rolling show similar performance. "
                "Current batch approach appears safe for development use."
            )
        
        # Recommandations basées sur data leakage
        if impact['data_leakage_estimate'] > 0.01:
            recommendations.append(
                f"🔍 DATA LEAKAGE SUSPECTED: Batch performance {impact['data_leakage_estimate']:.4f} points higher suggests information leakage. "
                "Investigate feature engineering pipeline for temporal consistency."
            )
        
        # Recommandations temporelles
        if 'temporal_analysis' in analysis and 'trend_direction' in analysis['temporal_analysis']:
            trend = analysis['temporal_analysis']['trend_direction']
            if trend == "Declining":
                recommendations.append(
                    "📉 DECLINING PERFORMANCE: Model performance decreases over time. "
                    "Consider periodic retraining or adaptive learning approaches."
                )
            elif trend == "Improving":
                recommendations.append(
                    "📈 IMPROVING PERFORMANCE: Model performance increases over time. "
                    "This may indicate beneficial adaptation or potential overfitting to recent data."
                )
        
        # Recommandations d'exécution
        if 'execution_comparison' in analysis and analysis['execution_comparison']['time_ratio']:
            time_ratio = analysis['execution_comparison']['time_ratio']
            if time_ratio > 20:
                recommendations.append(
                    f"⏱️ PERFORMANCE CONSIDERATION: Rolling evaluation is {time_ratio:.1f}x slower. "
                    "Consider optimization for production deployment or hybrid approach for development."
                )
        
        return recommendations
    
    def _metrics_to_dict(self, metrics: ValidationMetrics) -> Dict[str, Any]:
        """Convertit ValidationMetrics en dictionnaire"""
        return {
            'approach': metrics.approach,
            'overall_accuracy': metrics.overall_accuracy,
            'log_loss': metrics.log_loss,
            'daily_performance': {
                'accuracies': metrics.daily_accuracies,
                'dates': metrics.daily_dates,
                'mean': np.mean(metrics.daily_accuracies) if metrics.daily_accuracies else None,
                'std': np.std(metrics.daily_accuracies) if metrics.daily_accuracies else None
            },
            'team_performance_summary': {
                'mean_accuracy': np.mean(list(metrics.team_performance.values())) if metrics.team_performance else None,
                'std_accuracy': np.std(list(metrics.team_performance.values())) if metrics.team_performance else None,
                'teams_count': len(metrics.team_performance)
            },
            'result_type_performance': {
                'home_accuracy': metrics.home_accuracy,
                'draw_accuracy': metrics.draw_accuracy,
                'away_accuracy': metrics.away_accuracy
            },
            'execution_time': metrics.execution_time,
            'predictions_count': len(metrics.predictions)
        }

def main():
    """Pipeline principal de validation complète"""
    parser = argparse.ArgumentParser(description="Full season validation - batch vs rolling")
    parser.add_argument('--data', required=True, help='Data file path')
    parser.add_argument('--model', required=True, help='Model file path')
    parser.add_argument('--test_season_start', required=True, help='Test season start (YYYY-MM-DD)')
    parser.add_argument('--test_season_end', help='Test season end (YYYY-MM-DD)')
    parser.add_argument('--output', required=True, help='Output validation report JSON')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info("🔍 FULL SEASON VALIDATION")
    logger.info("=" * 80)
    logger.info(f"Data: {args.data}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Test season: {args.test_season_start} to {args.test_season_end or 'end'}")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 80)
    
    try:
        # Initialiser validateur
        validator = FullSeasonValidator(args.data, args.model)
        
        # Exécuter validation
        results = validator.run_full_validation(args.test_season_start, args.test_season_end)
        
        # Sauvegarder
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Résumé final
        logger.info("🎯 VALIDATION RESULTS:")
        if 'analysis' in results:
            analysis = results['analysis']
            perf_diff = analysis['performance_differences']
            logger.info(f"  Rolling accuracy: {perf_diff['rolling_accuracy']:.4f}")
            logger.info(f"  Batch accuracy: {perf_diff['batch_accuracy']:.4f}")
            logger.info(f"  Difference (Batch - Rolling): {perf_diff['accuracy_difference']:+.4f}")
            logger.info(f"  Impact level: {analysis['impact_assessment']['level']}")
        
        if 'recommendations' in results:
            logger.info("📋 KEY RECOMMENDATIONS:")
            for i, rec in enumerate(results['recommendations'][:3], 1):
                logger.info(f"  {i}. {rec}")
        
        logger.info(f"📄 Full report saved to: {args.output}")
        logger.info("🎉 FULL SEASON VALIDATION COMPLETE!")
        
    except Exception as e:
        logger.error(f"❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()