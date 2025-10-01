"""
Feature Fallback Tracker - Monitoring Quality des Données
========================================================
Tracker pour monitorer le pourcentage de features en mode fallback
par journée et identifier la dégradation de qualité des données
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class FeatureFallbackTracker:
    """Tracker global pour monitoring fallback des features"""
    
    def __init__(self):
        self.fallback_log = defaultdict(lambda: defaultdict(list))
        self.matchday_stats = defaultdict(dict)
        self.feature_definitions = {
            'form_diff_normalized': 'Différence de forme normalisée',
            'elo_diff_normalized': 'Différence ELO normalisée',
            'h2h_score': 'Score historique H2H',
            'matchday_normalized': 'Journée normalisée',
            'shots_diff_normalized': 'Différence tirs normalisée',
            'corners_diff_normalized': 'Différence corners normalisée',
            'market_entropy_norm': 'Entropie marché normalisée',
            'home_xg_eff_10': 'Efficacité xG domicile (10 matchs)',
            'away_goals_sum_5': 'Somme buts extérieur (5 matchs)',
            'away_xg_eff_10': 'Efficacité xG extérieur (10 matchs)'
        }
    
    def track_feature_calculation(self, matchday, match_id, feature_name, is_fallback, fallback_reason=None, data_quality=None):
        """
        Enregistre le calcul d'une feature avec son mode (réel/fallback)
        
        Args:
            matchday: Numéro de journée (ex: 'J7')
            match_id: Identifiant unique du match (ex: 'Arsenal_vs_Chelsea_2025-10-05')
            feature_name: Nom de la feature calculée
            is_fallback: True si mode fallback utilisé
            fallback_reason: Raison du fallback (données manquantes, seuil k<3, etc.)
            data_quality: Score qualité données 0-1 si disponible
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'matchday': matchday,
            'match_id': match_id,
            'feature_name': feature_name,
            'is_fallback': is_fallback,
            'fallback_reason': fallback_reason,
            'data_quality': data_quality
        }
        
        self.fallback_log[matchday][feature_name].append(entry)
        
        # Log pour debugging
        status = "FALLBACK" if is_fallback else "REAL"
        reason_str = f" ({fallback_reason})" if fallback_reason else ""
        print(f"   📊 {feature_name}: {status}{reason_str}")
    
    def track_insufficient_data(self, matchday, match_id, feature_name, actual_count, required_count):
        """
        Enregistre un fallback dû à données insuffisantes (k < seuil)
        
        Args:
            matchday: Journée 
            match_id: ID match
            feature_name: Nom feature
            actual_count: Nombre réel de données disponibles
            required_count: Nombre minimum requis (ex: 3)
        """
        fallback_reason = f"Données insuffisantes: {actual_count}/{required_count}"
        
        self.track_feature_calculation(
            matchday, match_id, feature_name, 
            is_fallback=True, 
            fallback_reason=fallback_reason,
            data_quality=actual_count / required_count if required_count > 0 else 0
        )
    
    def track_missing_fbref_data(self, matchday, match_id, feature_name):
        """Enregistre un fallback dû à données FBref manquantes"""
        self.track_feature_calculation(
            matchday, match_id, feature_name,
            is_fallback=True,
            fallback_reason="Données FBref indisponibles",
            data_quality=0.0
        )
    
    def track_successful_calculation(self, matchday, match_id, feature_name, source="real_data"):
        """Enregistre un calcul réussi avec vraies données"""
        self.track_feature_calculation(
            matchday, match_id, feature_name,
            is_fallback=False,
            fallback_reason=None,
            data_quality=1.0
        )
    
    def calculate_matchday_fallback_percentage(self, matchday):
        """
        Calcule le pourcentage de fallback pour une journée donnée
        
        Returns:
            Dict avec stats par feature et globales
        """
        if matchday not in self.fallback_log:
            return None
        
        matchday_data = self.fallback_log[matchday]
        stats = {
            'matchday': matchday,
            'total_features_calculated': 0,
            'total_fallbacks': 0,
            'overall_fallback_percentage': 0.0,
            'by_feature': {},
            'matches_analyzed': set()
        }
        
        # Calculer stats par feature
        for feature_name, entries in matchday_data.items():
            feature_stats = {
                'total_calculations': len(entries),
                'fallback_count': sum(1 for e in entries if e['is_fallback']),
                'fallback_percentage': 0.0,
                'fallback_reasons': defaultdict(int),
                'avg_data_quality': 0.0
            }
            
            if feature_stats['total_calculations'] > 0:
                feature_stats['fallback_percentage'] = (
                    feature_stats['fallback_count'] / feature_stats['total_calculations'] * 100
                )
                
                # Qualité moyenne des données
                qualities = [e['data_quality'] for e in entries if e['data_quality'] is not None]
                if qualities:
                    feature_stats['avg_data_quality'] = np.mean(qualities)
                
                # Raisons de fallback
                for entry in entries:
                    if entry['is_fallback'] and entry['fallback_reason']:
                        feature_stats['fallback_reasons'][entry['fallback_reason']] += 1
                    
                    # Collecter IDs matchs
                    stats['matches_analyzed'].add(entry['match_id'])
            
            stats['by_feature'][feature_name] = feature_stats
            stats['total_features_calculated'] += feature_stats['total_calculations']
            stats['total_fallbacks'] += feature_stats['fallback_count']
        
        # Pourcentage global
        if stats['total_features_calculated'] > 0:
            stats['overall_fallback_percentage'] = (
                stats['total_fallbacks'] / stats['total_features_calculated'] * 100
            )
        
        stats['matches_analyzed'] = len(stats['matches_analyzed'])
        
        # Stocker pour accès rapide
        self.matchday_stats[matchday] = stats
        
        return stats
    
    def get_fallback_trend_analysis(self):
        """
        Analyse les tendances de fallback à travers les journées
        
        Returns:
            Dict avec analyse de tendance
        """
        matchdays = sorted(self.fallback_log.keys())
        
        if len(matchdays) < 2:
            return {'error': 'Pas assez de journées pour analyse de tendance'}
        
        trend_data = []
        
        for matchday in matchdays:
            stats = self.calculate_matchday_fallback_percentage(matchday)
            if stats:
                trend_data.append({
                    'matchday': matchday,
                    'fallback_percentage': stats['overall_fallback_percentage'],
                    'matches_count': stats['matches_analyzed'],
                    'total_features': stats['total_features_calculated']
                })
        
        if not trend_data:
            return {'error': 'Aucune donnée pour analyse de tendance'}
        
        # Calculer tendance
        percentages = [d['fallback_percentage'] for d in trend_data]
        
        analysis = {
            'matchdays_analyzed': len(trend_data),
            'trend_data': trend_data,
            'avg_fallback_percentage': np.mean(percentages),
            'min_fallback_percentage': min(percentages),
            'max_fallback_percentage': max(percentages),
            'fallback_std': np.std(percentages),
            'trend_direction': 'stable'
        }
        
        # Déterminer direction tendance
        if len(percentages) >= 3:
            recent_avg = np.mean(percentages[-3:])
            early_avg = np.mean(percentages[:3])
            
            if recent_avg > early_avg + 5:
                analysis['trend_direction'] = 'degrading'
            elif recent_avg < early_avg - 5:
                analysis['trend_direction'] = 'improving'
        
        return analysis
    
    def export_fallback_report(self, output_path=None):
        """
        Exporte rapport complet de fallback en JSON
        
        Args:
            output_path: Chemin de sortie (auto si None)
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"outputs/fallback_report_{timestamp}.json"
        
        # Calculer stats pour toutes les journées
        all_matchday_stats = {}
        for matchday in self.fallback_log.keys():
            all_matchday_stats[matchday] = self.calculate_matchday_fallback_percentage(matchday)
        
        # Rapport complet
        report = {
            'report_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_matchdays': len(self.fallback_log),
                'feature_definitions': self.feature_definitions
            },
            'matchday_stats': all_matchday_stats,
            'trend_analysis': self.get_fallback_trend_analysis(),
            'feature_reliability_ranking': self._rank_features_by_reliability(),
            'recommendations': self._generate_recommendations()
        }
        
        # Export JSON
        import os
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create directory if path has a directory component
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📋 Rapport fallback exporté: {output_path}")
        return output_path
    
    def _rank_features_by_reliability(self):
        """Classe les features par fiabilité (faible % fallback)"""
        feature_reliability = {}
        
        for matchday_data in self.fallback_log.values():
            for feature_name, entries in matchday_data.items():
                if feature_name not in feature_reliability:
                    feature_reliability[feature_name] = {'total': 0, 'fallbacks': 0}
                
                feature_reliability[feature_name]['total'] += len(entries)
                feature_reliability[feature_name]['fallbacks'] += sum(1 for e in entries if e['is_fallback'])
        
        # Calculer pourcentages et trier
        ranking = []
        for feature, stats in feature_reliability.items():
            if stats['total'] > 0:
                fallback_pct = stats['fallbacks'] / stats['total'] * 100
                ranking.append({
                    'feature': feature,
                    'fallback_percentage': fallback_pct,
                    'total_calculations': stats['total'],
                    'reliability_score': 100 - fallback_pct
                })
        
        return sorted(ranking, key=lambda x: x['reliability_score'], reverse=True)
    
    def _generate_recommendations(self):
        """Génère recommandations basées sur l'analyse"""
        trend = self.get_fallback_trend_analysis()
        
        recommendations = []
        
        if trend.get('avg_fallback_percentage', 0) > 30:
            recommendations.append("⚠️ Taux de fallback élevé (>30%) - Améliorer collecte données sources")
        
        if trend.get('trend_direction') == 'degrading':
            recommendations.append("📉 Tendance dégradante - Vérifier pipeline de données")
        
        if trend.get('fallback_std', 0) > 15:
            recommendations.append("📊 Forte variabilité - Stabiliser sources de données")
        
        return recommendations
    
    def get_live_stats_summary(self):
        """Résumé en temps réel pour dashboard"""
        latest_matchday = max(self.fallback_log.keys()) if self.fallback_log else None
        
        if not latest_matchday:
            return {'status': 'no_data'}
        
        latest_stats = self.calculate_matchday_fallback_percentage(latest_matchday)
        trend = self.get_fallback_trend_analysis()
        
        return {
            'latest_matchday': latest_matchday,
            'latest_fallback_percentage': latest_stats['overall_fallback_percentage'],
            'matches_analyzed': latest_stats['matches_analyzed'],
            'trend_direction': trend.get('trend_direction', 'unknown'),
            'avg_fallback_percentage': trend.get('avg_fallback_percentage', 0),
            'data_quality_alert': latest_stats['overall_fallback_percentage'] > 25
        }


# Instance globale pour tracking
global_fallback_tracker = FeatureFallbackTracker()


def track_fallback(matchday, match_id, feature_name, is_fallback, reason=None):
    """Helper function pour tracking fallback global"""
    global_fallback_tracker.track_feature_calculation(
        matchday, match_id, feature_name, is_fallback, reason
    )


def track_insufficient_data(matchday, match_id, feature_name, actual, required):
    """Helper function pour tracking données insuffisantes"""
    global_fallback_tracker.track_insufficient_data(
        matchday, match_id, feature_name, actual, required
    )


def get_matchday_report(matchday):
    """Helper function pour rapport journée"""
    return global_fallback_tracker.calculate_matchday_fallback_percentage(matchday)


def export_global_report(output_path=None):
    """Helper function pour export rapport global"""
    return global_fallback_tracker.export_fallback_report(output_path)


if __name__ == "__main__":
    print("🧪 Test Feature Fallback Tracker...")
    
    tracker = FeatureFallbackTracker()
    
    # Simuler quelques calculs J7
    tracker.track_feature_calculation('J7', 'Arsenal_vs_Chelsea', 'form_diff_normalized', False)
    tracker.track_feature_calculation('J7', 'Arsenal_vs_Chelsea', 'shots_diff_normalized', True, 'FBref indisponible')
    tracker.track_insufficient_data('J7', 'Arsenal_vs_Chelsea', 'home_xg_eff_10', 2, 3)
    
    # Stats J7
    j7_stats = tracker.calculate_matchday_fallback_percentage('J7')
    print(f"📊 J7 Fallback: {j7_stats['overall_fallback_percentage']:.1f}%")
    
    # Export test
    tracker.export_fallback_report("test_fallback_report.json")
    print("✅ Test terminé")