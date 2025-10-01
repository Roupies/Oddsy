#!/usr/bin/env python3
"""
📊 DASHBOARD MONITORING HEBDOMADAIRE

Objectif: Surveillance continue performance modèle avec seuils alerte
- Tracking accuracy hebdomadaire
- Détection drift features 
- Alertes automatiques < 47%
- Historique performance mensuelle
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, classification_report
import joblib
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

class WeeklyPerformanceMonitor:
    def __init__(self, model_path='models/final_robust_model_20250915_163023.joblib', 
                 data_path='data/processed/v15_final_enhanced.csv',
                 alert_threshold=0.47):
        """
        Initialiser moniteur de performance
        
        Args:
            model_path: Chemin vers modèle de production
            data_path: Chemin vers dataset principal
            alert_threshold: Seuil alerte accuracy (défaut: 47%)
        """
        self.model_path = model_path
        self.data_path = data_path
        self.alert_threshold = alert_threshold
        
        # Features baseline
        self.features = [
            'form_diff_normalized', 'elo_diff_normalized', 'h2h_score', 'matchday_normalized',
            'shots_diff_normalized', 'corners_diff_normalized', 'market_entropy_norm',
            'home_xg_eff_10', 'away_goals_sum_5', 'away_xg_eff_10'
        ]
        
        # Charger modèle
        try:
            self.model = joblib.load(model_path)
            print(f"✅ Modèle chargé: {model_path}")
        except FileNotFoundError:
            print(f"❌ Modèle non trouvé: {model_path}")
            self.model = None
        
        # Créer dossier monitoring
        self.monitoring_dir = Path('monitoring')
        self.monitoring_dir.mkdir(exist_ok=True)
        
        # Historique performance
        self.history_file = self.monitoring_dir / 'performance_history.json'
        self.load_history()
    
    def load_history(self):
        """Charger historique performance"""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = {
                'weekly_performance': [],
                'alerts': [],
                'feature_drift': []
            }
    
    def save_history(self):
        """Sauvegarder historique performance"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)
    
    def get_recent_matches(self, days_back=7):
        """Récupérer matches récents pour monitoring"""
        df = pd.read_csv(self.data_path, parse_dates=['Date'])
        
        # Date limite (derniers X jours)
        cutoff_date = datetime.now() - timedelta(days=days_back)
        cutoff_date = pd.Timestamp(cutoff_date)
        
        # Matches récents
        recent_matches = df[df['Date'] >= cutoff_date].copy()
        
        # Si pas de matches récents, prendre derniers matches disponibles
        if len(recent_matches) == 0:
            recent_matches = df.tail(10).copy()  # Derniers 10 matches
            print(f"⚠️ Pas de matches dans les {days_back} derniers jours, utilisation derniers matches")
        
        return recent_matches
    
    def calculate_weekly_performance(self, matches_df):
        """Calculer performance hebdomadaire"""
        if self.model is None:
            return None
        
        # Nettoyer données
        clean_matches = matches_df.dropna(subset=self.features + ['FullTimeResult'])
        
        if len(clean_matches) == 0:
            print("❌ Pas de matches propres pour évaluation")
            return None
        
        # Préparer données
        X = clean_matches[self.features]
        y_true = clean_matches['FullTimeResult'].map({'H': 0, 'D': 1, 'A': 2})
        
        # Prédictions
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)
        
        # Métriques
        accuracy = accuracy_score(y_true, y_pred)
        
        # Performance par classe
        class_report = classification_report(y_true, y_pred, target_names=['H', 'D', 'A'], output_dict=True)
        
        # Confiance moyenne
        avg_confidence = np.mean(np.max(y_proba, axis=1))
        
        performance_data = {
            'timestamp': datetime.now().isoformat(),
            'matches_evaluated': len(clean_matches),
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'class_performance': {
                'H': {
                    'precision': class_report['H']['precision'],
                    'recall': class_report['H']['recall'],
                    'support': class_report['H']['support']
                },
                'D': {
                    'precision': class_report['D']['precision'],
                    'recall': class_report['D']['recall'],
                    'support': class_report['D']['support']
                },
                'A': {
                    'precision': class_report['A']['precision'],
                    'recall': class_report['A']['recall'],
                    'support': class_report['A']['support']
                }
            },
            'date_range': {
                'start': clean_matches['Date'].min().isoformat(),
                'end': clean_matches['Date'].max().isoformat()
            }
        }
        
        return performance_data
    
    def calculate_feature_drift(self, recent_matches):
        """Calculer drift des features vs baseline training"""
        # Charger données training pour référence
        df_full = pd.read_csv(self.data_path, parse_dates=['Date'])
        training_cutoff = pd.Timestamp('2025-08-01')
        training_data = df_full[df_full['Date'] < training_cutoff]
        
        drift_analysis = []
        
        for feature in self.features:
            if feature in training_data.columns and feature in recent_matches.columns:
                # Statistiques training
                train_vals = training_data[feature].dropna()
                train_mean = train_vals.mean()
                train_std = train_vals.std()
                
                # Statistiques récentes
                recent_vals = recent_matches[feature].dropna()
                if len(recent_vals) > 0:
                    recent_mean = recent_vals.mean()
                    recent_std = recent_vals.std()
                    
                    # Score drift (écart en nombre d'écarts-types)
                    drift_score = abs(recent_mean - train_mean) / train_std if train_std > 0 else 0
                    
                    drift_analysis.append({
                        'feature': feature,
                        'train_mean': train_mean,
                        'recent_mean': recent_mean,
                        'train_std': train_std,
                        'recent_std': recent_std,
                        'drift_score': drift_score,
                        'drift_level': 'HIGH' if drift_score > 1.0 else ('MEDIUM' if drift_score > 0.5 else 'LOW')
                    })
        
        return drift_analysis
    
    def check_alerts(self, performance_data, drift_analysis):
        """Vérifier conditions d'alerte"""
        alerts = []
        
        # Alerte accuracy
        if performance_data and performance_data['accuracy'] < self.alert_threshold:
            alerts.append({
                'type': 'ACCURACY_LOW',
                'severity': 'HIGH',
                'message': f"Accuracy {performance_data['accuracy']:.1%} < seuil {self.alert_threshold:.1%}",
                'value': performance_data['accuracy'],
                'threshold': self.alert_threshold
            })
        
        # Alerte draw performance
        if performance_data and 'D' in performance_data['class_performance']:
            draw_recall = performance_data['class_performance']['D']['recall']
            if draw_recall < 0.15:  # Seuil 15% recall draws
                alerts.append({
                    'type': 'DRAW_PERFORMANCE_CRITICAL',
                    'severity': 'MEDIUM',
                    'message': f"Draw recall {draw_recall:.1%} critique < 15%",
                    'value': draw_recall,
                    'threshold': 0.15
                })
        
        # Alerte drift features
        high_drift_features = [d for d in drift_analysis if d['drift_level'] == 'HIGH']
        if len(high_drift_features) > 2:
            alerts.append({
                'type': 'FEATURE_DRIFT_HIGH',
                'severity': 'MEDIUM',
                'message': f"{len(high_drift_features)} features avec drift élevé",
                'features': [f['feature'] for f in high_drift_features],
                'max_drift': max(f['drift_score'] for f in high_drift_features)
            })
        
        # Alerte confiance
        if performance_data and performance_data['avg_confidence'] < 0.4:
            alerts.append({
                'type': 'CONFIDENCE_LOW',
                'severity': 'LOW',
                'message': f"Confiance moyenne {performance_data['avg_confidence']:.1%} faible",
                'value': performance_data['avg_confidence'],
                'threshold': 0.4
            })
        
        return alerts
    
    def generate_dashboard_report(self, performance_data, drift_analysis, alerts):
        """Générer rapport dashboard"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
🔍 DASHBOARD MONITORING HEBDOMADAIRE
=====================================
Généré le: {timestamp}

📊 PERFORMANCE RÉCENTE
----------------------
"""
        
        if performance_data:
            report += f"""
✅ Matches évalués: {performance_data['matches_evaluated']}
🎯 Accuracy: {performance_data['accuracy']:.1%}
🎲 Confiance moyenne: {performance_data['avg_confidence']:.1%}
📅 Période: {performance_data['date_range']['start'][:10]} → {performance_data['date_range']['end'][:10]}

📋 Performance par classe:
  Home (H): Precision={performance_data['class_performance']['H']['precision']:.3f}, Recall={performance_data['class_performance']['H']['recall']:.3f}
  Draw (D): Precision={performance_data['class_performance']['D']['precision']:.3f}, Recall={performance_data['class_performance']['D']['recall']:.3f}
  Away (A): Precision={performance_data['class_performance']['A']['precision']:.3f}, Recall={performance_data['class_performance']['A']['recall']:.3f}
"""
        else:
            report += "\n❌ Pas de données de performance disponibles\n"
        
        report += f"\n📈 DRIFT FEATURES\n----------------\n"
        
        if drift_analysis:
            for drift in sorted(drift_analysis, key=lambda x: x['drift_score'], reverse=True)[:5]:
                status = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[drift['drift_level']]
                report += f"{status} {drift['feature']:25}: {drift['drift_score']:.3f} ({drift['drift_level']})\n"
        else:
            report += "❌ Pas d'analyse drift disponible\n"
        
        report += f"\n🚨 ALERTES\n----------\n"
        
        if alerts:
            for alert in alerts:
                severity_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[alert['severity']]
                report += f"{severity_emoji} [{alert['severity']}] {alert['type']}: {alert['message']}\n"
        else:
            report += "✅ Aucune alerte\n"
        
        report += f"\n📊 STATUT GÉNÉRAL\n-----------------\n"
        
        if alerts:
            high_alerts = [a for a in alerts if a['severity'] == 'HIGH']
            if high_alerts:
                report += "🔴 ATTENTION REQUISE: Alertes critiques détectées\n"
            else:
                report += "🟡 SURVEILLANCE: Alertes mineures, monitoring continu\n"
        else:
            report += "✅ SYSTÈME NOMINAL: Performances dans les normes\n"
        
        return report
    
    def save_dashboard_report(self, report):
        """Sauvegarder rapport dashboard"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.monitoring_dir / f"dashboard_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Garder aussi rapport courant
        current_report = self.monitoring_dir / "latest_dashboard_report.txt"
        with open(current_report, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Rapport sauvé: {report_file}")
        return report_file
    
    def create_performance_chart(self):
        """Créer graphique historique performance"""
        if len(self.history['weekly_performance']) < 2:
            print("⚠️ Pas assez d'historique pour graphique")
            return None
        
        # Préparer données pour graphique
        timestamps = [entry['timestamp'] for entry in self.history['weekly_performance']]
        accuracies = [entry['accuracy'] for entry in self.history['weekly_performance']]
        
        # Créer graphique
        plt.figure(figsize=(12, 6))
        
        # Performance accuracy
        plt.subplot(1, 2, 1)
        dates = pd.to_datetime(timestamps)
        plt.plot(dates, accuracies, marker='o', linewidth=2, markersize=6)
        plt.axhline(y=self.alert_threshold, color='red', linestyle='--', alpha=0.7, label=f'Seuil alerte ({self.alert_threshold:.1%})')
        plt.title('Performance Hebdomadaire')
        plt.ylabel('Accuracy')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Performance par classe (dernière semaine)
        if self.history['weekly_performance']:
            last_perf = self.history['weekly_performance'][-1]['class_performance']
            
            plt.subplot(1, 2, 2)
            classes = ['H', 'D', 'A']
            recalls = [last_perf[c]['recall'] for c in classes]
            precisions = [last_perf[c]['precision'] for c in classes]
            
            x = np.arange(len(classes))
            width = 0.35
            
            plt.bar(x - width/2, recalls, width, label='Recall', alpha=0.8)
            plt.bar(x + width/2, precisions, width, label='Precision', alpha=0.8)
            
            plt.title('Performance par Classe (Dernière Semaine)')
            plt.ylabel('Score')
            plt.xlabel('Classe')
            plt.xticks(x, classes)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Sauvegarder graphique
        chart_file = self.monitoring_dir / f"performance_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Graphique sauvé: {chart_file}")
        return chart_file
    
    def run_weekly_monitoring(self, days_back=7):
        """Exécuter monitoring hebdomadaire complet"""
        print(f"📊 MONITORING HEBDOMADAIRE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # 1. Récupérer matches récents
        recent_matches = self.get_recent_matches(days_back)
        print(f"📅 Matches récents: {len(recent_matches)} (derniers {days_back} jours)")
        
        # 2. Calculer performance
        performance_data = self.calculate_weekly_performance(recent_matches)
        
        # 3. Analyser drift features
        drift_analysis = self.calculate_feature_drift(recent_matches)
        
        # 4. Vérifier alertes
        alerts = self.check_alerts(performance_data, drift_analysis)
        
        # 5. Générer rapport
        report = self.generate_dashboard_report(performance_data, drift_analysis, alerts)
        print(report)
        
        # 6. Sauvegarder
        report_file = self.save_dashboard_report(report)
        
        # 7. Mettre à jour historique
        if performance_data:
            self.history['weekly_performance'].append(performance_data)
        
        if alerts:
            for alert in alerts:
                alert['timestamp'] = datetime.now().isoformat()
                self.history['alerts'].append(alert)
        
        if drift_analysis:
            drift_record = {
                'timestamp': datetime.now().isoformat(),
                'drift_features': drift_analysis
            }
            self.history['feature_drift'].append(drift_record)
        
        # Limiter taille historique (garder 20 dernières entrées)
        for key in self.history:
            if len(self.history[key]) > 20:
                self.history[key] = self.history[key][-20:]
        
        self.save_history()
        
        # 8. Créer graphique si assez d'historique
        chart_file = self.create_performance_chart()
        
        # 9. Résumé final
        print(f"\n📋 RÉSUMÉ MONITORING:")
        print("-" * 30)
        if performance_data:
            accuracy = performance_data['accuracy']
            status = "🔴 CRITIQUE" if accuracy < 0.43 else ("🟡 ATTENTION" if accuracy < 0.47 else "✅ OK")
            print(f"Performance: {accuracy:.1%} {status}")
        
        high_alerts = [a for a in alerts if a['severity'] == 'HIGH']
        print(f"Alertes critiques: {len(high_alerts)}")
        print(f"Drift élevé: {len([d for d in drift_analysis if d['drift_level'] == 'HIGH'])}")
        
        print(f"\n✅ Monitoring terminé - Rapport: {report_file.name}")
        
        return {
            'performance': performance_data,
            'drift': drift_analysis,
            'alerts': alerts,
            'report_file': report_file
        }

def main():
    """Exécuter monitoring hebdomadaire"""
    monitor = WeeklyPerformanceMonitor()
    results = monitor.run_weekly_monitoring(days_back=30)  # Derniers 30 jours
    
    # Recommandations automatiques basées sur résultats
    if results['alerts']:
        high_alerts = [a for a in results['alerts'] if a['severity'] == 'HIGH']
        if high_alerts:
            print(f"\n🚨 ACTIONS RECOMMANDÉES:")
            print("- Vérifier données récentes")
            print("- Recalibrer modèle si drift persistant")
            print("- Investiguer causes performance faible")

if __name__ == "__main__":
    main()