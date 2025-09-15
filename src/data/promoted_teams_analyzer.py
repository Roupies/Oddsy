#!/usr/bin/env python3
"""
Promoted Teams Analysis for EPL 2025-26 Rolling Validation

Deep analysis of model performance on promoted teams (Leeds, Sunderland, Burnley)
vs established teams, with insights and recommendations.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

class PromotedTeamsAnalyzer:
    """Analyze model performance specifically on promoted teams"""
    
    def __init__(self, report_file):
        self.report_file = report_file
        self.report_data = None
        self.promoted_teams = ['Leeds', 'Sunderland', 'Burnley']
        
    def load_report(self):
        """Load the rolling validation report"""
        with open(self.report_file, 'r') as f:
            self.report_data = json.load(f)
        print(f"📊 Loaded rolling validation report: {self.report_file}")
        
    def analyze_promoted_performance(self):
        """Deep analysis of promoted teams performance"""
        if not self.report_data:
            self.load_report()
            
        all_predictions = self.report_data['all_predictions']
        
        print("\n🆕 PROMOTED TEAMS DETAILED ANALYSIS")
        print("="*60)
        
        # Overall stats
        promoted_matches = [p for p in all_predictions if p.get('involves_promoted', False)]
        non_promoted_matches = [p for p in all_predictions if not p.get('involves_promoted', False)]
        
        print(f"\n📈 Overall Performance Comparison:")
        print(f"   Promoted Teams Matches: {len(promoted_matches)}")
        print(f"   Established Teams Matches: {len(non_promoted_matches)}")
        
        promoted_accuracy = sum(p['correct'] for p in promoted_matches) / len(promoted_matches) if promoted_matches else 0
        established_accuracy = sum(p['correct'] for p in non_promoted_matches) / len(non_promoted_matches) if non_promoted_matches else 0
        
        print(f"\n🎯 Accuracy Comparison:")
        print(f"   Promoted: {promoted_accuracy:.1%} ({sum(p['correct'] for p in promoted_matches)}/{len(promoted_matches)})")
        print(f"   Established: {established_accuracy:.1%} ({sum(p['correct'] for p in non_promoted_matches)}/{len(non_promoted_matches)})")
        
        accuracy_diff = promoted_accuracy - established_accuracy
        if accuracy_diff > 0:
            print(f"   🟢 Model performs {accuracy_diff:.1%} BETTER on promoted teams")
        else:
            print(f"   🔴 Model performs {abs(accuracy_diff):.1%} WORSE on promoted teams")
            
        # Individual team analysis
        print(f"\n🏟️  Individual Promoted Team Analysis:")
        
        team_stats = {}
        for team in self.promoted_teams:
            team_matches = [p for p in all_predictions if team in p.get('promoted_teams', [])]
            
            if team_matches:
                correct = sum(p['correct'] for p in team_matches)
                total = len(team_matches)
                accuracy = correct / total
                
                team_stats[team] = {
                    'matches': total,
                    'correct': correct,
                    'accuracy': accuracy,
                    'matches_data': team_matches
                }
                
                print(f"   🆕 {team:<12} | {correct:2d}/{total:2d} ({accuracy:.1%}) | Matches: {total}")
                
        # Position-specific analysis (Home vs Away)
        print(f"\n🏠 Home vs Away Performance (Promoted Teams):")
        
        promoted_home = [p for p in promoted_matches if p.get('home_status') == 'promoted']
        promoted_away = [p for p in promoted_matches if p.get('away_status') == 'promoted']
        
        if promoted_home:
            home_acc = sum(p['correct'] for p in promoted_home) / len(promoted_home)
            print(f"   Home: {home_acc:.1%} ({sum(p['correct'] for p in promoted_home)}/{len(promoted_home)})")
            
        if promoted_away:
            away_acc = sum(p['correct'] for p in promoted_away) / len(promoted_away)
            print(f"   Away: {away_acc:.1%} ({sum(p['correct'] for p in promoted_away)}/{len(promoted_away)})")
            
        # Confidence analysis
        print(f"\n🎯 Prediction Confidence Analysis:")
        
        promoted_confidences = [p['confidence'] for p in promoted_matches]
        established_confidences = [p['confidence'] for p in non_promoted_matches]
        
        if promoted_confidences and established_confidences:
            avg_promoted_conf = np.mean(promoted_confidences)
            avg_established_conf = np.mean(established_confidences)
            
            print(f"   Promoted Avg Confidence: {avg_promoted_conf:.3f}")
            print(f"   Established Avg Confidence: {avg_established_conf:.3f}")
            
            conf_diff = avg_promoted_conf - avg_established_conf
            if conf_diff > 0:
                print(f"   🟢 Model is {conf_diff:.3f} more confident on promoted teams")
            else:
                print(f"   🔴 Model is {abs(conf_diff):.3f} less confident on promoted teams")
                
        return {
            'team_stats': team_stats,
            'promoted_accuracy': promoted_accuracy,
            'established_accuracy': established_accuracy,
            'accuracy_difference': accuracy_diff,
            'promoted_matches_count': len(promoted_matches),
            'established_matches_count': len(non_promoted_matches)
        }
        
    def analyze_prediction_patterns(self):
        """Analyze prediction patterns and biases"""
        if not self.report_data:
            self.load_report()
            
        all_predictions = self.report_data['all_predictions']
        
        print(f"\n🔍 PREDICTION PATTERNS ANALYSIS")
        print("="*50)
        
        # Class prediction breakdown for promoted vs established
        promoted_matches = [p for p in all_predictions if p.get('involves_promoted', False)]
        non_promoted_matches = [p for p in all_predictions if not p.get('involves_promoted', False)]
        
        print(f"\n📊 Prediction Distribution:")
        
        for match_type, matches in [("Promoted Teams", promoted_matches), ("Established Teams", non_promoted_matches)]:
            if not matches:
                continue
                
            pred_counts = {'H': 0, 'D': 0, 'A': 0}
            actual_counts = {'H': 0, 'D': 0, 'A': 0}
            
            for p in matches:
                pred_counts[p['predicted_result']] += 1
                actual_counts[p['actual_result']] += 1
                
            total = len(matches)
            print(f"\n   {match_type}:")
            print(f"     Predicted - H: {pred_counts['H']:2d} ({pred_counts['H']/total:.1%}) | " +
                  f"D: {pred_counts['D']:2d} ({pred_counts['D']/total:.1%}) | " +
                  f"A: {pred_counts['A']:2d} ({pred_counts['A']/total:.1%})")
            print(f"     Actual    - H: {actual_counts['H']:2d} ({actual_counts['H']/total:.1%}) | " +
                  f"D: {actual_counts['D']:2d} ({actual_counts['D']/total:.1%}) | " +
                  f"A: {actual_counts['A']:2d} ({actual_counts['A']/total:.1%})")
        
        # Draw prediction analysis (known weakness)
        draw_analysis = self.report_data.get('insights', {}).get('draw_analysis', {})
        
        print(f"\n🎲 Draw Prediction Analysis:")
        print(f"   Predicted Draws: {draw_analysis.get('predicted_draws', 0)}")
        print(f"   Actual Draws: {draw_analysis.get('actual_draws', 0)}")
        print(f"   Draw Recall: {draw_analysis.get('draw_recall', 0):.1%}")
        print(f"   Draw Precision: {draw_analysis.get('draw_precision', 0):.1%}")
        
    def generate_recommendations(self, analysis_results):
        """Generate actionable recommendations based on analysis"""
        
        print(f"\n💡 RECOMMENDATIONS & INSIGHTS")
        print("="*50)
        
        promoted_acc = analysis_results['promoted_accuracy']
        established_acc = analysis_results['established_accuracy']
        acc_diff = analysis_results['accuracy_difference']
        
        print(f"\n🎯 Key Findings:")
        
        if acc_diff > 0.1:  # 10pp better
            print(f"   ✅ EXCELLENT: Model performs significantly better on promoted teams (+{acc_diff:.1%})")
            print(f"   💡 Recommendation: Model initialization for promoted teams is well-calibrated")
        elif acc_diff > 0:
            print(f"   🟢 GOOD: Model performs better on promoted teams (+{acc_diff:.1%})")
            print(f"   💡 Recommendation: Continue current promoted team initialization approach")
        elif acc_diff > -0.1:  # Within 10pp
            print(f"   🟡 NEUTRAL: Similar performance on both team types ({acc_diff:.1%} difference)")
            print(f"   💡 Recommendation: Model is reasonably robust to team status")
        else:
            print(f"   🔴 CONCERNING: Model performs worse on promoted teams ({acc_diff:.1%})")
            print(f"   💡 Recommendation: Review promoted team initialization and consider adjustments")
            
        # Specific recommendations
        print(f"\n🛠️  Specific Action Items:")
        
        if promoted_acc > 0.6:
            print(f"   1. 📈 Promoted teams performance is strong ({promoted_acc:.1%}) - maintain approach")
        else:
            print(f"   1. 📉 Promoted teams performance needs improvement ({promoted_acc:.1%})")
            print(f"      - Consider more conservative Elo initialization")
            print(f"      - Review Championship carry-over assumptions")
            
        # Draw prediction recommendations (known issue)
        draw_analysis = self.report_data.get('insights', {}).get('draw_analysis', {})
        draw_recall = draw_analysis.get('draw_recall', 0)
        
        if draw_recall < 0.2:  # Less than 20% draw recall
            print(f"   2. 🎲 Draw prediction is very weak ({draw_recall:.1%} recall)")
            print(f"      - Consider specialized draw classifier")
            print(f"      - Implement cascade model: General classifier → Draw specialist")
            
        print(f"\n🚀 Next Steps:")
        print(f"   1. Monitor J5-8 to validate these patterns")
        print(f"   2. If promoted teams performance drops, adjust initialization")
        print(f"   3. Consider draw prediction improvements if recall stays < 20%")
        print(f"   4. Track Elo convergence for promoted teams over 10+ matches")
        
        return analysis_results
        
    def save_detailed_analysis(self, analysis_results):
        """Save detailed analysis to file"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results/rolling_validation_2025_26/promoted_teams_analysis_{timestamp}.json"
        
        detailed_analysis = {
            'timestamp': datetime.now().isoformat(),
            'source_report': str(self.report_file),
            'promoted_teams': self.promoted_teams,
            'analysis_results': analysis_results,
            'recommendations': {
                'performance_assessment': 'good' if analysis_results['accuracy_difference'] > 0 else 'needs_review',
                'initialization_status': 'maintain' if analysis_results['promoted_accuracy'] > 0.5 else 'adjust',
                'monitoring_priority': 'high' if analysis_results['accuracy_difference'] < -0.1 else 'normal'
            }
        }
        
        Path("results/rolling_validation_2025_26").mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(detailed_analysis, f, indent=2, default=str)
            
        print(f"\n💾 Detailed analysis saved: {output_file}")
        return output_file
        
    def run_complete_analysis(self):
        """Run complete promoted teams analysis"""
        
        print("🔍 PROMOTED TEAMS COMPLETE ANALYSIS")
        print("="*60)
        
        self.load_report()
        
        # Main analysis
        analysis_results = self.analyze_promoted_performance()
        
        # Pattern analysis
        self.analyze_prediction_patterns()
        
        # Recommendations
        self.generate_recommendations(analysis_results)
        
        # Save results
        output_file = self.save_detailed_analysis(analysis_results)
        
        print(f"\n🎉 Analysis completed! See: {output_file}")
        
        return analysis_results

def main():
    """Find latest report and run analysis"""
    
    # Find the latest rolling validation report
    reports_dir = Path("results/rolling_validation_2025_26")
    
    if not reports_dir.exists():
        print("❌ No rolling validation reports found. Run rolling_epl_2025_26_validator.py first.")
        return
        
    report_files = list(reports_dir.glob("rolling_validation_report_*.json"))
    
    if not report_files:
        print("❌ No rolling validation reports found. Run rolling_epl_2025_26_validator.py first.")
        return
        
    # Get the most recent report
    latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
    
    print(f"📊 Using report: {latest_report}")
    
    # Run analysis
    analyzer = PromotedTeamsAnalyzer(latest_report)
    results = analyzer.run_complete_analysis()
    
    return results

if __name__ == "__main__":
    main()