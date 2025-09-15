"""
Persistent metrics tracking system for Oddsy ML pipeline
Tracks model performance over time and generates reports
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from typing import Dict, List, Any, Optional
from utils import setup_logging, load_config

class MetricsTracker:
    """Track and analyze ML model performance metrics over time"""
    
    def __init__(self, config_path: str = "config/config.json"):
        self.config = load_config(config_path)
        self.logger = setup_logging(config_path)
        self.metrics_file = "evaluation/metrics_history.json"
        self.reports_dir = "evaluation/reports/"
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def load_metrics_history(self) -> List[Dict[str, Any]]:
        """Load historical metrics data"""
        if not os.path.exists(self.metrics_file):
            self.logger.warning(f"No metrics history found at {self.metrics_file}")
            return []
        
        with open(self.metrics_file, 'r') as f:
            return json.load(f)
    
    def add_metrics(self, metrics: Dict[str, Any]) -> None:
        """Add new metrics to history"""
        history = self.load_metrics_history()
        
        # Add timestamp if not present
        if 'timestamp' not in metrics:
            metrics['timestamp'] = datetime.now().isoformat()
        
        history.append(metrics)
        
        # Save updated history
        with open(self.metrics_file, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        self.logger.info(f"Added metrics entry. Total entries: {len(history)}")
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Convert metrics history to pandas DataFrame for analysis"""
        history = self.load_metrics_history()
        
        if not history:
            return pd.DataFrame()
        
        # Flatten nested metrics
        flattened_data = []
        for entry in history:
            flat_entry = {'timestamp': entry.get('timestamp')}
            
            # Add overall metrics
            if 'overall_metrics' in entry:
                for key, value in entry['overall_metrics'].items():
                    flat_entry[f"overall_{key}"] = value
            
            # Add cross-validation metrics
            if 'cross_validation' in entry:
                for key, cv_data in entry['cross_validation'].items():
                    if isinstance(cv_data, dict) and 'mean' in cv_data:
                        flat_entry[f"cv_{key}_mean"] = cv_data['mean']
                        flat_entry[f"cv_{key}_std"] = cv_data['std']
            
            # Add baseline comparisons
            if 'baseline_comparisons' in entry:
                for key, value in entry['baseline_comparisons'].items():
                    flat_entry[f"baseline_{key}"] = value
            
            # Add model info
            if 'model_name' in entry:
                flat_entry['model_name'] = entry['model_name']
            
            flattened_data.append(flat_entry)
        
        df = pd.DataFrame(flattened_data)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        df = self.get_metrics_dataframe()
        
        if df.empty:
            return {"status": "no_data", "message": "No metrics data available"}
        
        # Latest metrics
        latest = df.iloc[-1] if len(df) > 0 else None
        
        # Performance trends
        metrics_cols = [col for col in df.columns if col.startswith(('overall_', 'cv_'))]
        trends = {}
        
        if len(df) > 1:
            for col in metrics_cols:
                if col in df.columns and df[col].notna().sum() > 1:
                    # Calculate trend (recent vs older)
                    recent_avg = df[col].tail(3).mean()
                    older_avg = df[col].head(max(1, len(df)-3)).mean()
                    trends[col] = {
                        'recent_avg': recent_avg,
                        'older_avg': older_avg,
                        'trend': 'improving' if recent_avg > older_avg else 'declining',
                        'change': recent_avg - older_avg
                    }
        
        # Baseline achievements
        baseline_cols = [col for col in df.columns if col.startswith('baseline_')]
        baseline_summary = {}
        
        if latest is not None:
            for col in baseline_cols:
                if col in latest:
                    baseline_summary[col] = latest[col]
        
        # Model stability (coefficient of variation)
        stability = {}
        for col in ['overall_accuracy', 'cv_accuracy_mean']:
            if col in df.columns and df[col].notna().sum() > 1:
                mean_val = df[col].mean()
                std_val = df[col].std()
                stability[col] = {
                    'mean': mean_val,
                    'std': std_val,
                    'cv': std_val / mean_val if mean_val != 0 else float('inf'),
                    'stability': 'stable' if (std_val / mean_val) < 0.05 else 'unstable'
                }
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_experiments': len(df),
            'date_range': {
                'first': df['timestamp'].min().isoformat() if 'timestamp' in df else None,
                'last': df['timestamp'].max().isoformat() if 'timestamp' in df else None
            },
            'latest_metrics': latest.to_dict() if latest is not None else None,
            'performance_trends': trends,
            'baseline_achievements': baseline_summary,
            'model_stability': stability
        }
        
        return report
    
    def create_visualizations(self) -> List[str]:
        """Create performance visualization charts"""
        df = self.get_metrics_dataframe()
        
        if df.empty:
            self.logger.warning("No data available for visualizations")
            return []
        
        created_files = []
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        
        # 1. Performance over time
        plt.figure(figsize=(12, 8))
        
        metrics_to_plot = ['overall_accuracy', 'cv_accuracy_mean', 'overall_f1_macro']
        colors = ['blue', 'red', 'green']
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in df.columns and df[metric].notna().sum() > 0:
                plt.plot(df['timestamp'], df[metric], 
                        marker='o', label=metric.replace('_', ' ').title(), 
                        color=colors[i % len(colors)])
        
        # Add baseline lines
        baselines = self.config['baselines']
        plt.axhline(y=baselines['random_accuracy'], color='gray', 
                   linestyle='--', alpha=0.7, label='Random Baseline')
        plt.axhline(y=baselines['target_thresholds']['good'], color='orange', 
                   linestyle='--', alpha=0.7, label='Good Threshold (50%)')
        plt.axhline(y=baselines['target_thresholds']['excellent'], color='gold', 
                   linestyle='--', alpha=0.7, label='Excellent Threshold (55%)')
        
        plt.title('Model Performance Over Time')
        plt.xlabel('Date')
        plt.ylabel('Score')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        perf_file = f"{self.reports_dir}performance_over_time_{timestamp}.png"
        plt.savefig(perf_file, dpi=300, bbox_inches='tight')
        plt.close()
        created_files.append(perf_file)
        
        # 2. Metrics distribution
        if len(df) > 1:
            plt.figure(figsize=(10, 6))
            
            metrics_for_dist = ['overall_accuracy', 'overall_precision_macro', 
                               'overall_recall_macro', 'overall_f1_macro']
            available_metrics = [m for m in metrics_for_dist if m in df.columns]
            
            if available_metrics:
                df[available_metrics].hist(bins=min(10, len(df)), alpha=0.7, figsize=(12, 8))
                plt.suptitle('Metrics Distribution')
                plt.tight_layout()
                
                dist_file = f"{self.reports_dir}metrics_distribution_{timestamp}.png"
                plt.savefig(dist_file, dpi=300, bbox_inches='tight')
                plt.close()
                created_files.append(dist_file)
        
        # 3. Cross-validation stability
        cv_metrics = [col for col in df.columns if col.startswith('cv_') and col.endswith('_mean')]
        if cv_metrics:
            plt.figure(figsize=(10, 6))
            
            for i, metric in enumerate(cv_metrics[:4]):  # Limit to 4 metrics
                if df[metric].notna().sum() > 0:
                    plt.plot(df['timestamp'], df[metric], 
                            marker='s', label=metric.replace('cv_', '').replace('_mean', ''))
            
            plt.title('Cross-Validation Metrics Stability')
            plt.xlabel('Date')
            plt.ylabel('CV Score')
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            cv_file = f"{self.reports_dir}cv_stability_{timestamp}.png"
            plt.savefig(cv_file, dpi=300, bbox_inches='tight')
            plt.close()
            created_files.append(cv_file)
        
        self.logger.info(f"Created {len(created_files)} visualization files")
        return created_files
    
    def export_report(self) -> str:
        """Export comprehensive report to file"""
        report = self.generate_performance_report()
        
        # Create visualizations
        viz_files = self.create_visualizations()
        report['visualizations'] = viz_files
        
        # Save report
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        report_file = f"{self.reports_dir}performance_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create markdown summary
        md_file = f"{self.reports_dir}performance_summary_{timestamp}.md"
        self._create_markdown_summary(report, md_file)
        
        self.logger.info(f"Exported report: {report_file}")
        return report_file
    
    def _create_markdown_summary(self, report: Dict[str, Any], filename: str) -> None:
        """Create human-readable markdown summary"""
        with open(filename, 'w') as f:
            f.write("# Oddsy ML Model Performance Report\n\n")
            f.write(f"Generated: {report['generated_at']}\n\n")
            
            if report['total_experiments'] > 0:
                f.write(f"## Overview\n")
                f.write(f"- Total experiments: {report['total_experiments']}\n")
                if report['date_range']['first']:
                    f.write(f"- Date range: {report['date_range']['first']} to {report['date_range']['last']}\n\n")
                
                # Latest metrics
                if report['latest_metrics']:
                    f.write(f"## Latest Performance\n")
                    latest = report['latest_metrics']
                    
                    if 'overall_accuracy' in latest:
                        f.write(f"- Accuracy: {latest['overall_accuracy']:.3f}\n")
                    if 'overall_f1_macro' in latest:
                        f.write(f"- F1-Score: {latest['overall_f1_macro']:.3f}\n")
                    if 'cv_accuracy_mean' in latest:
                        f.write(f"- CV Accuracy: {latest['cv_accuracy_mean']:.3f}\n")
                    
                    f.write("\n")
                
                # Baseline achievements
                if report['baseline_achievements']:
                    f.write(f"## Baseline Comparisons\n")
                    baselines = report['baseline_achievements']
                    
                    for key, value in baselines.items():
                        if isinstance(value, bool):
                            status = "✅" if value else "❌"
                            readable_key = key.replace('baseline_', '').replace('_', ' ').title()
                            f.write(f"- {readable_key}: {status}\n")
                    
                    f.write("\n")
                
                # Trends
                if report['performance_trends']:
                    f.write(f"## Performance Trends\n")
                    for metric, trend_data in report['performance_trends'].items():
                        readable_metric = metric.replace('overall_', '').replace('cv_', '').replace('_', ' ').title()
                        trend_icon = "📈" if trend_data['trend'] == 'improving' else "📉"
                        f.write(f"- {readable_metric}: {trend_icon} {trend_data['trend'].title()}\n")
                    
                    f.write("\n")
                
                # Stability
                if report['model_stability']:
                    f.write(f"## Model Stability\n")
                    for metric, stability_data in report['model_stability'].items():
                        readable_metric = metric.replace('overall_', '').replace('cv_', '').replace('_', ' ').title()
                        stability_icon = "🟢" if stability_data['stability'] == 'stable' else "🟡"
                        f.write(f"- {readable_metric}: {stability_icon} {stability_data['stability'].title()}\n")
            else:
                f.write("No experiment data available.\n")

def main():
    """CLI interface for metrics tracking"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Oddsy Metrics Tracker")
    parser.add_argument('--report', action='store_true', help='Generate performance report')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    
    args = parser.parse_args()
    
    tracker = MetricsTracker()
    
    if args.report:
        report_file = tracker.export_report()
        print(f"Performance report generated: {report_file}")
    elif args.visualize:
        viz_files = tracker.create_visualizations()
        print(f"Created visualizations: {viz_files}")
    else:
        # Default: show latest metrics
        df = tracker.get_metrics_dataframe()
        if not df.empty:
            latest = df.iloc[-1]
            print("Latest Metrics:")
            for col, value in latest.items():
                if col not in ['timestamp', 'model_name']:
                    print(f"  {col}: {value}")
        else:
            print("No metrics data available")

if __name__ == "__main__":
    main()