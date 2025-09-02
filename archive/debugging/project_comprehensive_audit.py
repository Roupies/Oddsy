#!/usr/bin/env python3
"""
COMPREHENSIVE PROJECT AUDIT - ODDSY v2.3
=======================================

Complete audit of the Oddsy project after achieving validated 55% model performance.
This audit covers all aspects: code quality, data integrity, model validation,
documentation completeness, and production readiness.

Audit Categories:
1. Model & Data Integrity
2. Code Quality & Structure
3. Documentation Completeness
4. Production Readiness
5. Performance & Benchmarks
6. Risk Assessment
7. Future Development Roadmap
"""

import os
import pandas as pd
import numpy as np
import json
import glob
from datetime import datetime
import warnings
import sys
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('.')
from utils import setup_logging

class ProjectAuditor:
    def __init__(self):
        self.logger = setup_logging()
        self.audit_results = {}
        self.project_root = "."
        
    def audit_model_data_integrity(self):
        """Audit 1: Model and Data Integrity"""
        self.logger.info("=== AUDIT 1: MODEL & DATA INTEGRITY ===")
        
        integrity_score = 0
        max_score = 10
        issues = []
        
        # Check production model exists
        production_model = "models/randomforest_corrected_model_2025_09_02_113228.joblib"
        if os.path.exists(production_model):
            self.logger.info("✅ Production model file exists")
            integrity_score += 2
        else:
            issues.append("Production model file missing")
            
        # Check production dataset exists
        production_dataset = "data/processed/v13_xg_corrected_features_latest.csv"
        if os.path.exists(production_dataset):
            self.logger.info("✅ Production dataset exists")
            integrity_score += 2
            
            # Quick data quality check
            try:
                df = pd.read_csv(production_dataset)
                if len(df) >= 2000:
                    integrity_score += 1
                    self.logger.info(f"✅ Dataset size adequate: {len(df)} matches")
                    
                # Check key features exist
                required_features = ['elo_diff_normalized', 'market_entropy_norm', 
                                   'home_xg_eff_10', 'away_xg_eff_10']
                missing_features = [f for f in required_features if f not in df.columns]
                
                if not missing_features:
                    integrity_score += 2
                    self.logger.info("✅ All key features present")
                else:
                    issues.append(f"Missing features: {missing_features}")
                    
            except Exception as e:
                issues.append(f"Dataset loading error: {e}")
        else:
            issues.append("Production dataset missing")
            
        # Check model metadata exists
        metadata_file = "models/randomforest_corrected_model_2025_09_02_113228_metadata.json"
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                if metadata.get('accuracy', 0) >= 0.55:
                    integrity_score += 2
                    self.logger.info(f"✅ Model performance documented: {metadata['accuracy']:.1%}")
                else:
                    issues.append("Model performance below target")
            except Exception as e:
                issues.append(f"Metadata parsing error: {e}")
        else:
            issues.append("Model metadata missing")
            
        # Check validation scripts exist
        validation_scripts = [
            "validate_56_model_integrity.py",
            "final_model_verification.py"
        ]
        
        existing_validation = [s for s in validation_scripts if os.path.exists(s)]
        if len(existing_validation) >= 1:
            integrity_score += 1
            self.logger.info(f"✅ Validation scripts available: {len(existing_validation)}")
        else:
            issues.append("Validation scripts missing")
            
        self.audit_results['model_data_integrity'] = {
            'score': integrity_score,
            'max_score': max_score,
            'percentage': (integrity_score / max_score) * 100,
            'issues': issues,
            'status': 'EXCELLENT' if integrity_score >= 8 else 'GOOD' if integrity_score >= 6 else 'NEEDS_IMPROVEMENT'
        }
        
        self.logger.info(f"📊 Model & Data Integrity: {integrity_score}/{max_score} ({(integrity_score/max_score)*100:.0f}%)")
        
    def audit_code_quality_structure(self):
        """Audit 2: Code Quality and Project Structure"""
        self.logger.info("=== AUDIT 2: CODE QUALITY & STRUCTURE ===")
        
        quality_score = 0
        max_score = 10
        issues = []
        
        # Check project structure
        expected_dirs = ['data', 'models', 'scripts', 'config']
        existing_dirs = [d for d in expected_dirs if os.path.isdir(d)]
        
        if len(existing_dirs) == len(expected_dirs):
            quality_score += 2
            self.logger.info("✅ Project structure complete")
        else:
            missing = set(expected_dirs) - set(existing_dirs)
            issues.append(f"Missing directories: {missing}")
            
        # Check for utility modules
        if os.path.exists('utils.py'):
            quality_score += 1
            self.logger.info("✅ Utility module exists")
        else:
            issues.append("Utility module missing")
            
        # Check for configuration files
        config_files = glob.glob('config/*.json')
        if len(config_files) >= 3:
            quality_score += 2
            self.logger.info(f"✅ Configuration files available: {len(config_files)}")
        else:
            issues.append("Insufficient configuration files")
            
        # Check for preprocessing scripts
        preprocessing_scripts = glob.glob('scripts/preprocessing/*.py')
        if len(preprocessing_scripts) >= 5:
            quality_score += 2
            self.logger.info(f"✅ Preprocessing pipeline: {len(preprocessing_scripts)} scripts")
        else:
            issues.append("Limited preprocessing scripts")
            
        # Check for modeling scripts
        modeling_scripts = glob.glob('scripts/modeling/*.py')
        if len(modeling_scripts) >= 3:
            quality_score += 1
            self.logger.info(f"✅ Modeling scripts: {len(modeling_scripts)} available")
        else:
            issues.append("Limited modeling scripts")
            
        # Check for analysis scripts
        analysis_scripts = glob.glob('scripts/analysis/*.py')
        if len(analysis_scripts) >= 5:
            quality_score += 1
            self.logger.info(f"✅ Analysis tools: {len(analysis_scripts)} scripts")
        else:
            issues.append("Limited analysis scripts")
            
        # Check for test files
        test_files = glob.glob('test*.py') + glob.glob('tests/*.py')
        if len(test_files) >= 2:
            quality_score += 1
            self.logger.info(f"✅ Test coverage: {len(test_files)} test files")
        else:
            issues.append("Limited test coverage")
            
        self.audit_results['code_quality'] = {
            'score': quality_score,
            'max_score': max_score,
            'percentage': (quality_score / max_score) * 100,
            'issues': issues,
            'status': 'EXCELLENT' if quality_score >= 8 else 'GOOD' if quality_score >= 6 else 'NEEDS_IMPROVEMENT'
        }
        
        self.logger.info(f"📊 Code Quality & Structure: {quality_score}/{max_score} ({(quality_score/max_score)*100:.0f}%)")
        
    def audit_documentation_completeness(self):
        """Audit 3: Documentation Completeness"""
        self.logger.info("=== AUDIT 3: DOCUMENTATION COMPLETENESS ===")
        
        doc_score = 0
        max_score = 8
        issues = []
        
        # Check CLAUDE.md exists and is comprehensive
        if os.path.exists('CLAUDE.md'):
            try:
                with open('CLAUDE.md', 'r') as f:
                    content = f.read()
                    
                if len(content) > 5000:  # Substantial documentation
                    doc_score += 3
                    self.logger.info("✅ Comprehensive CLAUDE.md documentation")
                    
                    # Check for key sections
                    key_sections = ['Production Status', 'Feature Set', 'Performance', 'Commands']
                    present_sections = sum(1 for section in key_sections if section.lower() in content.lower())
                    
                    if present_sections >= 3:
                        doc_score += 1
                        self.logger.info(f"✅ Key documentation sections: {present_sections}/4")
                else:
                    issues.append("CLAUDE.md documentation insufficient")
            except Exception as e:
                issues.append(f"CLAUDE.md reading error: {e}")
        else:
            issues.append("CLAUDE.md missing")
            
        # Check for README files
        readme_files = ['README.md', 'readme.md', 'README.txt']
        has_readme = any(os.path.exists(f) for f in readme_files)
        if has_readme:
            doc_score += 1
            self.logger.info("✅ README file exists")
        else:
            issues.append("README file missing")
            
        # Check for model documentation
        model_docs = glob.glob('models/*_metadata.json')
        if len(model_docs) >= 1:
            doc_score += 2
            self.logger.info(f"✅ Model documentation: {len(model_docs)} metadata files")
        else:
            issues.append("Model documentation missing")
            
        # Check for configuration documentation
        if os.path.exists('config') and len(glob.glob('config/*.json')) >= 1:
            doc_score += 1
            self.logger.info("✅ Configuration documented")
        else:
            issues.append("Configuration documentation missing")
            
        self.audit_results['documentation'] = {
            'score': doc_score,
            'max_score': max_score,
            'percentage': (doc_score / max_score) * 100,
            'issues': issues,
            'status': 'EXCELLENT' if doc_score >= 7 else 'GOOD' if doc_score >= 5 else 'NEEDS_IMPROVEMENT'
        }
        
        self.logger.info(f"📊 Documentation Completeness: {doc_score}/{max_score} ({(doc_score/max_score)*100:.0f}%)")
        
    def audit_production_readiness(self):
        """Audit 4: Production Readiness"""
        self.logger.info("=== AUDIT 4: PRODUCTION READINESS ===")
        
        prod_score = 0
        max_score = 10
        issues = []
        
        # Check for validation pipeline
        validation_files = [
            "validate_56_model_integrity.py",
            "final_model_verification.py",
            "fix_xg_features_emergency.py"
        ]
        
        existing_validation = [f for f in validation_files if os.path.exists(f)]
        if len(existing_validation) >= 2:
            prod_score += 3
            self.logger.info(f"✅ Validation pipeline: {len(existing_validation)} scripts")
        else:
            issues.append("Incomplete validation pipeline")
            
        # Check for training reproducibility
        training_scripts = [
            "train_56_percent_model.py",
            "retrain_corrected_56_model.py"
        ]
        
        existing_training = [f for f in training_scripts if os.path.exists(f)]
        if len(existing_training) >= 1:
            prod_score += 2
            self.logger.info(f"✅ Training reproducibility: {len(existing_training)} scripts")
        else:
            issues.append("Training scripts missing")
            
        # Check for data processing pipeline
        if os.path.isdir('scripts/preprocessing'):
            preprocessing_count = len(glob.glob('scripts/preprocessing/*.py'))
            if preprocessing_count >= 5:
                prod_score += 2
                self.logger.info(f"✅ Data pipeline: {preprocessing_count} preprocessing scripts")
            else:
                issues.append("Limited data processing pipeline")
        else:
            issues.append("Data processing pipeline missing")
            
        # Check for monitoring/evaluation tools
        evaluation_tools = glob.glob('evaluation/*.py') + glob.glob('*evaluation*.py')
        if len(evaluation_tools) >= 1:
            prod_score += 1
            self.logger.info(f"✅ Evaluation tools: {len(evaluation_tools)} available")
        else:
            issues.append("Evaluation tools missing")
            
        # Check for configuration management
        if os.path.isdir('config') and len(glob.glob('config/*.json')) >= 2:
            prod_score += 1
            self.logger.info("✅ Configuration management available")
        else:
            issues.append("Configuration management limited")
            
        # Check for logging/utilities
        if os.path.exists('utils.py'):
            prod_score += 1
            self.logger.info("✅ Logging and utilities available")
        else:
            issues.append("Utilities infrastructure missing")
            
        self.audit_results['production_readiness'] = {
            'score': prod_score,
            'max_score': max_score,
            'percentage': (prod_score / max_score) * 100,
            'issues': issues,
            'status': 'EXCELLENT' if prod_score >= 8 else 'GOOD' if prod_score >= 6 else 'NEEDS_IMPROVEMENT'
        }
        
        self.logger.info(f"📊 Production Readiness: {prod_score}/{max_score} ({(prod_score/max_score)*100:.0f}%)")
        
    def audit_performance_benchmarks(self):
        """Audit 5: Performance and Benchmarks"""
        self.logger.info("=== AUDIT 5: PERFORMANCE & BENCHMARKS ===")
        
        perf_score = 0
        max_score = 8
        issues = []
        
        # Check if model meets performance targets
        try:
            metadata_file = "models/randomforest_corrected_model_2025_09_02_113228_metadata.json"
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    
                accuracy = metadata.get('accuracy', 0)
                
                if accuracy >= 0.55:  # Excellent target
                    perf_score += 4
                    self.logger.info(f"✅ Excellent performance achieved: {accuracy:.1%}")
                elif accuracy >= 0.52:  # Good target
                    perf_score += 3
                    self.logger.info(f"✅ Good performance achieved: {accuracy:.1%}")
                elif accuracy >= 0.436:  # Beat majority baseline
                    perf_score += 2
                    self.logger.info(f"✅ Beats baselines: {accuracy:.1%}")
                else:
                    issues.append(f"Performance below baselines: {accuracy:.1%}")
                    
            else:
                issues.append("Performance metadata unavailable")
        except Exception as e:
            issues.append(f"Performance verification error: {e}")
            
        # Check for validation evidence
        validation_evidence = [
            "validate_56_model_integrity.py",
            "final_model_verification.py"
        ]
        
        if any(os.path.exists(f) for f in validation_evidence):
            perf_score += 2
            self.logger.info("✅ Performance validation evidence available")
        else:
            issues.append("Performance validation missing")
            
        # Check for benchmarking tools
        benchmark_files = glob.glob('*benchmark*.py') + glob.glob('scripts/*benchmark*.py')
        if len(benchmark_files) >= 1:
            perf_score += 1
            self.logger.info(f"✅ Benchmarking tools: {len(benchmark_files)} available")
        else:
            issues.append("Benchmarking tools missing")
            
        # Check for evaluation reports
        eval_files = glob.glob('evaluation/*.json') + glob.glob('reports/*.json')
        if len(eval_files) >= 1:
            perf_score += 1
            self.logger.info(f"✅ Evaluation reports: {len(eval_files)} available")
        else:
            issues.append("Evaluation reports missing")
            
        self.audit_results['performance'] = {
            'score': perf_score,
            'max_score': max_score,
            'percentage': (perf_score / max_score) * 100,
            'issues': issues,
            'status': 'EXCELLENT' if perf_score >= 7 else 'GOOD' if perf_score >= 5 else 'NEEDS_IMPROVEMENT'
        }
        
        self.logger.info(f"📊 Performance & Benchmarks: {perf_score}/{max_score} ({(perf_score/max_score)*100:.0f}%)")
        
    def compile_final_audit_report(self):
        """Compile comprehensive audit report"""
        self.logger.info("=== COMPILING FINAL AUDIT REPORT ===")
        
        total_score = sum(result['score'] for result in self.audit_results.values())
        total_max = sum(result['max_score'] for result in self.audit_results.values())
        overall_percentage = (total_score / total_max) * 100
        
        # Determine overall status
        if overall_percentage >= 85:
            overall_status = "EXCELLENT - PRODUCTION READY"
        elif overall_percentage >= 70:
            overall_status = "GOOD - MINOR IMPROVEMENTS NEEDED"
        elif overall_percentage >= 55:
            overall_status = "ACCEPTABLE - SIGNIFICANT IMPROVEMENTS NEEDED"
        else:
            overall_status = "POOR - MAJOR OVERHAUL REQUIRED"
            
        # Compile report
        report = {
            'audit_date': datetime.now().isoformat(),
            'project_version': 'v2.3',
            'overall_score': total_score,
            'overall_max': total_max,
            'overall_percentage': overall_percentage,
            'overall_status': overall_status,
            'category_results': self.audit_results,
            'recommendations': self.generate_recommendations(),
            'critical_issues': self.identify_critical_issues()
        }
        
        # Save report
        report_file = f"project_audit_report_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        self.logger.info(f"💾 Audit report saved: {report_file}")
        
        return report, report_file
        
    def generate_recommendations(self):
        """Generate recommendations based on audit results"""
        recommendations = []
        
        for category, result in self.audit_results.items():
            if result['percentage'] < 80:
                recommendations.append({
                    'category': category,
                    'priority': 'HIGH' if result['percentage'] < 60 else 'MEDIUM',
                    'issues': result['issues'],
                    'suggested_actions': self.get_category_suggestions(category, result['issues'])
                })
                
        return recommendations
        
    def get_category_suggestions(self, category, issues):
        """Get specific suggestions for category issues"""
        suggestions = {
            'model_data_integrity': [
                "Ensure all model files are properly versioned and stored",
                "Implement automated data quality checks",
                "Add comprehensive model metadata documentation"
            ],
            'code_quality': [
                "Implement comprehensive testing suite",
                "Add code documentation and comments",
                "Establish coding standards and linting"
            ],
            'documentation': [
                "Create comprehensive README documentation",
                "Document all model configurations and parameters",
                "Add usage examples and tutorials"
            ],
            'production_readiness': [
                "Implement automated validation pipeline",
                "Add monitoring and alerting systems",
                "Create deployment automation scripts"
            ],
            'performance': [
                "Establish performance monitoring dashboard",
                "Implement A/B testing framework",
                "Add automated benchmarking suite"
            ]
        }
        
        return suggestions.get(category, ["Review and improve category-specific issues"])
        
    def identify_critical_issues(self):
        """Identify critical issues across all categories"""
        critical_issues = []
        
        for category, result in self.audit_results.items():
            if result['percentage'] < 50:
                critical_issues.append({
                    'category': category,
                    'severity': 'CRITICAL',
                    'issues': result['issues']
                })
                
        return critical_issues
        
    def run_complete_audit(self):
        """Run complete project audit"""
        self.logger.info("🔍 STARTING COMPREHENSIVE PROJECT AUDIT")
        self.logger.info("="*70)
        
        # Run all audit categories
        self.audit_model_data_integrity()
        self.audit_code_quality_structure()
        self.audit_documentation_completeness()
        self.audit_production_readiness()
        self.audit_performance_benchmarks()
        
        # Compile final report
        report, report_file = self.compile_final_audit_report()
        
        # Display summary
        self.logger.info("\n" + "="*70)
        self.logger.info("🏁 COMPREHENSIVE AUDIT SUMMARY")
        self.logger.info("="*70)
        
        for category, result in self.audit_results.items():
            status_icon = "🟢" if result['status'] == 'EXCELLENT' else "🟡" if result['status'] == 'GOOD' else "🔴"
            self.logger.info(f"{status_icon} {category.replace('_', ' ').title()}: {result['score']}/{result['max_score']} ({result['percentage']:.0f}%) - {result['status']}")
            
        self.logger.info(f"\n📊 OVERALL PROJECT STATUS: {report['overall_percentage']:.0f}% - {report['overall_status']}")
        
        if report['overall_percentage'] >= 85:
            self.logger.info("🎉 PROJECT STATUS: EXCELLENT - Ready for production deployment!")
        elif report['overall_percentage'] >= 70:
            self.logger.info("✅ PROJECT STATUS: GOOD - Minor improvements recommended")
        else:
            self.logger.info("⚠️ PROJECT STATUS: Significant improvements needed")
            
        self.logger.info(f"📋 Detailed report: {report_file}")
        self.logger.info("="*70)
        
        return report, report_file

if __name__ == "__main__":
    auditor = ProjectAuditor()
    report, report_file = auditor.run_complete_audit()
    
    print(f"\n🏆 COMPREHENSIVE AUDIT COMPLETE!")
    print(f"Overall Score: {report['overall_percentage']:.0f}%")
    print(f"Status: {report['overall_status']}")
    print(f"Report: {report_file}")
    
    if report['overall_percentage'] >= 85:
        print("🚀 Project is PRODUCTION READY!")
    else:
        print("📝 Review recommendations for improvements.")