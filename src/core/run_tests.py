#!/usr/bin/env python3
"""
Test runner for Oddsy project
Run all automated tests to validate data quality and ML pipeline
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle output"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"❌ {description} - ERROR: {str(e)}")
        return False

def main():
    """Main test runner"""
    print("🧪 Oddsy ML Pipeline Test Suite")
    print("=" * 60)
    
    # Ensure we're in the right directory
    if not os.path.exists('config/config.json'):
        print("❌ Error: Not in Oddsy project directory (config/config.json not found)")
        sys.exit(1)
    
    # Test results
    test_results = []
    
    # 1. Run unit tests
    test_results.append(
        run_command(
            [sys.executable, 'tests/test_data_quality.py'],
            "Unit Tests - Data Quality & Validation"
        )
    )
    
    # 2. Test data preparation (dry run)
    if os.path.exists('data/processed/premier_league_ml_ready.csv'):
        test_results.append(
            run_command(
                [sys.executable, 'prepare_ml_data.py'],
                "Data Preparation Pipeline"
            )
        )
    else:
        print("⚠️  Skipping data preparation test - source file not found")
        print("   Expected: data/processed/premier_league_ml_ready.csv")
    
    # 3. Validate feature engineering output
    if os.path.exists('validate_features.py'):
        test_results.append(
            run_command(
                [sys.executable, 'validate_features.py'],
                "Feature Engineering Validation"
            )
        )
    
    # 4. Test configuration loading
    test_config = """
import json
from utils import load_config, setup_logging

try:
    config = load_config()
    logger = setup_logging()
    
    required_sections = ['data', 'model', 'training', 'evaluation', 'versioning', 'logging']
    for section in required_sections:
        assert section in config, f"Missing config section: {section}"
    
    logger.info("Configuration validation passed")
    print("✅ Configuration validation - PASSED")
    
except Exception as e:
    print(f"❌ Configuration validation - FAILED: {e}")
    exit(1)
    """
    
    # Write and run config test
    with open('temp_config_test.py', 'w') as f:
        f.write(test_config)
    
    test_results.append(
        run_command(
            [sys.executable, 'temp_config_test.py'],
            "Configuration Loading & Validation"
        )
    )
    
    # Clean up temp file
    if os.path.exists('temp_config_test.py'):
        os.unlink('temp_config_test.py')
    
    # Summary
    print(f"\n{'='*60}")
    print("🏁 TEST SUITE SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Pipeline is ready for production!")
        return 0
    else:
        print("💥 SOME TESTS FAILED - Fix issues before proceeding")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)