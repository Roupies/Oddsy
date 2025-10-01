#!/usr/bin/env python3
"""
🔄 DASHBOARD DATA UPDATER
========================
Automation script for regular dashboard data updates.
Run weekly or after significant EPL match results.
"""

import subprocess
import sys
import os
import json
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dashboard_update.log'),
        logging.StreamHandler()
    ]
)

def run_command(command: str) -> bool:
    """Run shell command and return success status."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            logging.info(f"✅ Success: {command}")
            return True
        else:
            logging.error(f"❌ Failed: {command}")
            logging.error(f"Error: {result.stderr}")
            return False
    except Exception as e:
        logging.error(f"❌ Exception running {command}: {e}")
        return False

def validate_files() -> bool:
    """Validate that all required files exist."""
    required_files = [
        "data/dashboard/real_predictions.json",
        "data/dashboard/real_performance.json", 
        "data/dashboard/real_metrics.json"
    ]
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    for file_path in required_files:
        full_path = os.path.join(project_root, file_path)
        if not os.path.exists(full_path):
            logging.error(f"❌ Missing required file: {full_path}")
            return False
    
    logging.info("✅ All required files exist")
    return True

def check_data_freshness() -> dict:
    """Check how fresh the current data is."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    performance_file = os.path.join(project_root, "data/dashboard/real_performance.json")
    
    try:
        with open(performance_file, 'r') as f:
            data = json.load(f)
        
        timestamp_str = data.get('timestamp', '')
        if timestamp_str:
            data_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            age_hours = (datetime.now() - data_time.replace(tzinfo=None)).total_seconds() / 3600
            
            return {
                'timestamp': timestamp_str,
                'age_hours': age_hours,
                'is_fresh': age_hours < 24  # Consider fresh if less than 24 hours old
            }
    except Exception as e:
        logging.error(f"❌ Error checking data freshness: {e}")
    
    return {'is_fresh': False, 'age_hours': 999}

def main():
    """Main update process."""
    logging.info("🔄 Starting Dashboard Data Update")
    logging.info("=" * 50)
    
    # Check current data freshness
    freshness = check_data_freshness()
    logging.info(f"📊 Current data age: {freshness.get('age_hours', 'unknown'):.1f} hours")
    
    if freshness.get('is_fresh', False):
        logging.info("✅ Data is fresh, skipping update")
        return True
    
    # Change to project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(project_root)
    logging.info(f"📁 Working directory: {project_root}")
    
    # Step 1: Generate new production data
    logging.info("🏭 Generating new production predictions...")
    if not run_command("python3 scripts/production/generate_real_predictions.py"):
        logging.error("❌ Failed to generate production data")
        return False
    
    # Step 2: Validate generated files
    if not validate_files():
        logging.error("❌ File validation failed")
        return False
    
    # Step 3: Test data loader
    logging.info("🧪 Testing data loader...")
    if not run_command("python3 dashboards/core/data_loader.py"):
        logging.error("❌ Data loader test failed")
        return False
    
    # Step 4: Quick dashboard test (optional)
    logging.info("🎯 Dashboard update completed successfully!")
    
    # Report results
    new_freshness = check_data_freshness()
    logging.info(f"📊 New data timestamp: {new_freshness.get('timestamp', 'unknown')}")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            logging.info("🎉 Dashboard data update completed successfully!")
            sys.exit(0)
        else:
            logging.error("💥 Dashboard data update failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        logging.info("⏹️ Update cancelled by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"💥 Unexpected error: {e}")
        sys.exit(1)