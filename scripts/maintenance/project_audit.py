import os
import pandas as pd
from datetime import datetime
import json

def audit_project():
    """
    Professional project audit - identify structure, redundancies, and cleanup opportunities
    """
    
    print("=== 🔍 PROJECT AUDIT REPORT ===")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Directory structure analysis
    print("📁 DIRECTORY STRUCTURE:")
    directories = {
        "data/raw": [],
        "data/processed": [],
        "config": [],
        "models": [],
        "scripts": [],
        "evaluation": [],
        "tests": []
    }
    
    # Collect files by directory
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith(('.py', '.json', '.csv', '.md')):
                file_path = os.path.join(root, file)
                for dir_key in directories.keys():
                    if file_path.startswith(f'./{dir_key}') or file_path.startswith(f'.\\{dir_key}'):
                        directories[dir_key].append(file_path)
                        break
                else:
                    # Root files
                    if root == '.':
                        if "root" not in directories:
                            directories["root"] = []
                        directories["root"].append(file_path)
    
    # Report by directory
    total_files = 0
    for dir_name, files in directories.items():
        if files:
            print(f"  {dir_name}: {len(files)} files")
            total_files += len(files)
    
    print(f"  TOTAL: {total_files} files")
    print()
    
    # Data analysis
    print("📊 DATA FILES ANALYSIS:")
    data_files = directories.get("data/processed", []) + directories.get("data/raw", [])
    csv_files = [f for f in data_files if f.endswith('.csv')]
    
    print(f"  Total CSV files: {len(csv_files)}")
    
    # Check for versioned duplicates
    versioned_patterns = {}
    for file in csv_files:
        base_name = file.split('/')[-1]
        # Remove version patterns like _v2025_08_30_HHMMSS
        import re
        clean_name = re.sub(r'_v?\d{4}_\d{2}_\d{2}_\d{6}', '', base_name)
        clean_name = re.sub(r'_\d{4}_\d{2}_\d{2}', '', clean_name)
        
        if clean_name not in versioned_patterns:
            versioned_patterns[clean_name] = []
        versioned_patterns[clean_name].append(file)
    
    print("  Potential duplicate versions:")
    duplicates_found = 0
    for pattern, files in versioned_patterns.items():
        if len(files) > 1:
            duplicates_found += len(files) - 1
            print(f"    {pattern}: {len(files)} versions")
            for file in sorted(files):
                size = "N/A"
                try:
                    size = f"{os.path.getsize(file)//1024}KB"
                except:
                    pass
                print(f"      - {file} ({size})")
    
    if duplicates_found == 0:
        print("    ✅ No obvious duplicates found")
    print()
    
    # Config files analysis
    print("⚙️ CONFIG FILES ANALYSIS:")
    config_files = directories.get("config", [])
    print(f"  Total config files: {len(config_files)}")
    
    # Group by type
    config_types = {}
    for file in config_files:
        if 'features' in file:
            config_types.setdefault('Features', []).append(file)
        elif 'metadata' in file:
            config_types.setdefault('Metadata', []).append(file)
        else:
            config_types.setdefault('Other', []).append(file)
    
    for config_type, files in config_types.items():
        print(f"    {config_type}: {len(files)} files")
    print()
    
    # Models analysis
    print("🤖 MODELS ANALYSIS:")
    model_files = directories.get("models", [])
    print(f"  Total model files: {len(model_files)}")
    
    # Check for actual model files vs metadata
    actual_models = [f for f in model_files if f.endswith('.joblib') or f.endswith('.pkl')]
    metadata_files = [f for f in model_files if f.endswith('.json')]
    
    print(f"    Trained models: {len(actual_models)}")
    print(f"    Metadata files: {len(metadata_files)}")
    
    if len(actual_models) == 0:
        print("    ⚠️ No trained model files found (.joblib/.pkl)")
    print()
    
    # Scripts analysis
    print("🛠️ SCRIPTS ANALYSIS:")
    script_files = directories.get("scripts", [])
    print(f"  Total script files: {len(script_files)}")
    
    # Categorize scripts
    script_categories = {
        "preprocessing": [],
        "modeling": [],
        "analysis": [],
        "exploration": []
    }
    
    for file in script_files:
        for category in script_categories.keys():
            if f'/{category}/' in file or f'\\{category}\\' in file:
                script_categories[category].append(file)
                break
    
    for category, files in script_categories.items():
        print(f"    {category}: {len(files)} files")
    print()
    
    # Identify core vs experimental files
    print("🎯 FILE CLASSIFICATION:")
    
    core_files = [
        './CLAUDE.md',
        './utils.py',
        './config/config.json',
        './data/raw/premier_league_2019_2024.csv',
    ]
    
    # Find the latest/best versions
    latest_processed = None
    latest_model_metadata = None
    
    # Find most recent processed data
    processed_files = [f for f in directories.get("data/processed", []) if 'premier_league' in f and 'csv' in f]
    if processed_files:
        latest_processed = max(processed_files, key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0)
    
    # Find most recent model metadata
    if metadata_files:
        latest_model_metadata = max(metadata_files, key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0)
    
    print("  CORE FILES (essential):")
    essential_files = [
        './CLAUDE.md',
        './utils.py', 
        './config/config.json',
        latest_processed,
        latest_model_metadata
    ]
    
    for file in essential_files:
        if file and os.path.exists(file):
            print(f"    ✅ {file}")
        else:
            print(f"    ❌ {file} (missing)")
    
    print(f"\n  EXPERIMENTAL FILES (can be archived): ~{total_files - len(essential_files)} files")
    print()
    
    # Cleanup recommendations
    print("🧹 CLEANUP RECOMMENDATIONS:")
    
    recommendations = []
    
    # Old versioned files
    if duplicates_found > 0:
        recommendations.append(f"Archive {duplicates_found} old versioned files")
    
    # Experimental scripts
    experimental_scripts = len(script_files) - 5  # Keep ~5 core scripts
    if experimental_scripts > 0:
        recommendations.append(f"Archive {experimental_scripts} experimental scripts")
    
    # Old config files  
    old_configs = len(config_files) - 3  # Keep ~3 core configs
    if old_configs > 0:
        recommendations.append(f"Archive {old_configs} old config files")
    
    if recommendations:
        for rec in recommendations:
            print(f"  • {rec}")
    else:
        print("  ✅ Project structure looks clean")
    
    print("\n=== END PROJECT AUDIT ===")
    
    # Return summary for further processing
    return {
        'total_files': total_files,
        'duplicates_found': duplicates_found,
        'essential_files': [f for f in essential_files if f and os.path.exists(f)],
        'recommendations': recommendations,
        'latest_processed': latest_processed,
        'latest_model_metadata': latest_model_metadata
    }

if __name__ == "__main__":
    audit_result = audit_project()
    print(f"\n📋 AUDIT SUMMARY:")
    print(f"Total files: {audit_result['total_files']}")
    print(f"Essential files: {len(audit_result['essential_files'])}")
    print(f"Cleanup opportunities: {len(audit_result['recommendations'])}")