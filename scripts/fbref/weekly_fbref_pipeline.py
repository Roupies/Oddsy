"""
Pipeline FBref Hebdomadaire - Automatisation complète
=====================================================
Pipeline automatisé pour collecte, fusion et intégration FBref
Execution: hebdomadaire (dimanche soir post-journée EPL)
"""

import subprocess
import pandas as pd
import os
import json
from datetime import datetime, timedelta
import time
import sys

# Ajouter répertoire script pour imports
sys.path.append('/Users/maxime/Desktop/Oddsy/scripts/fbref')
from fbref_data_fusion import FBrefDataFusion

class WeeklyFBrefPipeline:
    """Pipeline automatisé FBref hebdomadaire"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = "/Users/maxime/Desktop/Oddsy"
        self.fbref_dir = f"{self.base_dir}/data/fbref"
        self.processed_dir = f"{self.base_dir}/data/processed"
        self.logs_dir = f"{self.base_dir}/logs"
        
        # Statistiques pipeline
        self.pipeline_stats = {
            'start_time': datetime.now().isoformat(),
            'steps_completed': [],
            'steps_failed': [],
            'files_generated': [],
            'errors': []
        }
        
        # Créer répertoires si nécessaire
        os.makedirs(self.fbref_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
    
    def log(self, message, level="INFO"):
        """Log avec timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        
        # Log vers fichier
        log_file = f"{self.logs_dir}/fbref_pipeline_{self.timestamp[:8]}.log"
        with open(log_file, 'a') as f:
            f.write(f"[{timestamp}] {level}: {message}\n")
    
    def step1_extract_fbref_data(self):
        """Étape 1: Extraction données FBref via R"""
        self.log("=== ÉTAPE 1: EXTRACTION FBREF ===")
        
        try:
            # Vérifier R et packages
            result = subprocess.run(['which', 'R'], capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception("R non trouvé")
            
            self.log("✅ R disponible")
            
            # Exécuter script extraction FBref
            r_script = f"{self.base_dir}/scripts/fbref/extract_epl_data.R"
            
            self.log(f"🔄 Exécution script R: {r_script}")
            
            # Changer vers répertoire base pour paths relatifs
            original_cwd = os.getcwd()
            os.chdir(self.base_dir)
            
            try:
                result = subprocess.run(
                    ['Rscript', r_script],
                    capture_output=True,
                    text=True,
                    timeout=1800  # 30 minutes max
                )
                
                if result.returncode == 0:
                    self.log("✅ Extraction FBref réussie")
                    self.log(f"📋 Output R: {result.stdout[:500]}...")  # Premier 500 chars
                    self.pipeline_stats['steps_completed'].append('extract_fbref')
                    return True
                else:
                    error_msg = f"Erreur extraction FBref: {result.stderr}"
                    self.log(error_msg, "ERROR")
                    self.pipeline_stats['steps_failed'].append('extract_fbref')
                    self.pipeline_stats['errors'].append(error_msg)
                    return False
                    
            finally:
                os.chdir(original_cwd)
            
        except Exception as e:
            error_msg = f"Exception extraction FBref: {e}"
            self.log(error_msg, "ERROR")
            self.pipeline_stats['steps_failed'].append('extract_fbref')
            self.pipeline_stats['errors'].append(error_msg)
            return False
    
    def step2_find_latest_files(self):
        """Étape 2: Identifier fichiers les plus récents"""
        self.log("=== ÉTAPE 2: IDENTIFICATION FICHIERS ===")
        
        try:
            # Chercher fichiers FBref les plus récents
            fbref_files = [f for f in os.listdir(self.fbref_dir) if f.endswith('.csv')]
            
            if not fbref_files:
                raise Exception("Aucun fichier FBref trouvé")
            
            # Trier par nom (contient timestamp)
            fbref_files.sort(reverse=True)
            
            # Prendre team_logs le plus récent
            team_logs_files = [f for f in fbref_files if 'team_logs' in f]
            if not team_logs_files:
                raise Exception("Aucun fichier team_logs trouvé")
            
            latest_team_logs = team_logs_files[0]
            self.latest_fbref_path = f"{self.fbref_dir}/{latest_team_logs}"
            
            self.log(f"📊 FBref le plus récent: {latest_team_logs}")
            
            # Chercher Football-Data le plus récent
            raw_dir = f"{self.base_dir}/data/raw"
            e0_files = [f for f in os.listdir(raw_dir) if f.startswith('E0') and f.endswith('.csv')]
            
            if not e0_files:
                raise Exception("Aucun fichier Football-Data E0 trouvé")
            
            # Trier par nom
            e0_files.sort(reverse=True)
            latest_e0 = e0_files[0]
            self.latest_football_data_path = f"{raw_dir}/{latest_e0}"
            
            self.log(f"📊 Football-Data le plus récent: {latest_e0}")
            
            self.pipeline_stats['steps_completed'].append('find_files')
            return True
            
        except Exception as e:
            error_msg = f"Exception identification fichiers: {e}"
            self.log(error_msg, "ERROR")
            self.pipeline_stats['steps_failed'].append('find_files')
            self.pipeline_stats['errors'].append(error_msg)
            return False
    
    def step3_fusion_data(self):
        """Étape 3: Fusion FBref + Football-Data"""
        self.log("=== ÉTAPE 3: FUSION DONNÉES ===")
        
        try:
            # Initialiser fusionneur
            fusion = FBrefDataFusion()
            
            # Chemin sortie
            output_path = f"{self.processed_dir}/epl_2025_26_enhanced_fbref_{self.timestamp}.csv"
            
            # Exécuter fusion
            self.log("🔗 Démarrage fusion...")
            result_path = fusion.process_fusion(
                self.latest_football_data_path,
                self.latest_fbref_path,
                output_path
            )
            
            if result_path:
                self.log(f"✅ Fusion réussie: {result_path}")
                self.enhanced_data_path = result_path
                self.pipeline_stats['files_generated'].append(result_path)
                self.pipeline_stats['steps_completed'].append('fusion_data')
                
                # Créer lien symbolique vers "latest"
                latest_link = f"{self.processed_dir}/epl_2025_26_enhanced_fbref_latest.csv"
                if os.path.exists(latest_link):
                    os.remove(latest_link)
                os.symlink(os.path.basename(result_path), latest_link)
                self.log(f"🔗 Lien latest créé: {latest_link}")
                
                return True
            else:
                raise Exception("Fusion échouée")
                
        except Exception as e:
            error_msg = f"Exception fusion données: {e}"
            self.log(error_msg, "ERROR")
            self.pipeline_stats['steps_failed'].append('fusion_data')
            self.pipeline_stats['errors'].append(error_msg)
            return False
    
    def step4_validate_data(self):
        """Étape 4: Validation données fusionnées"""
        self.log("=== ÉTAPE 4: VALIDATION DONNÉES ===")
        
        try:
            # Charger données fusionnées
            df = pd.read_csv(self.enhanced_data_path)
            
            # Validations de base
            validations = {
                'total_matches': len(df),
                'date_range': {
                    'start': df['Date'].min() if 'Date' in df.columns else None,
                    'end': df['Date'].max() if 'Date' in df.columns else None
                },
                'fbref_columns': len([col for col in df.columns if any(x in col for x in ['xG', 'Shots', 'Corner', 'H_', 'A_'])]),
                'missing_values': df.isnull().sum().sum(),
                'duplicates': df.duplicated().sum()
            }
            
            self.log(f"📊 Validation: {validations['total_matches']} matchs")
            self.log(f"📊 Colonnes FBref: {validations['fbref_columns']}")
            self.log(f"📊 Valeurs manquantes: {validations['missing_values']}")
            self.log(f"📊 Doublons: {validations['duplicates']}")
            
            # Seuils de validation
            if validations['total_matches'] < 50:
                raise Exception(f"Trop peu de matchs: {validations['total_matches']}")
            
            if validations['fbref_columns'] < 3:
                self.log("⚠️ Peu de colonnes FBref intégrées", "WARNING")
            
            # Sauvegarder rapport validation
            validation_path = self.enhanced_data_path.replace('.csv', '_validation.json')
            with open(validation_path, 'w') as f:
                json.dump(validations, f, indent=2, default=str)
            
            self.log(f"📋 Rapport validation: {validation_path}")
            self.pipeline_stats['files_generated'].append(validation_path)
            self.pipeline_stats['steps_completed'].append('validate_data')
            
            return True
            
        except Exception as e:
            error_msg = f"Exception validation données: {e}"
            self.log(error_msg, "ERROR")
            self.pipeline_stats['steps_failed'].append('validate_data')
            self.pipeline_stats['errors'].append(error_msg)
            return False
    
    def step5_generate_report(self):
        """Étape 5: Génération rapport final"""
        self.log("=== ÉTAPE 5: RAPPORT FINAL ===")
        
        try:
            # Finaliser stats pipeline
            self.pipeline_stats['end_time'] = datetime.now().isoformat()
            self.pipeline_stats['duration_minutes'] = (
                datetime.fromisoformat(self.pipeline_stats['end_time']) - 
                datetime.fromisoformat(self.pipeline_stats['start_time'])
            ).total_seconds() / 60
            
            self.pipeline_stats['success'] = len(self.pipeline_stats['steps_failed']) == 0
            
            # Sauvegarder rapport
            report_path = f"{self.logs_dir}/fbref_pipeline_report_{self.timestamp}.json"
            with open(report_path, 'w') as f:
                json.dump(self.pipeline_stats, f, indent=2, default=str)
            
            self.log(f"📋 Rapport final: {report_path}")
            
            # Résumé console
            success_count = len(self.pipeline_stats['steps_completed'])
            total_steps = success_count + len(self.pipeline_stats['steps_failed'])
            
            self.log(f"📊 RÉSUMÉ: {success_count}/{total_steps} étapes réussies")
            self.log(f"⏱️ Durée: {self.pipeline_stats['duration_minutes']:.1f} minutes")
            
            if self.pipeline_stats['success']:
                self.log("✅ Pipeline complète RÉUSSIE")
            else:
                self.log("❌ Pipeline ÉCHOUÉE avec erreurs")
            
            return True
            
        except Exception as e:
            error_msg = f"Exception génération rapport: {e}"
            self.log(error_msg, "ERROR")
            return False
    
    def run_complete_pipeline(self):
        """Exécute pipeline complète"""
        self.log("🚀 DÉMARRAGE PIPELINE FBREF HEBDOMADAIRE")
        self.log(f"📅 Timestamp: {self.timestamp}")
        
        # Étapes pipeline
        steps = [
            ("Extraction FBref", self.step1_extract_fbref_data),
            ("Identification fichiers", self.step2_find_latest_files),
            ("Fusion données", self.step3_fusion_data),
            ("Validation données", self.step4_validate_data),
            ("Génération rapport", self.step5_generate_report)
        ]
        
        # Exécuter étapes
        for step_name, step_func in steps:
            self.log(f"🔄 {step_name}...")
            
            try:
                success = step_func()
                if not success:
                    self.log(f"❌ Échec: {step_name}")
                    break
                else:
                    self.log(f"✅ Succès: {step_name}")
                    
            except Exception as e:
                self.log(f"❌ Exception {step_name}: {e}", "ERROR")
                break
        
        # Rapport final toujours généré
        self.step5_generate_report()
        
        return self.pipeline_stats['success']

def main():
    """Point d'entrée pipeline"""
    pipeline = WeeklyFBrefPipeline()
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n🎉 Pipeline FBref hebdomadaire TERMINÉE avec succès!")
        exit(0)
    else:
        print("\n💥 Pipeline FBref hebdomadaire ÉCHOUÉE")
        exit(1)

if __name__ == "__main__":
    main()