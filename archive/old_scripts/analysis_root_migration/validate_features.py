import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class OddsyFeatureValidator:
    """
    Valide que nos features font du SENS dans le contexte football
    """
    
    def __init__(self):
        self.df = None
        self.issues_found = []
        
    def load_data(self, filepath):
        """Load the ML-ready dataset"""
        print("🔍 ODDSY FEATURE VALIDATION")
        print("=" * 50)
        
        self.df = pd.read_csv(filepath)
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            
        print(f"Dataset loaded: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        return self
    
    def test_basic_data_quality(self):
        """Test 1: Vérifier la qualité de base des données"""
        print(f"\n📊 TEST 1: QUALITÉ DES DONNÉES")
        print("-" * 30)
        
        issues = []
        
        # Test des valeurs manquantes
        missing = self.df.isnull().sum()
        missing_cols = missing[missing > 0]
        if len(missing_cols) > 0:
            issues.append(f"❌ Valeurs manquantes trouvées: {dict(missing_cols)}")
        else:
            print("✅ Aucune valeur manquante")
            
        # Test des features normalisées (doivent être entre 0 et 1)
        feature_cols = [col for col in self.df.columns if 'normalized' in col or col in ['home_form', 'away_form', 'h2h_score', 'home_advantage']]
        
        for col in feature_cols:
            if col in self.df.columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if min_val < -0.01 or max_val > 1.01:  # Small tolerance
                    issues.append(f"❌ {col}: range [{min_val:.3f}, {max_val:.3f}] - pas entre 0-1")
                else:
                    print(f"✅ {col}: range [{min_val:.3f}, {max_val:.3f}]")
        
        # Test des valeurs infinies
        inf_cols = []
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if np.isinf(self.df[col]).any():
                inf_cols.append(col)
        
        if inf_cols:
            issues.append(f"❌ Valeurs infinies dans: {inf_cols}")
        else:
            print("✅ Aucune valeur infinie")
            
        self.issues_found.extend(issues)
        return self
    
    def test_feature_logic(self):
        """Test 2: Vérifier la logique des features"""
        print(f"\n🧠 TEST 2: LOGIQUE DES FEATURES")
        print("-" * 30)
        
        # Test: home_advantage devrait toujours être 1
        if 'home_advantage' in self.df.columns:
            unique_vals = self.df['home_advantage'].unique()
            if len(unique_vals) == 1 and unique_vals[0] == 1:
                print("✅ home_advantage = 1 partout (correct)")
            else:
                self.issues_found.append(f"❌ home_advantage a des valeurs bizarres: {unique_vals}")
        
        # Test: form_diff_normalized vs individual forms
        if all(col in self.df.columns for col in ['home_form', 'away_form', 'form_diff_normalized']):
            # Calculer form_diff à partir des formes individuelles
            calculated_diff = (self.df['home_form'] - self.df['away_form'] + 1) / 2
            actual_diff = self.df['form_diff_normalized']
            
            diff_error = abs(calculated_diff - actual_diff).max()
            if diff_error < 0.01:
                print("✅ form_diff_normalized cohérent avec home_form - away_form")
            else:
                self.issues_found.append(f"❌ form_diff_normalized incohérent (erreur max: {diff_error:.3f})")
        
        return self
    
    def test_known_matches(self):
        """Test 3: Valider sur des matchs connus où on connaît le contexte"""
        print(f"\n🏟️ TEST 3: VALIDATION SUR MATCHS CONNUS")
        print("-" * 30)
        
        if not all(col in self.df.columns for col in ['Date', 'HomeTeam', 'AwayTeam']):
            print("⚠️ Pas de colonnes Date/Teams - skip test matchs connus")
            return self
        
        # Test cases avec contexte connu
        test_cases = [
            {
                'name': 'Liverpool (fort) vs Norwich (faible) - début 2019-20',
                'home': 'Liverpool',
                'away': 'Norwich',
                'date_range': ('2019-08-01', '2019-09-01'),
                'expected': {
                    'elo_diff_normalized': (0.6, 1.0),  # Liverpool devrait être favori
                    'home_advantage': 1.0,
                    'result_expected': 'H'
                }
            },
            {
                'name': 'Man City (fort) vs équipe quelconque',
                'home': 'Man City',
                'away': None,  # N'importe quelle équipe
                'date_range': ('2020-01-01', '2020-12-31'),
                'expected': {
                    'elo_diff_normalized': (0.55, 1.0),  # City souvent favori à domicile
                }
            }
        ]
        
        for test_case in test_cases:
            print(f"\n🔍 Test: {test_case['name']}")
            
            # Trouver les matchs correspondants
            mask = (
                (self.df['HomeTeam'] == test_case['home']) &
                (self.df['Date'] >= test_case['date_range'][0]) &
                (self.df['Date'] <= test_case['date_range'][1])
            )
            
            if test_case['away']:
                mask &= (self.df['AwayTeam'] == test_case['away'])
            
            matches = self.df[mask]
            
            if len(matches) == 0:
                print(f"  ⚠️ Aucun match trouvé pour ce test")
                continue
                
            print(f"  📊 {len(matches)} match(s) trouvé(s)")
            
            # Vérifier les expectations
            for feature, expected in test_case['expected'].items():
                if feature == 'result_expected':
                    continue
                    
                # Handle both tuple (min, max) and single value expectations
                if isinstance(expected, tuple):
                    min_exp, max_exp = expected
                else:
                    min_exp = max_exp = expected
                    
                if feature in matches.columns:
                    actual_values = matches[feature]
                    avg_val = actual_values.mean()
                    
                    if min_exp <= avg_val <= max_exp:
                        print(f"  ✅ {feature}: {avg_val:.3f} (dans range attendu [{min_exp:.1f}-{max_exp:.1f}])")
                    else:
                        print(f"  ❌ {feature}: {avg_val:.3f} (HORS range attendu [{min_exp:.1f}-{max_exp:.1f}])")
                        self.issues_found.append(f"Match test '{test_case['name']}': {feature} = {avg_val:.3f} (attendu [{min_exp:.1f}-{max_exp:.1f}])")
                        
            # Montrer quelques exemples
            if len(matches) > 0:
                example = matches.iloc[0]
                print(f"  📋 Exemple: {example['HomeTeam']} vs {example['AwayTeam']} ({example['Date'].strftime('%Y-%m-%d') if 'Date' in example else 'date inconnue'})")
                
                feature_cols = ['elo_diff_normalized', 'form_diff_normalized', 'h2h_score']
                for col in feature_cols:
                    if col in example:
                        print(f"       {col}: {example[col]:.3f}")
        
        return self
    
    def test_distribution_sanity(self):
        """Test 4: Vérifier que les distributions sont sensées"""
        print(f"\n📈 TEST 4: DISTRIBUTIONS DES FEATURES")
        print("-" * 30)
        
        feature_cols = [col for col in self.df.columns if any(x in col for x in ['form', 'elo', 'h2h', 'advantage', 'normalized'])]
        
        for col in feature_cols[:5]:  # Top 5 features
            if col in self.df.columns:
                data = self.df[col]
                
                print(f"\n📊 {col}:")
                print(f"   Mean: {data.mean():.3f}")
                print(f"   Std:  {data.std():.3f}")
                print(f"   Min:  {data.min():.3f}")
                print(f"   Max:  {data.max():.3f}")
                
                # Test si trop de valeurs identiques (suspect)
                value_counts = data.value_counts()
                most_common_pct = value_counts.iloc[0] / len(data) * 100
                
                if most_common_pct > 50:
                    self.issues_found.append(f"❌ {col}: {most_common_pct:.1f}% des valeurs identiques (valeur: {value_counts.index[0]})")
                    print(f"   ⚠️ {most_common_pct:.1f}% des valeurs = {value_counts.index[0]}")
                else:
                    print(f"   ✅ Distribution variée (valeur max: {most_common_pct:.1f}%)")
        
        return self
    
    def test_target_distribution(self):
        """Test 5: Vérifier la distribution des résultats"""
        print(f"\n🎯 TEST 5: DISTRIBUTION DES RÉSULTATS")
        print("-" * 30)
        
        if 'FullTimeResult' in self.df.columns:
            target_counts = self.df['FullTimeResult'].value_counts()
            total = len(self.df)
            
            print("Distribution des résultats:")
            for result in ['H', 'D', 'A']:
                count = target_counts.get(result, 0)
                pct = count / total * 100
                print(f"  {result}: {count} matchs ({pct:.1f}%)")
            
            # Vérifier si trop déséquilibré
            min_pct = min([target_counts.get(r, 0) / total * 100 for r in ['H', 'D', 'A']])
            max_pct = max([target_counts.get(r, 0) / total * 100 for r in ['H', 'D', 'A']])
            
            if max_pct - min_pct > 20:  # Plus de 20% d'écart
                self.issues_found.append(f"❌ Dataset déséquilibré: écart {max_pct - min_pct:.1f}% entre min et max")
            else:
                print(f"✅ Distribution équilibrée (écart: {max_pct - min_pct:.1f}%)")
        
        return self
    
    def generate_report(self):
        """Générer le rapport final"""
        print(f"\n📋 RAPPORT FINAL DE VALIDATION")
        print("=" * 50)
        
        if len(self.issues_found) == 0:
            print("🎉 TOUTES LES VALIDATIONS PASSÉES !")
            print("✅ Dataset prêt pour le Machine Learning")
        else:
            print(f"⚠️ {len(self.issues_found)} PROBLÈME(S) DÉTECTÉ(S):")
            for i, issue in enumerate(self.issues_found, 1):
                print(f"{i:2d}. {issue}")
            
            print(f"\n🔧 Action requise: Corriger ces problèmes avant ML")
        
        # Stats générales
        print(f"\n📊 STATISTIQUES GÉNÉRALES:")
        print(f"   • Dataset: {self.df.shape[0]} matchs, {self.df.shape[1]} colonnes")
        
        if 'Date' in self.df.columns:
            print(f"   • Période: {self.df['Date'].min()} à {self.df['Date'].max()}")
        
        feature_cols = [col for col in self.df.columns if any(x in col for x in ['form', 'elo', 'h2h', 'advantage', 'normalized'])]
        print(f"   • Features ML: {len(feature_cols)} features")
        
        return len(self.issues_found) == 0

# Exécution
if __name__ == "__main__":
    # Tester sur le dataset enhanced (le plus récent)
    validator = OddsyFeatureValidator()
    
    # Essayer différents fichiers selon ce qui existe
    possible_files = [
        '/Users/maxime/Desktop/Oddsy/data/processed/premier_league_ml_ready.csv',
        '/Users/maxime/Desktop/Oddsy/data/processed/premier_league_2019_2024_final.csv',
        '/Users/maxime/Desktop/Oddsy/data/processed/premier_league_2019_2024_enhanced.csv'
    ]
    
    import os
    file_found = None
    for filepath in possible_files:
        if os.path.exists(filepath):
            file_found = filepath
            break
    
    if file_found:
        print(f"🎯 Testing file: {file_found}")
        
        validator.load_data(file_found)
        validator.test_basic_data_quality()
        validator.test_feature_logic()
        validator.test_known_matches()
        validator.test_distribution_sanity()
        validator.test_target_distribution()
        
        is_valid = validator.generate_report()
        
        if is_valid:
            print(f"\n🚀 PRÊT POUR LA PHASE 5 : MACHINE LEARNING!")
        else:
            print(f"\n🔧 CORRIGER LES PROBLÈMES AVANT ML")
            
    else:
        print("❌ Aucun fichier de données trouvé!")
        print("Fichiers cherchés:")
        for f in possible_files:
            print(f"  - {f}")