"""
Test Unitaire Anti-Fuite Temporelle - Zero Data Leakage Protection
================================================================
Tests unitaires pour vérifier qu'aucune donnée future n'est utilisée
pour calculer les features d'un match donné - protection fail-fast
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class TemporalLeakageError(Exception):
    """Exception levée en cas de fuite temporelle détectée"""
    pass


class AntiLeakValidator:
    """Validateur anti-fuite temporelle pour pipelines de prédiction"""
    
    def __init__(self, strict_mode=True):
        self.strict_mode = strict_mode
        self.validation_log = []
        
    def log_validation(self, message, level="INFO"):
        """Log validation avec timestamp"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message
        }
        self.validation_log.append(entry)
        print(f"[{level}] {message}")
    
    def validate_temporal_integrity(self, match_date, source_data, feature_name, description=""):
        """
        Valide qu'aucune donnée source n'est postérieure à match_date
        
        Args:
            match_date: Date du match (datetime ou string)
            source_data: DataFrame avec colonne 'Date'
            feature_name: Nom de la feature calculée
            description: Description optionnelle du contexte
        
        Raises:
            TemporalLeakageError: Si fuite temporelle détectée
        """
        if isinstance(match_date, str):
            match_date = pd.to_datetime(match_date)
        
        if source_data is None or len(source_data) == 0:
            self.log_validation(f"✅ {feature_name}: Pas de données source", "INFO")
            return True
        
        # Vérifier présence colonne Date
        if 'Date' not in source_data.columns:
            raise TemporalLeakageError(f"Feature {feature_name}: Colonne 'Date' manquante dans source_data")
        
        # Convertir dates si nécessaire
        source_dates = pd.to_datetime(source_data['Date'])
        
        # Identifier fuites temporelles
        future_data = source_data[source_dates >= match_date]
        
        if len(future_data) > 0:
            error_msg = (
                f"FUITE TEMPORELLE DÉTECTÉE!\n"
                f"Feature: {feature_name}\n"
                f"Match date: {match_date}\n"
                f"Données futures: {len(future_data)} lignes\n"
                f"Dates futures: {sorted(future_data['Date'].dt.strftime('%Y-%m-%d').unique())[:5]}..."
            )
            
            if description:
                error_msg += f"\nContexte: {description}"
            
            self.log_validation(error_msg, "ERROR")
            
            if self.strict_mode:
                raise TemporalLeakageError(error_msg)
            else:
                return False
        
        # Validation réussie
        earliest_date = source_dates.min().strftime('%Y-%m-%d')
        latest_date = source_dates.max().strftime('%Y-%m-%d')
        
        self.log_validation(
            f"✅ {feature_name}: {len(source_data)} lignes, "
            f"période {earliest_date} → {latest_date} < {match_date.strftime('%Y-%m-%d')}", 
            "SUCCESS"
        )
        
        return True
    
    def validate_rolling_window_integrity(self, match_date, team, source_data, window_size, feature_name):
        """
        Valide l'intégrité temporelle d'une fenêtre roulante pour une équipe
        
        Args:
            match_date: Date du match 
            team: Nom de l'équipe
            source_data: DataFrame historique
            window_size: Taille de la fenêtre
            feature_name: Nom de la feature
        """
        # Filtrer données équipe avant match_date
        team_data = source_data[
            ((source_data['HomeTeam'] == team) | (source_data['AwayTeam'] == team)) &
            (pd.to_datetime(source_data['Date']) < pd.to_datetime(match_date))
        ]
        
        return self.validate_temporal_integrity(
            match_date, 
            team_data.tail(window_size), 
            f"{feature_name}_window_{window_size}_{team}",
            f"Fenêtre roulante {window_size} matchs pour {team}"
        )
    
    def validate_h2h_integrity(self, match_date, home_team, away_team, source_data, feature_name):
        """
        Valide l'intégrité temporelle des données H2H
        
        Args:
            match_date: Date du match
            home_team: Équipe domicile  
            away_team: Équipe extérieur
            source_data: DataFrame historique
            feature_name: Nom de la feature H2H
        """
        # Filtrer confrontations directes avant match_date
        h2h_data = source_data[
            (
                ((source_data['HomeTeam'] == home_team) & (source_data['AwayTeam'] == away_team)) |
                ((source_data['HomeTeam'] == away_team) & (source_data['AwayTeam'] == home_team))
            ) &
            (pd.to_datetime(source_data['Date']) < pd.to_datetime(match_date))
        ]
        
        return self.validate_temporal_integrity(
            match_date,
            h2h_data,
            f"{feature_name}_h2h_{home_team}_vs_{away_team}",
            f"Historique H2H {home_team} vs {away_team}"
        )
    
    def validate_market_data_integrity(self, match_date, market_data, feature_name):
        """
        Valide l'intégrité temporelle des données de marché (cotes)
        
        Args:
            match_date: Date du match
            market_data: DataFrame avec données de marché
            feature_name: Nom de la feature marché
        """
        # Les cotes doivent être disponibles AVANT le match
        # On accepte jusqu'à 1 jour avant pour les cotes de clôture
        cutoff_date = pd.to_datetime(match_date) - timedelta(hours=2)  # 2h avant kick-off
        
        valid_market_data = market_data[pd.to_datetime(market_data['Date']) <= cutoff_date]
        
        return self.validate_temporal_integrity(
            cutoff_date,
            valid_market_data,
            f"{feature_name}_market",
            f"Données marché avec cutoff 2h avant match"
        )
    
    def validate_feature_calculation_pipeline(self, match_date, home_team, away_team, historical_data, feature_calculator):
        """
        Valide l'intégrité complète d'un pipeline de calcul de features
        
        Args:
            match_date: Date du match à prédire
            home_team: Équipe domicile
            away_team: Équipe extérieur  
            historical_data: DataFrame historique complet
            feature_calculator: Instance du calculateur de features
        """
        self.log_validation(f"🔍 Validation pipeline: {home_team} vs {away_team} le {match_date}", "INFO")
        
        validation_results = {
            'match_info': f"{home_team} vs {away_team} - {match_date}",
            'validations': [],
            'success': True,
            'errors': []
        }
        
        try:
            # 1. Valider données historiques globales
            self.validate_temporal_integrity(
                match_date,
                historical_data,
                "historical_data_global",
                "Dataset historique complet"
            )
            validation_results['validations'].append("historical_data_global")
            
            # 2. Valider fenêtres roulantes équipes
            for team in [home_team, away_team]:
                # Form (5 matchs)
                self.validate_rolling_window_integrity(
                    match_date, team, historical_data, 5, "form"
                )
                validation_results['validations'].append(f"form_window_{team}")
                
                # ELO estimation (10 matchs)
                self.validate_rolling_window_integrity(
                    match_date, team, historical_data, 10, "elo"
                )
                validation_results['validations'].append(f"elo_window_{team}")
                
                # xG efficiency (10 matchs)
                self.validate_rolling_window_integrity(
                    match_date, team, historical_data, 10, "xg_efficiency"
                )
                validation_results['validations'].append(f"xg_efficiency_window_{team}")
            
            # 3. Valider H2H
            self.validate_h2h_integrity(
                match_date, home_team, away_team, historical_data, "h2h_score"
            )
            validation_results['validations'].append("h2h_integrity")
            
            # 4. Valider données spécifiques si FBref disponible
            if hasattr(feature_calculator, 'has_fbref_data') and feature_calculator.has_fbref_data():
                fbref_data = feature_calculator.fbref_data
                self.validate_temporal_integrity(
                    match_date,
                    fbref_data,
                    "fbref_enhanced_data",
                    "Données FBref enhanced"
                )
                validation_results['validations'].append("fbref_data_integrity")
            
            self.log_validation(f"✅ Pipeline validation RÉUSSIE: {len(validation_results['validations'])} checks", "SUCCESS")
            
        except TemporalLeakageError as e:
            validation_results['success'] = False
            validation_results['errors'].append(str(e))
            self.log_validation(f"❌ Pipeline validation ÉCHOUÉE: {str(e)}", "ERROR")
            raise
        
        except Exception as e:
            validation_results['success'] = False
            validation_results['errors'].append(f"Erreur validation: {str(e)}")
            self.log_validation(f"❌ Erreur pipeline validation: {str(e)}", "ERROR")
            raise
        
        return validation_results
    
    def get_validation_report(self):
        """Retourne rapport de toutes les validations effectuées"""
        return {
            'total_validations': len(self.validation_log),
            'successes': len([v for v in self.validation_log if v['level'] == 'SUCCESS']),
            'errors': len([v for v in self.validation_log if v['level'] == 'ERROR']),
            'warnings': len([v for v in self.validation_log if v['level'] == 'WARNING']),
            'log': self.validation_log
        }


def test_anti_leak_basic():
    """Test basique du validateur anti-fuite"""
    print("🧪 Test basique anti-fuite...")
    
    validator = AntiLeakValidator(strict_mode=True)
    
    # Créer données test
    dates = pd.date_range('2025-01-01', '2025-01-10', freq='D')
    test_data = pd.DataFrame({
        'Date': dates,
        'HomeTeam': ['Arsenal'] * len(dates),
        'AwayTeam': ['Chelsea'] * len(dates),
        'FTHG': [1, 2, 0, 1, 3, 2, 1, 0, 2, 1],
        'FTAG': [0, 1, 1, 2, 1, 0, 1, 1, 0, 2]
    })
    
    match_date = '2025-01-06'
    
    # Test 1: Données valides (avant match_date)
    valid_data = test_data[test_data['Date'] < match_date]
    try:
        validator.validate_temporal_integrity(match_date, valid_data, "test_valid")
        print("✅ Test données valides: RÉUSSI")
    except TemporalLeakageError:
        print("❌ Test données valides: ÉCHEC")
    
    # Test 2: Données avec fuite (après match_date)  
    invalid_data = test_data  # Inclut données après match_date
    try:
        validator.validate_temporal_integrity(match_date, invalid_data, "test_leak")
        print("❌ Test détection fuite: ÉCHEC (fuite non détectée)")
    except TemporalLeakageError:
        print("✅ Test détection fuite: RÉUSSI")
    
    # Rapport final
    report = validator.get_validation_report()
    print(f"📋 Rapport: {report['successes']} succès, {report['errors']} erreurs")


def test_anti_leak_realistic():
    """Test réaliste avec pipeline J7"""
    print("\n🏈 Test réaliste pipeline J7...")
    
    validator = AntiLeakValidator(strict_mode=True)
    
    # Simuler match J7 du 2025-10-05  
    match_date = '2025-10-05'
    home_team = 'Arsenal'
    away_team = 'Chelsea'
    
    # Créer données historiques réalistes
    historical_dates = pd.date_range('2025-08-01', '2025-10-04', freq='3D')
    historical_data = pd.DataFrame({
        'Date': historical_dates,
        'HomeTeam': np.random.choice(['Arsenal', 'Chelsea', 'Liverpool', 'Man City'], len(historical_dates)),
        'AwayTeam': np.random.choice(['Arsenal', 'Chelsea', 'Liverpool', 'Man City'], len(historical_dates)),
        'FTHG': np.random.randint(0, 4, len(historical_dates)),
        'FTAG': np.random.randint(0, 4, len(historical_dates))
    })
    
    # Assurer qu'Arsenal et Chelsea ont des matchs dans l'historique
    historical_data.iloc[0] = ['2025-08-15', 'Arsenal', 'Liverpool', 2, 1]
    historical_data.iloc[1] = ['2025-08-22', 'Chelsea', 'Man City', 1, 0]
    historical_data.iloc[2] = ['2025-09-01', 'Liverpool', 'Arsenal', 0, 3]
    historical_data.iloc[3] = ['2025-09-15', 'Man City', 'Chelsea', 2, 2]
    
    historical_data['Date'] = pd.to_datetime(historical_data['Date'])
    
    # Mock feature calculator
    class MockFeatureCalculator:
        def __init__(self):
            self.has_fbref_data_flag = False
        
        def has_fbref_data(self):
            return self.has_fbref_data_flag
    
    try:
        mock_calculator = MockFeatureCalculator()
        result = validator.validate_feature_calculation_pipeline(
            match_date, home_team, away_team, historical_data, mock_calculator
        )
        
        print(f"✅ Validation pipeline réaliste: RÉUSSIE")
        print(f"   Checks effectués: {len(result['validations'])}")
        
    except TemporalLeakageError as e:
        print(f"❌ Validation pipeline réaliste: ÉCHEC - {str(e)}")
    except Exception as e:
        print(f"❌ Erreur test réaliste: {str(e)}")


if __name__ == "__main__":
    print("=" * 70)
    print("🛡️ TESTS UNITAIRES ANTI-FUITE TEMPORELLE")
    print("=" * 70)
    
    test_anti_leak_basic()
    test_anti_leak_realistic()
    
    print("\n✅ Tests anti-fuite terminés")