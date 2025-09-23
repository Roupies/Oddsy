#!/usr/bin/env python3
"""
🔍 CSV Structure Validator
=========================

Analyse et valide la structure du CSV E0 pour compatibilité.
Détecte colonnes, types, contraintes et prépare mapping PostgreSQL.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from collections import Counter

class CSVStructureAnalyzer:
    """Analyseur de structure CSV pour validation"""
    
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.df = None
        self.analysis_report = {}
    
    def load_and_analyze(self):
        """Charge et analyse le CSV complet"""
        logging.info(f"📊 Analyse structure: {self.csv_path}")
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV non trouvé: {self.csv_path}")
        
        # Charger avec détection encoding
        try:
            self.df = pd.read_csv(self.csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            self.df = pd.read_csv(self.csv_path, encoding='utf-8-sig')
        
        logging.info(f"✅ {len(self.df)} lignes, {len(self.df.columns)} colonnes chargées")
        
        # Analyses détaillées
        self._analyze_basic_structure()
        self._analyze_column_types()
        self._analyze_missing_values()
        self._analyze_betting_odds()
        self._analyze_match_data()
        self._analyze_postgresql_mapping()
        
        return self.analysis_report
    
    def _analyze_basic_structure(self):
        """Analyse structure de base"""
        basic = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'columns_list': self.df.columns.tolist(),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024 / 1024,
            'file_size_mb': self.csv_path.stat().st_size / 1024 / 1024
        }
        
        self.analysis_report['basic_structure'] = basic
        logging.info(f"📏 Structure: {basic['total_rows']}x{basic['total_columns']}")
    
    def _analyze_column_types(self):
        """Analyse types de colonnes"""
        type_analysis = {}
        
        # Détecter types pandas
        dtypes = self.df.dtypes.value_counts()
        type_analysis['pandas_types'] = dtypes.to_dict()
        
        # Analyser colonnes par catégorie
        categories = self._categorize_columns()
        type_analysis['categories'] = categories
        
        # Types inférés pour PostgreSQL
        pg_types = self._infer_postgresql_types()
        type_analysis['postgresql_types'] = pg_types
        
        self.analysis_report['column_types'] = type_analysis
        logging.info(f"📊 Types détectés: {len(categories)} catégories")
    
    def _categorize_columns(self):
        """Catégorise les colonnes par fonction"""
        categories = {
            'match_info': ['Div', 'Date', 'Time', 'HomeTeam', 'AwayTeam', 'Referee'],
            'results': ['FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR'],
            'statistics': ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR'],
            'betting_odds': [],
            'over_under': [],
            'asian_handicap': [],
            'corner_betting': [],
            'unknown': []
        }
        
        # Classifier colonnes odds
        for col in self.df.columns:
            if col.endswith(('H', 'D', 'A')) and any(bm in col for bm in ['B365', 'BFD', 'BW', 'PS']):
                if 'C' not in col:  # Pas corner betting
                    categories['betting_odds'].append(col)
            elif '>2.5' in col or '<2.5' in col:
                if 'C' in col:
                    categories['corner_betting'].append(col)
                else:
                    categories['over_under'].append(col)
            elif 'AH' in col:
                if 'C' in col:
                    categories['corner_betting'].append(col)
                else:
                    categories['asian_handicap'].append(col)
            elif col not in sum(categories.values(), []):
                categories['unknown'].append(col)
        
        return {k: v for k, v in categories.items() if v}  # Retirer catégories vides
    
    def _infer_postgresql_types(self):
        """Infère types PostgreSQL appropriés"""
        pg_types = {}
        
        for col in self.df.columns:
            # Types spéciaux
            if col == 'Date':
                pg_types[col] = 'DATE'
            elif col == 'Time':
                pg_types[col] = 'TIME'
            elif col in ['HomeTeam', 'AwayTeam', 'Referee', 'Div']:
                pg_types[col] = 'VARCHAR(100)'
            elif col in ['FTR', 'HTR']:
                pg_types[col] = 'CHAR(1)'
            
            # Types basés sur données
            elif self.df[col].dtype == 'object':
                max_len = self.df[col].astype(str).str.len().max()
                pg_types[col] = f'VARCHAR({min(max_len + 10, 255)})'
            
            elif self.df[col].dtype in ['int64', 'int32']:
                if self.df[col].max() < 32767:
                    pg_types[col] = 'SMALLINT'
                else:
                    pg_types[col] = 'INTEGER'
            
            elif self.df[col].dtype in ['float64', 'float32']:
                # Odds et stats décimales
                if col.endswith(('H', 'D', 'A')) or any(x in col for x in ['Avg', 'Max']):
                    pg_types[col] = 'DECIMAL(6,3)'
                else:
                    pg_types[col] = 'DECIMAL(5,2)'
            
            else:
                pg_types[col] = 'TEXT'
        
        return pg_types
    
    def _analyze_missing_values(self):
        """Analyse valeurs manquantes"""
        missing = {}
        
        total_cells = len(self.df) * len(self.df.columns)
        missing_counts = self.df.isnull().sum()
        
        missing['total_missing'] = missing_counts.sum()
        missing['missing_percentage'] = (missing_counts.sum() / total_cells) * 100
        missing['columns_with_missing'] = missing_counts[missing_counts > 0].to_dict()
        missing['complete_columns'] = missing_counts[missing_counts == 0].index.tolist()
        
        self.analysis_report['missing_values'] = missing
        logging.info(f"🕳️ Manquantes: {missing['missing_percentage']:.1f}%")
    
    def _analyze_betting_odds(self):
        """Analyse spécifique des odds"""
        betting_analysis = {}
        
        # Colonnes odds principales
        main_odds_cols = ['B365H', 'B365D', 'B365A']
        
        if all(col in self.df.columns for col in main_odds_cols):
            # Statistiques odds
            for col in main_odds_cols:
                if col in self.df.columns:
                    stats = {
                        'min': float(self.df[col].min()),
                        'max': float(self.df[col].max()),
                        'mean': float(self.df[col].mean()),
                        'missing_count': int(self.df[col].isnull().sum())
                    }
                    betting_analysis[col] = stats
            
            # Validation cohérence odds
            coherence = self._validate_odds_coherence()
            betting_analysis['coherence_check'] = coherence
        
        # Comptage bookmakers
        bookmakers = self._count_bookmakers()
        betting_analysis['bookmakers'] = bookmakers
        
        self.analysis_report['betting_odds'] = betting_analysis
    
    def _validate_odds_coherence(self):
        """Valide cohérence des odds (somme probabilités ≈ 1)"""
        coherence = {'valid_rows': 0, 'invalid_rows': 0, 'examples': []}
        
        main_odds = ['B365H', 'B365D', 'B365A']
        
        for idx, row in self.df.iterrows():
            if all(pd.notna(row[col]) for col in main_odds):
                # Calculer probabilités implicites
                prob_sum = sum(1/row[col] for col in main_odds)
                
                # Tolérance pour marge bookmaker (≈ 1.05-1.10)
                if 1.02 <= prob_sum <= 1.15:
                    coherence['valid_rows'] += 1
                else:
                    coherence['invalid_rows'] += 1
                    if len(coherence['examples']) < 3:
                        coherence['examples'].append({
                            'row': idx,
                            'odds': [row[col] for col in main_odds],
                            'prob_sum': prob_sum
                        })
        
        coherence['validity_rate'] = coherence['valid_rows'] / (coherence['valid_rows'] + coherence['invalid_rows'])
        return coherence
    
    def _count_bookmakers(self):
        """Compte bookmakers disponibles"""
        bookmaker_prefixes = ['B365', 'BFD', 'BW', 'PS', 'BV', 'CL', 'LB', 'BMGM']
        bookmaker_counts = {}
        
        for prefix in bookmaker_prefixes:
            h_col = f'{prefix}H'
            if h_col in self.df.columns:
                coverage = (~self.df[h_col].isnull()).sum()
                bookmaker_counts[prefix] = {
                    'coverage_count': int(coverage),
                    'coverage_percentage': float(coverage / len(self.df) * 100)
                }
        
        return bookmaker_counts
    
    def _analyze_match_data(self):
        """Analyse données de matchs"""
        match_analysis = {}
        
        # Distribution résultats
        if 'FTR' in self.df.columns:
            result_dist = self.df['FTR'].value_counts().to_dict()
            match_analysis['result_distribution'] = result_dist
        
        # Équipes les plus fréquentes
        if 'HomeTeam' in self.df.columns:
            team_counts = Counter(list(self.df['HomeTeam']) + list(self.df['AwayTeam']))
            match_analysis['team_frequency'] = dict(team_counts.most_common(10))
        
        # Plage de dates
        if 'Date' in self.df.columns:
            try:
                dates = pd.to_datetime(self.df['Date'], format='%d/%m/%Y')
                match_analysis['date_range'] = {
                    'earliest': dates.min().strftime('%Y-%m-%d'),
                    'latest': dates.max().strftime('%Y-%m-%d'),
                    'span_days': (dates.max() - dates.min()).days
                }
            except:
                match_analysis['date_range'] = 'Could not parse dates'
        
        self.analysis_report['match_data'] = match_analysis
    
    def _analyze_postgresql_mapping(self):
        """Analyse mapping vers PostgreSQL"""
        pg_mapping = {}
        
        # Colonnes pour table matches
        matches_columns = {
            'match_date': 'Date',
            'home_team': 'HomeTeam', 
            'away_team': 'AwayTeam',
            'full_time_result': 'FTR',
            'home_goals': 'FTHG',
            'away_goals': 'FTAG',
            'home_odds': 'B365H',
            'draw_odds': 'B365D',
            'away_odds': 'B365A',
            'home_shots': 'HS',
            'away_shots': 'AS',
            'home_corners': 'HC',
            'away_corners': 'AC'
        }
        
        pg_mapping['matches_table_mapping'] = matches_columns
        
        # Vérifier colonnes disponibles
        available = {pg_col: csv_col for pg_col, csv_col in matches_columns.items() if csv_col in self.df.columns}
        missing = {pg_col: csv_col for pg_col, csv_col in matches_columns.items() if csv_col not in self.df.columns}
        
        pg_mapping['available_mappings'] = available
        pg_mapping['missing_mappings'] = missing
        
        # Recommandations
        recommendations = []
        if missing:
            recommendations.append(f"Colonnes manquantes: {list(missing.keys())}")
        if len(available) >= len(matches_columns) * 0.8:
            recommendations.append("Structure compatible PostgreSQL (>80% colonnes)")
        
        pg_mapping['recommendations'] = recommendations
        
        self.analysis_report['postgresql_mapping'] = pg_mapping
    
    def generate_report(self):
        """Génère rapport complet"""
        if not self.analysis_report:
            self.load_and_analyze()
        
        print("=" * 60)
        print("📊 RAPPORT D'ANALYSE CSV E0")
        print("=" * 60)
        
        # Structure de base
        basic = self.analysis_report['basic_structure']
        print(f"\\n📏 STRUCTURE:")
        print(f"   Lignes: {basic['total_rows']:,}")
        print(f"   Colonnes: {basic['total_columns']}")
        print(f"   Taille: {basic['file_size_mb']:.1f} MB")
        
        # Valeurs manquantes
        missing = self.analysis_report['missing_values']
        print(f"\\n🕳️ DONNÉES MANQUANTES:")
        print(f"   Total: {missing['missing_percentage']:.1f}%")
        print(f"   Colonnes complètes: {len(missing['complete_columns'])}")
        
        # Odds
        if 'betting_odds' in self.analysis_report:
            betting = self.analysis_report['betting_odds']
            if 'coherence_check' in betting:
                coh = betting['coherence_check']
                print(f"\\n🎰 ODDS:")
                print(f"   Cohérence: {coh['validity_rate']:.1%}")
                print(f"   Bookmakers: {len(betting['bookmakers'])}")
        
        # PostgreSQL
        if 'postgresql_mapping' in self.analysis_report:
            pg = self.analysis_report['postgresql_mapping']
            avail = len(pg['available_mappings'])
            total = len(pg['matches_table_mapping'])
            print(f"\\n🗄️ POSTGRESQL:")
            print(f"   Compatibilité: {avail}/{total} ({avail/total:.1%})")
        
        # Matchs
        if 'match_data' in self.analysis_report:
            match = self.analysis_report['match_data']
            if 'result_distribution' in match:
                print(f"\\n⚽ RÉSULTATS:")
                for result, count in match['result_distribution'].items():
                    print(f"   {result}: {count} ({count/basic['total_rows']:.1%})")
        
        print("\\n" + "=" * 60)
        
        return self.analysis_report

# =============================================================
# SCRIPT PRINCIPAL  
# =============================================================

def main():
    """Analyse complète CSV E0"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    csv_path = "../../data/raw/E0 (9).csv"
    
    try:
        analyzer = CSVStructureAnalyzer(csv_path)
        report = analyzer.generate_report()
        
        # Sauvegarder rapport
        import json
        report_path = Path("csv_structure_report.json")
        with open(report_path, 'w') as f:
            # Convertir numpy types pour JSON
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(report, f, indent=2, default=convert_numpy)
        
        logging.info(f"📋 Rapport sauvé: {report_path}")
        
    except Exception as e:
        logging.error(f"❌ Erreur analyse: {e}")
        raise

if __name__ == "__main__":
    main()