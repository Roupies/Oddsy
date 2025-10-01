"""
FBref Data Fusion - Intégration avec Football-Data
==================================================
Fusionne données FBref (xG, tirs, corners) avec Football-Data E0
pour créer dataset complet avec vraies features
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import re

class FBrefDataFusion:
    """Gestionnaire fusion FBref + Football-Data"""
    
    def __init__(self):
        self.team_mapping = self._create_team_mapping()
        self.fusion_stats = {
            'fbref_matches': 0,
            'football_data_matches': 0,
            'successful_merges': 0,
            'failed_merges': 0,
            'missing_teams': set(),
            'date_mismatches': 0
        }
    
    def _create_team_mapping(self):
        """Mapping noms équipes FBref ↔ Football-Data"""
        return {
            # FBref → Football-Data
            'Arsenal': 'Arsenal',
            'Aston Villa': 'Aston Villa', 
            'Bournemouth': 'Bournemouth',
            'Brentford': 'Brentford',
            'Brighton & Hove Albion': 'Brighton',
            'Brighton': 'Brighton',
            'Burnley': 'Burnley',
            'Chelsea': 'Chelsea',
            'Crystal Palace': 'Crystal Palace',
            'Everton': 'Everton',
            'Fulham': 'Fulham',
            'Ipswich Town': 'Ipswich',
            'Leeds United': 'Leeds',
            'Leicester City': 'Leicester',
            'Liverpool': 'Liverpool',
            'Manchester City': 'Man City',
            'Manchester United': 'Man United',
            'Newcastle United': 'Newcastle',
            'Nottingham Forest': 'Nottm Forest',
            'Sheffield United': 'Sheffield United',
            'Southampton': 'Southampton',
            'Sunderland': 'Sunderland',
            'Tottenham Hotspur': 'Tottenham',
            'West Ham United': 'West Ham',
            'Wolverhampton Wanderers': 'Wolverhampton',
            'Wolves': 'Wolverhampton'
        }
    
    def normalize_team_name(self, team_name):
        """Normalise nom équipe vers format Football-Data"""
        if pd.isna(team_name):
            return None
            
        team_name = str(team_name).strip()
        
        # Mapping direct
        if team_name in self.team_mapping:
            return self.team_mapping[team_name]
        
        # Tentatives de mapping partiel
        for fbref_name, fd_name in self.team_mapping.items():
            if team_name.lower() in fbref_name.lower() or fbref_name.lower() in team_name.lower():
                return fd_name
        
        # Aucun mapping trouvé
        self.fusion_stats['missing_teams'].add(team_name)
        print(f"⚠️ Équipe non mappée: {team_name}")
        return team_name
    
    def load_football_data(self, filepath):
        """Charge données Football-Data E0"""
        print(f"📊 Chargement Football-Data: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            
            # Normaliser format date
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
            
            # Filtrer saison 2025-26
            if 'Date' in df.columns:
                df = df[df['Date'] >= '2025-08-01'].copy()
            
            self.fusion_stats['football_data_matches'] = len(df)
            print(f"   ✅ {len(df)} matchs Football-Data chargés")
            
            return df
            
        except Exception as e:
            print(f"❌ Erreur chargement Football-Data: {e}")
            return None
    
    def load_fbref_data(self, filepath):
        """Charge données FBref team logs"""
        print(f"📊 Chargement FBref: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            
            # Normaliser format date si présent
            date_columns = [col for col in df.columns if 'date' in col.lower()]
            if date_columns:
                date_col = date_columns[0]
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Normaliser noms équipes
            if 'Squad' in df.columns:
                df['Squad_normalized'] = df['Squad'].apply(self.normalize_team_name)
            if 'Opponent' in df.columns:
                df['Opponent_normalized'] = df['Opponent'].apply(self.normalize_team_name)
            
            self.fusion_stats['fbref_matches'] = len(df)
            print(f"   ✅ {len(df)} lignes FBref chargées")
            
            # Afficher colonnes disponibles
            print("   📋 Colonnes FBref disponibles:")
            for col in sorted(df.columns):
                print(f"      - {col}")
            
            return df
            
        except Exception as e:
            print(f"❌ Erreur chargement FBref: {e}")
            return None
    
    def extract_match_stats(self, fbref_df):
        """Extrait stats par match depuis FBref team logs"""
        print("🔧 Extraction stats par match...")
        
        if fbref_df is None or len(fbref_df) == 0:
            return None
        
        # Colonnes d'intérêt pour xG, tirs, corners
        stats_columns = {
            'xG': ['xG', 'xg', 'Expected Goals', 'xG_for'],
            'xGA': ['xGA', 'xga', 'xG Against', 'xG_against'],
            'Shots': ['Sh', 'shots', 'Shots', 'shots_total'],
            'ShotsOnTarget': ['SoT', 'shots_on_target', 'Shots on Target'],
            'Corners': ['Corner', 'corners', 'Corners']
        }
        
        # Identifier colonnes présentes
        available_stats = {}
        for stat_name, possible_cols in stats_columns.items():
            for col in possible_cols:
                if col in fbref_df.columns:
                    available_stats[stat_name] = col
                    break
        
        print(f"   📊 Stats trouvées: {list(available_stats.keys())}")
        
        # Extraire données par match
        match_stats = []
        
        for _, row in fbref_df.iterrows():
            try:
                squad = row.get('Squad_normalized', row.get('Squad'))
                opponent = row.get('Opponent_normalized', row.get('Opponent'))
                venue = row.get('Venue', 'Unknown')
                date = row.get('Date', None)
                
                if pd.isna(squad) or pd.isna(opponent):
                    continue
                
                # Déterminer home/away
                is_home = venue == 'Home'
                home_team = squad if is_home else opponent
                away_team = opponent if is_home else squad
                
                # Extraire stats
                stats = {
                    'Date': date,
                    'HomeTeam': home_team,
                    'AwayTeam': away_team,
                    'Venue': venue,
                    'Squad': squad
                }
                
                # Ajouter stats disponibles
                for stat_name, col_name in available_stats.items():
                    value = row.get(col_name)
                    if pd.notna(value):
                        # Préfixer par position (Home/Away)
                        prefix = 'H' if is_home else 'A'
                        stats[f'{prefix}_{stat_name}'] = value
                
                match_stats.append(stats)
                
            except Exception as e:
                print(f"⚠️ Erreur extraction ligne: {e}")
                continue
        
        if match_stats:
            stats_df = pd.DataFrame(match_stats)
            print(f"   ✅ {len(stats_df)} lignes stats extraites")
            return stats_df
        else:
            print("   ❌ Aucune stat extraite")
            return None
    
    def merge_with_football_data(self, football_data_df, fbref_stats_df):
        """Fusionne FBref stats avec Football-Data"""
        print("🔗 Fusion FBref + Football-Data...")
        
        if football_data_df is None or fbref_stats_df is None:
            print("❌ Données manquantes pour fusion")
            return None
        
        # Grouper stats FBref par match
        match_grouped = fbref_stats_df.groupby(['Date', 'HomeTeam', 'AwayTeam']).agg({
            col: 'first' for col in fbref_stats_df.columns 
            if col not in ['Date', 'HomeTeam', 'AwayTeam']
        }).reset_index()
        
        print(f"   📊 {len(match_grouped)} matchs FBref groupés")
        
        # Merger avec Football-Data
        merged_df = football_data_df.merge(
            match_grouped,
            on=['Date', 'HomeTeam', 'AwayTeam'],
            how='left',
            indicator=True
        )
        
        # Statistiques fusion
        merge_stats = merged_df['_merge'].value_counts()
        self.fusion_stats['successful_merges'] = merge_stats.get('both', 0)
        self.fusion_stats['failed_merges'] = merge_stats.get('left_only', 0)
        
        print(f"   ✅ Fusion réussie: {self.fusion_stats['successful_merges']} matchs")
        print(f"   ⚠️ Échecs fusion: {self.fusion_stats['failed_merges']} matchs")
        
        # Supprimer colonne indicateur
        merged_df = merged_df.drop('_merge', axis=1)
        
        return merged_df
    
    def export_merged_data(self, merged_df, output_path):
        """Exporte données fusionnées"""
        if merged_df is None:
            print("❌ Pas de données à exporter")
            return None
        
        try:
            # Créer répertoire si nécessaire
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Export CSV
            merged_df.to_csv(output_path, index=False)
            print(f"💾 Données fusionnées exportées: {output_path}")
            
            # Export métadonnées
            metadata_path = output_path.replace('.csv', '_metadata.json')
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'total_matches': len(merged_df),
                'fusion_stats': dict(self.fusion_stats),
                'columns': list(merged_df.columns),
                'date_range': {
                    'start': merged_df['Date'].min().isoformat() if 'Date' in merged_df.columns else None,
                    'end': merged_df['Date'].max().isoformat() if 'Date' in merged_df.columns else None
                }
            }
            
            # Convertir sets en listes pour JSON
            if 'missing_teams' in metadata['fusion_stats']:
                metadata['fusion_stats']['missing_teams'] = list(metadata['fusion_stats']['missing_teams'])
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"💾 Métadonnées exportées: {metadata_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Erreur export: {e}")
            return None
    
    def process_fusion(self, football_data_path, fbref_path, output_path):
        """Pipeline complète de fusion"""
        print("=" * 60)
        print("🔄 PIPELINE FUSION FBREF + FOOTBALL-DATA")
        print("=" * 60)
        
        # 1. Charger données
        football_data_df = self.load_football_data(football_data_path)
        fbref_df = self.load_fbref_data(fbref_path)
        
        # 2. Extraire stats par match
        fbref_stats_df = self.extract_match_stats(fbref_df)
        
        # 3. Fusionner
        merged_df = self.merge_with_football_data(football_data_df, fbref_stats_df)
        
        # 4. Exporter
        if merged_df is not None:
            result_path = self.export_merged_data(merged_df, output_path)
            
            # Résumé final
            print("\n" + "=" * 60)
            print("📋 RÉSUMÉ FUSION")
            print("=" * 60)
            print(f"Football-Data: {self.fusion_stats['football_data_matches']} matchs")
            print(f"FBref: {self.fusion_stats['fbref_matches']} lignes")
            print(f"Fusion réussie: {self.fusion_stats['successful_merges']} matchs")
            print(f"Fusion échouée: {self.fusion_stats['failed_merges']} matchs")
            
            if self.fusion_stats['missing_teams']:
                print(f"Équipes non mappées: {', '.join(self.fusion_stats['missing_teams'])}")
            
            return result_path
        
        return None

def main():
    """Test de la fusion"""
    fusion = FBrefDataFusion()
    
    # Chemins de test
    football_data_path = "data/raw/E0 (14).csv"  # Dernier fichier Football-Data
    fbref_path = "data/fbref/epl_2025_26_team_logs_latest.csv"  # À remplacer
    output_path = "data/processed/epl_2025_26_merged_fbref.csv"
    
    # Exécuter fusion
    result = fusion.process_fusion(football_data_path, fbref_path, output_path)
    
    if result:
        print(f"\n✅ Fusion terminée: {result}")
    else:
        print("\n❌ Fusion échouée")

if __name__ == "__main__":
    main()