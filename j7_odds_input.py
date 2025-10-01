"""
Données J7 EPL 2025-26 - Cotes de paris pour prédictions
Journée du 3-5 octobre 2025
"""

import pandas as pd
from datetime import datetime

# Données des matchs J7 avec cotes
j7_matches = [
    {
        'Date': '03/10/2025',
        'Time': '21:00', 
        'HomeTeam': 'Bournemouth',
        'AwayTeam': 'Fulham',
        'B365H': 1.85,
        'B365D': 3.80,
        'B365A': 4.00
    },
    {
        'Date': '04/10/2025',
        'Time': '13:30',
        'HomeTeam': 'Leeds',
        'AwayTeam': 'Tottenham', 
        'B365H': 2.70,
        'B365D': 3.50,
        'B365A': 2.50
    },
    {
        'Date': '04/10/2025',
        'Time': '16:00',
        'HomeTeam': 'Arsenal',
        'AwayTeam': 'West Ham',
        'B365H': 1.22,
        'B365D': 6.00,
        'B365A': 15.00
    },
    {
        'Date': '04/10/2025', 
        'Time': '16:00',
        'HomeTeam': 'Man United',
        'AwayTeam': 'Sunderland',
        'B365H': 1.48,
        'B365D': 4.75,
        'B365A': 6.00
    },
    {
        'Date': '04/10/2025',
        'Time': '18:30',
        'HomeTeam': 'Chelsea',
        'AwayTeam': 'Liverpool',
        'B365H': 2.87,
        'B365D': 3.80,
        'B365A': 2.25
    },
    {
        'Date': '05/10/2025',
        'Time': '15:00',
        'HomeTeam': 'Aston Villa', 
        'AwayTeam': 'Burnley',
        'B365H': 1.85,
        'B365D': 4.20,
        'B365A': 6.00
    },
    {
        'Date': '05/10/2025',
        'Time': '15:00', 
        'HomeTeam': 'Everton',
        'AwayTeam': 'Crystal Palace',
        'B365H': 2.50,
        'B365D': 3.40,
        'B365A': 2.80
    },
    {
        'Date': '05/10/2025',
        'Time': '15:00',
        'HomeTeam': 'Newcastle',
        'AwayTeam': 'Nottm Forest',
        'B365H': 1.61,
        'B365D': 4.20,
        'B365A': 5.00
    },
    {
        'Date': '05/10/2025',
        'Time': '15:00',
        'HomeTeam': 'Wolverhampton',
        'AwayTeam': 'Brighton',
        'B365H': 3.75,
        'B365D': 3.75, 
        'B365A': 1.90
    },
    {
        'Date': '05/10/2025',
        'Time': '17:30',
        'HomeTeam': 'Brentford',
        'AwayTeam': 'Man City',
        'B365H': 4.75,
        'B365D': 4.50,
        'B365A': 1.61
    }
]

def get_j7_dataframe():
    """Retourne un DataFrame avec les données J7"""
    df = pd.DataFrame(j7_matches)
    return df

if __name__ == "__main__":
    df = get_j7_dataframe()
    print("=== MATCHS J7 EPL 2025-26 ===")
    print(f"Nombre de matchs: {len(df)}")
    print()
    for _, match in df.iterrows():
        print(f"{match['Date']} {match['Time']}")
        print(f"{match['HomeTeam']} vs {match['AwayTeam']}")
        print(f"Cotes: {match['B365H']} | {match['B365D']} | {match['B365A']}")
        print("-" * 40)