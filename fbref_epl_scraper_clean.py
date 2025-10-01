#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests, time, re, os, sys
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime

BASE = "https://fbref.com"
FIXTURES_URL = "https://fbref.com/en/comps/9/2025-2026/schedule/2025-2026-Premier-League-Fixtures"

class FBrefEPLScraper:
    def __init__(self, pause=3, retries=3, timeout=20):
        self.s = requests.Session()
        self.s.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119 Safari/537.36"
        })
        self.pause = pause
        self.retries = retries
        self.timeout = timeout

    def get(self, url):
        for i in range(self.retries):
            try:
                r = self.s.get(url, timeout=self.timeout)
                if r.status_code == 200:
                    return r.text
                if r.status_code in (403, 429):
                    time.sleep(self.pause * (i + 1))
                else:
                    time.sleep(self.pause)
            except Exception:
                time.sleep(self.pause * (i + 1))
        return None

    def parse_fixtures_links_j1_j6(self, html):
        """
        Retourne une liste de dicts: {date, home, away, match_url, round}
        Limite aux 6 premières journées en détectant 'Wk' (Week) / 'Round'.
        """
        soup = BeautifulSoup(html, "html.parser")
        # Chercher toute table dont l'id commence par 'sched_2025-2026'
        tables = soup.find_all("table", id=re.compile(r"^sched_2025-2026"))
        if not tables:
            print("❌ Table fixtures introuvable")
            return []

        rows = tables[0].find("tbody").find_all("tr")
        out = []
        for tr in rows:
            if "class" in tr.attrs and "spacer" in tr["class"]:
                continue
            tds = tr.find_all("td")
            ths = tr.find_all("th")
            # Certaines colonnes utiles
            round_cell = tr.find("th", {"data-stat": "round"})
            date_cell = tr.find("td", {"data-stat": "date"})
            home_cell = tr.find("td", {"data-stat": "home_team"})
            away_cell = tr.find("td", {"data-stat": "away_team"})
            score_cell = tr.find("td", {"data-stat": "score"})

            if not (round_cell and date_cell and home_cell and away_cell):
                continue

            round_txt = round_cell.get_text(strip=True)
            # Tenter d'extraire le numéro de journée (Wk 1, Week 2, etc.)
            m = re.search(r"(\d+)", round_txt)
            wk = int(m.group(1)) if m else None
            if wk is None or wk > 6:
                continue

            date_txt = date_cell.get_text(strip=True)
            home = home_cell.get_text(strip=True)
            away = away_cell.get_text(strip=True)

            match_href = None
            if score_cell:
                a = score_cell.find("a")
                if a and a.get("href"):
                    match_href = urljoin(BASE, a["href"])

            if home and away and match_href:
                out.append({
                    "round": wk,
                    "date_raw": date_txt,
                    "home": home,
                    "away": away,
                    "match_url": match_href
                })
        print(f"✅ Fixtures J1→J6 trouvées: {len(out)}")
        return out

    def parse_match_stats(self, html):
        """
        Extrait Shots, Shots on Target, Corners, xG pour Home/Away
        Retourne dict: {H_Shots, A_Shots, H_SoT, A_SoT, H_Corner, A_Corner, H_xG, A_xG}
        """
        soup = BeautifulSoup(html, "html.parser")

        # 1) xG près du score (scorebox)
        H_xG, A_xG = None, None
        scorebox = soup.find("div", id="content")
        # FBref varie le rendu; tenter un pattern générique "xG" à proximité du score
        # Fallback: chercher dans les commentaires HTML (FBref encapsule parfois des tables)
        def try_find_xg():
            nonlocal H_xG, A_xG
            # Chercher un tableau "Expected" dans les commentaires
            comments = soup.find_all(string=lambda text: isinstance(text, type(soup.original_encoding)) or False)
            # Heuristique: BeautifulSoup ne renvoie pas facilement les commentaires en mode str; on fait simple:
            # Alternative: chercher "xG" dans tout le texte
            txt = soup.get_text(" ").lower()
            m = re.findall(r"xg[:\s]+(\d+(?:\.\d+)?)", txt)
            if len(m) >= 2:
                try:
                    H_xG = float(m[0]); A_xG = float(m[1])
                except:
                    pass

        try_find_xg()

        # 2) Team Stats (Shots, Shots on Target, Corners)
        # Chercher une table "Team Stats" avec data-stat reconnaissables
        H_Shots = A_Shots = H_SoT = A_SoT = H_Corner = A_Corner = None

        # Heuristique: tables avec "team_stats" dans l'id ou caption "Team Stats"
        team_tables = soup.find_all("table", id=re.compile(r"team_stats|stats"))
        for table in team_tables:
            # Essayer de lire par lignes label / home / away
            # On parcourt chaque ligne pour identifier le label (Shots, Shots on Target, Corners)
            for tr in table.find_all("tr"):
                cells = [c.get_text(strip=True) for c in tr.find_all(["th","td"])]
                if len(cells) < 3:
                    continue
                label = cells[0].lower()
                try:
                    home_val = float(cells[1]) if cells[1] not in ("", None) else None
                    away_val = float(cells[-1]) if cells[-1] not in ("", None) else None
                except:
                    continue
                if "shots on target" in label or "sot" in label or "shots on targ" in label:
                    H_SoT = H_SoT or home_val
                    A_SoT = A_SoT or away_val
                elif label == "shots" or "total shots" in label:
                    H_Shots = H_Shots or home_val
                    A_Shots = A_Shots or away_val
                elif "corners" in label:
                    H_Corner = H_Corner or home_val
                    A_Corner = A_Corner or away_val

        return {
            "H_Shots": H_Shots, "A_Shots": A_Shots,
            "H_SoT": H_SoT,     "A_SoT": A_SoT,
            "H_Corner": H_Corner,"A_Corner": A_Corner,
            "H_xG": H_xG,       "A_xG": A_xG
        }

def scrape_epl_2025_26_j1_j6(output_csv="data/fbref/epl_2025_26_stats_J1_J6.csv"):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    scraper = FBrefEPLScraper(pause=3, retries=3, timeout=20)

    html = scraper.get(FIXTURES_URL)
    if not html:
        print("❌ Impossible de charger la page fixtures")
        return None

    fixtures = scraper.parse_fixtures_links_j1_j6(html)
    rows = []
    for i, fx in enumerate(fixtures, 1):
        print(f"[{i}/{len(fixtures)}] {fx['date_raw']} - {fx['home']} vs {fx['away']}")
        match_html = scraper.get(fx["match_url"])
        if not match_html:
            print("   ⚠️ match page non accessible, on saute")
            continue
        stats = scraper.parse_match_stats(match_html)
        # Standardiser date
        date_std = fx["date_raw"]
        try:
            # FBref peut être "2025-08-17" déjà standard
            date_std = pd.to_datetime(date_std, errors="coerce").date().isoformat()
        except:
            pass
        row = {
            "Date": date_std,
            "Round": fx["round"],
            "HomeTeam": fx["home"],
            "AwayTeam": fx["away"],
            **stats
        }
        rows.append(row)
        time.sleep(scraper.pause)

    df = pd.DataFrame(rows)
    # Journalisation de complétude
    coverage = {
        "matches": len(df),
        "shots_coverage": float(df[["H_Shots","A_Shots"]].notna().all(axis=1).mean()),
        "sot_coverage": float(df[["H_SoT","A_SoT"]].notna().all(axis=1).mean()),
        "corners_coverage": float(df[["H_Corner","A_Corner"]].notna().all(axis=1).mean()),
        "xg_coverage": float(df[["H_xG","A_xG"]].notna().all(axis=1).mean()),
    }
    print("📊 Couverture:", coverage)

    df.to_csv(output_csv, index=False)
    print(f"💾 Export: {output_csv}")
    return output_csv

if __name__ == "__main__":
    scrape_epl_2025_26_j1_j6()