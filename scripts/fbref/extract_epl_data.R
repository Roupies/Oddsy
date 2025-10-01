#!/usr/bin/env Rscript
# =============================================================================
# FBref Data Extraction - EPL 2025-26
# =============================================================================
# Extrait xG, tirs, corners depuis FBref via worldfootballR
# Export CSV horodaté pour intégration Oddsy

library(worldfootballR)
library(dplyr)
library(readr)

# Configuration
SEASON <- 2025
COUNTRY <- "ENG"
TIER <- "1st"
PAUSE_SECONDS <- 3

# Créer horodatage pour fichiers
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
output_dir <- "data/fbref"

cat("=== EXTRACTION FBREF EPL 2025-26 ===\n")
cat("Timestamp:", timestamp, "\n")
cat("Saison:", SEASON, "\n")

# Fonction pour logs sécurisés
safe_log <- function(message) {
  cat("[", format(Sys.time(), "%H:%M:%S"), "] ", message, "\n", sep="")
}

# Fonction extraction avec gestion erreurs
safe_extract <- function(func_call, desc) {
  safe_log(paste("Début extraction:", desc))
  
  tryCatch({
    result <- func_call
    safe_log(paste("✅", desc, "- Succès:", nrow(result), "lignes"))
    return(result)
  }, error = function(e) {
    safe_log(paste("❌", desc, "- Erreur:", e$message))
    return(NULL)
  })
}

# =============================================================================
# 1. EXTRACTION RÉSULTATS MATCHS
# =============================================================================
safe_log("=== 1. EXTRACTION RÉSULTATS MATCHS ===")

results <- safe_extract(
  fb_match_results(
    country = COUNTRY,
    gender = "M", 
    season_end_year = SEASON,
    tier = TIER
  ),
  "Résultats matchs EPL 2025-26"
)

if (!is.null(results)) {
  # Export résultats
  results_file <- file.path(output_dir, paste0("epl_2025_26_results_", timestamp, ".csv"))
  write_csv(results, results_file)
  safe_log(paste("💾 Résultats sauvegardés:", results_file))
}

# Pause respectueuse
Sys.sleep(PAUSE_SECONDS)

# =============================================================================
# 2. EXTRACTION TEAM LOGS (xG, tirs, corners)
# =============================================================================
safe_log("=== 2. EXTRACTION TEAM LOGS ===")

# Obtenir URLs équipes EPL 2025-26
teams_urls <- safe_extract(
  fb_teams_urls(
    country = COUNTRY,
    gender = "M",
    season_end_year = SEASON, 
    tier = TIER
  ),
  "URLs équipes EPL"
)

if (!is.null(teams_urls) && length(teams_urls) > 0) {
  safe_log(paste("Équipes trouvées:", length(teams_urls)))
  
  # Extraire logs pour chaque équipe
  all_logs <- list()
  
  for (i in seq_along(teams_urls)) {
    team_url <- teams_urls[i]
    safe_log(paste("Extraction équipe", i, "/", length(teams_urls)))
    
    # Team match logs avec stats summary (contient xG, tirs, corners)
    team_logs <- safe_extract(
      fb_team_match_logs(
        team_urls = team_url,
        stat_type = "summary",
        time_pause = PAUSE_SECONDS
      ),
      paste("Team logs équipe", i)
    )
    
    if (!is.null(team_logs)) {
      all_logs[[i]] <- team_logs
    }
    
    # Pause entre équipes
    Sys.sleep(PAUSE_SECONDS)
  }
  
  # Combiner tous les logs
  if (length(all_logs) > 0) {
    combined_logs <- bind_rows(all_logs)
    safe_log(paste("✅ Logs combinés:", nrow(combined_logs), "lignes"))
    
    # Export team logs
    logs_file <- file.path(output_dir, paste0("epl_2025_26_team_logs_", timestamp, ".csv"))
    write_csv(combined_logs, logs_file)
    safe_log(paste("💾 Team logs sauvegardés:", logs_file))
    
    # Afficher colonnes disponibles
    safe_log("Colonnes disponibles:")
    cols <- colnames(combined_logs)
    for (col in cols) {
      cat("  -", col, "\n")
    }
  }
}

# =============================================================================
# 3. EXTRACTION DONNÉES SUPPLÉMENTAIRES (optionnel)
# =============================================================================
safe_log("=== 3. EXTRACTION DONNÉES SHOOTING (optionnel) ===")

# Tentative extraction stats shooting détaillées
if (!is.null(teams_urls) && length(teams_urls) > 0) {
  
  # Prendre seulement les 3 premières équipes pour test
  test_teams <- head(teams_urls, 3)
  shooting_logs <- list()
  
  for (i in seq_along(test_teams)) {
    team_url <- test_teams[i]
    safe_log(paste("Test shooting équipe", i, "/", length(test_teams)))
    
    shooting_data <- safe_extract(
      fb_team_match_logs(
        team_urls = team_url,
        stat_type = "shooting", 
        time_pause = PAUSE_SECONDS
      ),
      paste("Shooting logs équipe", i)
    )
    
    if (!is.null(shooting_data)) {
      shooting_logs[[i]] <- shooting_data
    }
    
    Sys.sleep(PAUSE_SECONDS)
  }
  
  if (length(shooting_logs) > 0) {
    combined_shooting <- bind_rows(shooting_logs)
    shooting_file <- file.path(output_dir, paste0("epl_2025_26_shooting_sample_", timestamp, ".csv"))
    write_csv(combined_shooting, shooting_file)
    safe_log(paste("💾 Shooting sample sauvegardé:", shooting_file))
  }
}

# =============================================================================
# 4. MÉTADONNÉES EXTRACTION
# =============================================================================
safe_log("=== 4. GÉNÉRATION MÉTADONNÉES ===")

metadata <- list(
  timestamp = timestamp,
  season = SEASON,
  country = COUNTRY,
  tier = TIER,
  extraction_time = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  files_generated = list()
)

if (!is.null(results)) {
  metadata$files_generated$results <- paste0("epl_2025_26_results_", timestamp, ".csv")
  metadata$results_count <- nrow(results)
}

if (exists("combined_logs") && !is.null(combined_logs)) {
  metadata$files_generated$team_logs <- paste0("epl_2025_26_team_logs_", timestamp, ".csv")
  metadata$team_logs_count <- nrow(combined_logs)
  metadata$columns_available <- colnames(combined_logs)
}

if (exists("combined_shooting") && !is.null(combined_shooting)) {
  metadata$files_generated$shooting_sample <- paste0("epl_2025_26_shooting_sample_", timestamp, ".csv")
  metadata$shooting_sample_count <- nrow(combined_shooting)
}

# Export métadonnées JSON
metadata_file <- file.path(output_dir, paste0("extraction_metadata_", timestamp, ".json"))
jsonlite::write_json(metadata, metadata_file, pretty = TRUE)
safe_log(paste("💾 Métadonnées sauvegardées:", metadata_file))

# =============================================================================
# 5. RÉSUMÉ FINAL
# =============================================================================
safe_log("=== RÉSUMÉ EXTRACTION ===")
safe_log(paste("✅ Extraction terminée:", timestamp))

if (!is.null(results)) {
  safe_log(paste("📊 Résultats matchs:", nrow(results), "lignes"))
}

if (exists("combined_logs") && !is.null(combined_logs)) {
  safe_log(paste("📊 Team logs:", nrow(combined_logs), "lignes"))
  safe_log(paste("📊 Colonnes:", ncol(combined_logs)))
}

safe_log("📁 Fichiers générés dans data/fbref/")
safe_log("🔄 Prêt pour intégration Python")

cat("\n=== EXTRACTION FBREF TERMINÉE ===\n")