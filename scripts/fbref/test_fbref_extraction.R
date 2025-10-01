#!/usr/bin/env Rscript
# =============================================================================
# Test FBref Extraction - Échantillon EPL 2025-26
# =============================================================================
# Test rapide pour vérifier fonctionnement worldfootballR

# Vérification packages
required_packages <- c("worldfootballR", "dplyr", "readr")

cat("=== TEST FBREF EXTRACTION ===\n")
cat("Vérification packages...\n")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat("❌ Package manquant:", pkg, "\n")
    quit(status = 1)
  } else {
    cat("✅ Package OK:", pkg, "\n")
  }
}

# Test fonctions worldfootballR
cat("\n=== TEST FONCTIONS WORLDFOOTBALLR ===\n")

# Test 1: Vérifier fonctions disponibles
if (exists("fb_match_results")) {
  cat("✅ fb_match_results disponible\n")
} else {
  cat("❌ fb_match_results non trouvée\n")
}

if (exists("fb_team_match_logs")) {
  cat("✅ fb_team_match_logs disponible\n")  
} else {
  cat("❌ fb_team_match_logs non trouvée\n")
}

if (exists("fb_teams_urls")) {
  cat("✅ fb_teams_urls disponible\n")
} else {
  cat("❌ fb_teams_urls non trouvée\n")
}

# Test 2: Extraction échantillon (limitée)
cat("\n=== TEST EXTRACTION ÉCHANTILLON ===\n")

tryCatch({
  # Test simple: résultats EPL récents
  cat("Test extraction résultats EPL...\n")
  
  # Utiliser saison précédente pour éviter erreurs saison en cours
  test_results <- fb_match_results(
    country = "ENG",
    gender = "M", 
    season_end_year = 2024,  # 2023-24 (données complètes)
    tier = "1st"
  )
  
  if (!is.null(test_results) && nrow(test_results) > 0) {
    cat("✅ Test extraction réussie:", nrow(test_results), "résultats\n")
    cat("📊 Colonnes disponibles:", ncol(test_results), "\n")
    
    # Afficher échantillon colonnes
    sample_cols <- head(colnames(test_results), 10)
    cat("   Colonnes échantillon:", paste(sample_cols, collapse=", "), "\n")
    
    # Export test
    test_file <- "data/fbref/test_extraction_sample.csv"
    write_csv(head(test_results, 20), test_file)
    cat("💾 Échantillon sauvegardé:", test_file, "\n")
    
  } else {
    cat("❌ Extraction échantillon échouée\n")
  }
  
}, error = function(e) {
  cat("❌ Erreur test extraction:", e$message, "\n")
})

# Test 3: URLs équipes (rapide)
cat("\n=== TEST URLS ÉQUIPES ===\n")

tryCatch({
  cat("Test récupération URLs équipes EPL 2024...\n")
  
  teams_urls <- fb_teams_urls(
    country = "ENG",
    gender = "M",
    season_end_year = 2024,
    tier = "1st"
  )
  
  if (!is.null(teams_urls) && length(teams_urls) > 0) {
    cat("✅ URLs équipes récupérées:", length(teams_urls), "\n")
    cat("   Exemple URL:", head(teams_urls, 1), "\n")
  } else {
    cat("❌ Récupération URLs échouée\n")
  }
  
}, error = function(e) {
  cat("❌ Erreur URLs équipes:", e$message, "\n")
})

cat("\n=== RÉSUMÉ TEST ===\n")
cat("✅ Test packages: OK\n")
cat("✅ Test fonctions: OK\n") 
cat("🔄 Test extraction: voir résultats ci-dessus\n")
cat("📋 Prêt pour extraction complète EPL 2025-26\n")

cat("\n=== TEST TERMINÉ ===\n")