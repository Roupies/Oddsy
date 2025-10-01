#!/usr/bin/env Rscript
# =============================================================================
# Installation packages R pour FBref extraction
# =============================================================================

cat("=== INSTALLATION PACKAGES R POUR FBREF ===\n")

# Liste des packages requis
required_packages <- c(
  "worldfootballR",
  "dplyr", 
  "readr",
  "jsonlite",
  "httr",
  "rvest"
)

cat("Packages requis:\n")
for (pkg in required_packages) {
  cat("  -", pkg, "\n")
}

# Fonction installation sécurisée
install_package_safe <- function(package_name) {
  cat("\n--- Installation", package_name, "---\n")
  
  if (package_name %in% rownames(installed.packages())) {
    cat("✅", package_name, "déjà installé\n")
    return(TRUE)
  }
  
  tryCatch({
    install.packages(package_name, repos = "https://cran.rstudio.com/")
    cat("✅", package_name, "installé avec succès\n")
    return(TRUE)
  }, error = function(e) {
    cat("❌ Erreur installation", package_name, ":", e$message, "\n")
    return(FALSE)
  })
}

# Installation de tous les packages
cat("\n=== DÉBUT INSTALLATION ===\n")
installation_results <- sapply(required_packages, install_package_safe)

# Vérification des installations
cat("\n=== VÉRIFICATION INSTALLATIONS ===\n")
for (pkg in required_packages) {
  if (pkg %in% rownames(installed.packages())) {
    cat("✅", pkg, "disponible\n")
  } else {
    cat("❌", pkg, "NON disponible\n")
  }
}

# Test de chargement worldfootballR
cat("\n=== TEST WORLDFOOTBALLR ===\n")
tryCatch({
  library(worldfootballR)
  cat("✅ worldfootballR chargé avec succès\n")
  
  # Test fonction basique
  if (exists("fb_match_results")) {
    cat("✅ Fonction fb_match_results disponible\n")
  } else {
    cat("❌ Fonction fb_match_results non trouvée\n")
  }
  
}, error = function(e) {
  cat("❌ Erreur chargement worldfootballR:", e$message, "\n")
})

cat("\n=== INSTALLATION TERMINÉE ===\n")
cat("Prêt pour extraction FBref\n")