/**
 * Centralized App Configuration
 * ============================
 * Single source of truth for all app constants
 */

export const APP_CONFIG = {
  // Model Information
  MODEL_VERSION: "Enhanced Baseline v3.0",
  MODEL_ACCURACY: "51.3%", // From PRODUCTION_VALIDATION_20251104.md
  PIPELINE_VERSION: "Pipeline v1.0",
  
  // Season Information
  CURRENT_SEASON: "25/26",
  
  // Dataset Information
  DATASET_SIZE: "2280 matches (2019-2025)",
  DATASET_ACCURACY_RANGE: "51.3% ± 5.7%",
  
  // Validation Information
  VALIDATION_MATCHES: "Production validated",
  
  // API Configuration
  PREDICTIONS_CACHE_TIME: 5 * 60 * 1000, // 5 minutes
  STATS_CACHE_TIME: 60 * 60 * 1000, // 1 hour
} as const

/**
 * Calculate dynamic metrics based on current gameweek
 */
export const getDynamicMetrics = (currentGameweek: number) => ({
  totalPredictions: currentGameweek * 10,
  validatedMatches: Math.max(0, (currentGameweek - 1) * 10),
  currentGameweekLabel: `GW${currentGameweek}`,
  nextGameweekLabel: `GW${currentGameweek + 1}`,
})

/**
 * Stadium image mapping for dynamic gameweek data
 */
export const STADIUM_IMAGE_MAP: Record<string, string> = {
  // Main teams
  "Arsenal": "/images/Stades/Arsenal.avif",
  "Liverpool": "/images/Stades/Liverpool.webp", 
  "Manchester City": "/images/Stades/Manchester_City.jpg",
  "Man City": "/images/Stades/Manchester_City.jpg", // API uses this format
  "Chelsea": "/images/Stades/Chelsea.webp",
  "Newcastle": "/images/Stades/Newcastle.jpg",
  "Newcastle United": "/images/Stades/Newcastle.jpg",
  "Tottenham": "/images/Stades/Tottenham.jpg",
  "Tottenham Hotspur": "/images/Stades/Tottenham.jpg",
  "Manchester United": "/images/Stades/Manchester_United.jpg",
  "Man United": "/images/Stades/Manchester_United.jpg",
  
  // Additional teams that might appear
  "Aston Villa": "/images/Stades/Arsenal.avif", // Using default for now
  "Bournemouth": "/images/Stades/Arsenal.avif",
  "Brentford": "/images/Stades/Arsenal.avif", 
  "Wolves": "/images/Stades/Arsenal.avif",
  
  // Fallback image for teams without specific stadium images
  "default": "/images/Stades/Arsenal.avif"
}

/**
 * Get stadium image for a team
 */
export const getStadiumImage = (teamName: string): string => {
  return STADIUM_IMAGE_MAP[teamName] || STADIUM_IMAGE_MAP["default"]
}