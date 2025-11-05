/**
 * Types TypeScript mirrored depuis backend Pydantic
 * ================================================
 * Types synchronisés avec schemas Python pour validation stricte
 */

// ===== COMMON TYPES =====

export interface APIMetadata {
  /** Version API */
  api_version: string;
  /** Version Pipeline Durci */
  pipeline_version: string;
  /** Timestamp génération données */
  generated_at: string;
  /** Version code déployée */
  git_sha?: string;
}

export interface APIResponse<T> {
  /** Métadonnées communes */
  meta: APIMetadata;
  /** Données de réponse */
  data: T;
  /** Message d'erreur éventuel */
  error?: string;
}

// ===== ODDS & BOOKMAKER TYPES =====

export type BookmakerId = 
  | "bet365" 
  | "pinnacle" 
  | "betfair" 
  | "william_hill" 
  | "ladbrokes" 
  | "unibet";

export type MarketConfidence = "high" | "medium" | "low";
export type OddsSource = "real" | "unavailable";

export interface OddsSnapshot {
  /** Bookmaker source */
  bookmaker: BookmakerId;
  /** Timestamp snapshot UTC strict (YYYY-MM-DDTHH:MM:SSZ) */
  snapshot_utc: string;
  /** Marge bookmaker (overround) */
  overround: number;
  /** Niveau confiance marché */
  market_confidence: MarketConfidence;
}

export interface Ko2hStatus {
  /** KO-2h constraint respecté */
  ok: boolean;
  /** Raison si non respecté */
  reason?: string;
  /** Minutes avant kickoff */
  minutes_to_kickoff?: number;
}

export interface MarketProbabilities {
  /** Probabilité vraie victoire domicile */
  home: number;
  /** Probabilité vraie match nul */
  draw: number;
  /** Probabilité vraie victoire extérieur */  
  away: number;
}

export interface OddsSelectionMetadata {
  /** Tier bookmaker utilisé */
  tier_used: "tier1" | "tier2" | "tier3";
  /** Nombre snapshots disponibles */
  snapshots_available: number;
  /** Cutoff KO-2h appliqué */
  ko2h_cutoff: string;
}

// ===== PREDICTION TYPES =====

export type PredictionOutcome = "H" | "D" | "A";

export interface PredictionProbabilities {
  /** Probabilité victoire domicile */
  home: number;
  /** Probabilité match nul */
  draw: number;
  /** Probabilité victoire extérieur */
  away: number;
}

export interface ModelPrediction {
  /** Prédiction du modèle */
  prediction: PredictionOutcome;
  /** Niveau de confiance */
  confidence: number;
  /** Probabilités détaillées */
  probabilities: PredictionProbabilities;
}

export interface EnsembleSystem {
  /** Nom système ensemble */
  system_name: string;
  /** Version ensemble */
  version: string;
  /** Configuration modèles */
  models: Record<string, any>;
  /** Stratégie agrégation */
  ensemble_strategy: string;
  /** Poids modèles */
  weights: Record<string, number>;
  /** Performance attendue */
  expected_performance: number;
}

export interface MatchPrediction {
  /** ID unique match */
  id: string;
  /** Équipe domicile */
  home_team: string;
  /** Équipe extérieur */
  away_team: string;
  /** Date match DD/MM/YYYY */
  date: string;
  /** Journée EPL */
  round: number;
  
  /** Prédiction ensemble */
  ensemble: ModelPrediction;
  
  /** Prédictions individuelles */
  models: Record<string, ModelPrediction>;
  
  /** Niveau désaccord modèles */
  disagreement: number;

  // ===== NOUVEAUX CHAMPS ODDS v5.3 =====
  
  /** Snapshot odds sélectionné */
  selected_snapshot?: OddsSnapshot;
  
  /** Status validation KO-2h */
  ko2h_ok: boolean;
  
  /** Source des odds */
  odds_source: OddsSource;
  
  /** Raison si odds indisponibles */
  missing_reason?: string;
  
  /** Probabilités marché vraies */
  market_probs_raw?: MarketProbabilities;
  
  /** Métadonnées sélection */
  selection_metadata?: OddsSelectionMetadata;
  
  /** Status individuel fixture */
  individual_status: "ready" | "blocked" | "ko2h_violation";
}

export interface RoundPredictions {
  /** Numéro journée */
  round: number;
  /** Saison */
  season: string;
  /** Compétition */
  competition: string;
  /** Nombre matchs */
  total_matches: number;
  
  /** Configuration ensemble */
  ensemble_system: EnsembleSystem;
  /** Liste des matchs */
  matches: MatchPrediction[];
}

// ===== PIPELINE TYPES =====

export type ComponentHealth = "healthy" | "error" | "unknown";

export interface PipelineStatus {
  /** Version Pipeline Durci */
  pipeline_version: string;
  /** Dernière exécution */
  last_run?: string;
  /** Status composants */
  components_status: Record<string, ComponentHealth>;
  /** Prochaine exécution */
  next_scheduled_run?: string;
  /** Fraîcheur des données */
  data_freshness: Record<string, any>;
}

export interface JobStatusResponse {
  /** ID du job */
  job_id: string;
  /** Journée à générer */
  round: number;
  /** Status actuel */
  status: string;
  /** Date création */
  created_at: string;
  /** Date démarrage */
  started_at?: string;
  /** Date completion */
  completed_at?: string;
  /** Message d'erreur */
  error_message?: string;
}

// ===== ODDS HEALTH STATUS TYPES =====

export interface BookmakerCoverage {
  [bookmaker: string]: number;
}

export interface OddsConfiguration {
  /** Bookmakers requis par tier */
  required_bookmakers: {
    tier1: BookmakerId[];
    tier2: BookmakerId[];
    tier3: BookmakerId[];
  };
  /** Stratégie sélection */
  selection_strategy: {
    mode: "intelligent" | "strict" | "permissive";
    minimum_tier1: number;
    minimum_total: number;
  };
  /** Saison courante */
  current_season: string;
}

export interface OddsRealTimeStats {
  /** Timestamp */
  timestamp: string;
  /** Fichiers odds disponibles */
  odds_files_available: number;
  /** Total snapshots */
  total_snapshots: number;
  /** Fixtures uniques */
  unique_fixtures: number;
  /** Bookmakers uniques */
  unique_bookmakers: number;
  /** Fraîcheur en minutes */
  freshness_minutes: number | null;
}

export interface OddsCoverage {
  /** Coverage par bookmaker */
  bookmaker_coverage: BookmakerCoverage;
  /** Total fixtures */
  total_fixtures: number;
  /** Pourcentage coverage */
  coverage_percentage: number;
}

export interface OddsHealthValidation {
  /** Status validator */
  validator_status: "healthy" | "degraded" | "error";
  /** Résumé rapport validation */
  validation_report_summary: {
    errors_count: number;
    warnings_count: number;
    production_ready: boolean;
  };
  /** Compliance SLA */
  sla_compliance: Record<string, any>;
}

export interface OddsCompliance {
  /** Status compliance */
  status: "healthy" | "warning" | "degraded" | "critical";
  /** Issues détectées */
  issues: string[];
  /** Prêt pour production */
  ready_for_production: boolean;
}

export interface OddsHealthStatus {
  /** Status global */
  status: "healthy" | "warning" | "degraded" | "critical" | "error";
  /** Timestamp */
  timestamp: string;
  /** Saison */
  season: string;
  /** Stats temps réel */
  real_time_stats: OddsRealTimeStats;
  /** Configuration */
  configuration: OddsConfiguration;
  /** Coverage */
  coverage: OddsCoverage;
  /** Validation santé */
  health_validation: OddsHealthValidation;
  /** Compliance */
  compliance: OddsCompliance;
  /** Message erreur si applicable */
  error?: string;
  /** Message détaillé */
  message?: string;
}

// ===== UI HELPER TYPES =====

export interface PredictionCardProps {
  match: MatchPrediction;
  showModels?: boolean;
  compact?: boolean;
}

export interface DualChampionsViewProps {
  system: EnsembleSystem;
  title?: string;
}

export interface PipelineStatusProps {
  status: PipelineStatus;
  showDetails?: boolean;
}

// ===== API CLIENT TYPES =====

export interface OddsyAPIConfig {
  baseURL: string;
  timeout?: number;
  apiVersion?: string;
}

export interface FetchOptions {
  cache?: RequestCache;
  revalidate?: number;
}

// ===== CACHE KEYS =====

export const CACHE_KEYS = {
  predictions: (season: string, round: number) => ['predictions', season, round],
  pipeline_status: () => ['pipeline', 'status'],
  models_performance: () => ['models', 'performance'],
  available_rounds: () => ['predictions', 'available']
} as const;

// ===== CONSTANTS =====

export const PREDICTION_OUTCOMES = {
  H: { label: "Home Win", icon: "🏠", color: "text-blue-600" },
  D: { label: "Draw", icon: "🤝", color: "text-yellow-600" },
  A: { label: "Away Win", icon: "✈️", color: "text-red-600" }
} as const;

export const CONFIDENCE_LEVELS = {
  HIGH: { min: 0.7, label: "High", color: "text-green-600", bgColor: "bg-green-50" },
  MEDIUM: { min: 0.55, label: "Medium", color: "text-yellow-600", bgColor: "bg-yellow-50" },
  LOW: { min: 0, label: "Low", color: "text-red-600", bgColor: "bg-red-50" }
} as const;

export const MODEL_BADGES = {
  enhanced_v24: { label: "Enhanced v3.0", color: "bg-green-600", icon: "🏆" },
  baseline: { label: "Enhanced v3.0", color: "bg-green-600", icon: "🏆" },
  ensemble: { label: "Ensemble", color: "bg-blue-600", icon: "⚡" }
} as const;

// ===== VALIDATION HELPERS =====

export function validateProbabilities(probs: PredictionProbabilities): boolean {
  const total = probs.home + probs.draw + probs.away;
  return total >= 0.98 && total <= 1.02;
}

export function getConfidenceLevel(confidence: number): keyof typeof CONFIDENCE_LEVELS {
  if (confidence >= CONFIDENCE_LEVELS.HIGH.min) return "HIGH";
  if (confidence >= CONFIDENCE_LEVELS.MEDIUM.min) return "MEDIUM";
  return "LOW";
}

export function formatProbability(prob: number): string {
  return `${(prob * 100).toFixed(0)}%`;
}

export function formatConfidence(confidence: number): string {
  return `${(confidence * 100).toFixed(1)}%`;
}