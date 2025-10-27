/**
 * Système de couleurs pour la confiance des prédictions
 * Palette "bleue encourageante" remplaçant le rouge alarmant
 */

export interface ConfidenceColorScheme {
  bg: string
  text: string
  bar: string
  border: string
  badge: string
  label: string
}

// Nouveau système de couleurs encourageant
export const CONFIDENCE_COLORS: Record<string, ConfidenceColorScheme> = {
  low: {
    bg: 'bg-blue-50',
    text: 'text-blue-700',
    bar: 'bg-blue-300',
    border: 'border-blue-200',
    badge: 'bg-blue-100',
    label: 'Uncertain Prediction'
  },
  moderate: {
    bg: 'bg-blue-100', 
    text: 'text-blue-800',
    bar: 'bg-blue-400',
    border: 'border-blue-300',
    badge: 'bg-blue-200',
    label: 'Moderate Prediction'
  },
  good: {
    bg: 'bg-blue-600',
    text: 'text-white',
    bar: 'bg-blue-500',
    border: 'border-blue-500',
    badge: 'bg-blue-500',
    label: 'Confident Prediction'
  },
  excellent: {
    bg: 'bg-green-500',
    text: 'text-white', 
    bar: 'bg-green-400',
    border: 'border-green-400',
    badge: 'bg-green-400',
    label: 'High Confidence'
  }
}

/**
 * Détermine le niveau de confiance selon le score
 */
export function getConfidenceLevel(confidence: number): keyof typeof CONFIDENCE_COLORS {
  if (confidence >= 0.60) return 'excellent'  // 60%+
  if (confidence >= 0.45) return 'good'       // 45-60%  
  if (confidence >= 0.30) return 'moderate'   // 30-45%
  return 'low'                                // 0-30%
}

/**
 * Récupère le schéma de couleurs pour un niveau de confiance
 */
export function getConfidenceColors(confidence: number): ConfidenceColorScheme {
  const level = getConfidenceLevel(confidence)
  return CONFIDENCE_COLORS[level]
}

/**
 * Couleur de la barre de progression selon la confiance
 */
export function getProgressBarColor(confidence: number): string {
  const percentage = confidence * 100
  
  if (percentage >= 60) return 'bg-green-500'
  if (percentage >= 45) return 'bg-blue-600' 
  if (percentage >= 30) return 'bg-blue-400'
  return 'bg-blue-300'
}

/**
 * Formatage du pourcentage de confiance
 */
export function formatConfidence(confidence: number): string {
  return `${(confidence * 100).toFixed(1)}%`
}