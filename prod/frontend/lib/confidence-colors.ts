/**
 * Prediction Confidence Color System for Oddsy
 * =============================================
 * 
 * This module defines the color scheme and utilities for displaying
 * prediction confidence levels in the UI. Uses an encouraging blue
 * palette instead of alarming red colors to create a positive UX.
 * 
 * Color Philosophy:
 * - Blue tones for uncertainty (calming, not alarming)
 * - Green for high confidence (positive reinforcement)
 * - Graduated intensity based on confidence levels
 */

/**
 * Configuration object for a complete confidence level color scheme
 * Includes all UI elements that need color coordination
 */
export interface ConfidenceColorScheme {
  bg: string      // Background color class
  text: string    // Text color class
  bar: string     // Progress bar color class
  border: string  // Border color class
  badge: string   // Badge/chip background color class
  label: string   // Human-readable confidence label
}

/**
 * Complete color scheme definitions for all confidence levels
 * Uses encouraging color palette design for positive user experience
 */
export const CONFIDENCE_COLORS: Record<string, ConfidenceColorScheme> = {
  // Low confidence (0-30%): Light blue - uncertain but not alarming
  low: {
    bg: 'bg-blue-50',
    text: 'text-blue-700',
    bar: 'bg-blue-300',
    border: 'border-blue-200',
    badge: 'bg-blue-100',
    label: 'Uncertain Prediction'
  },
  // Moderate confidence (30-45%): Medium blue - growing confidence
  moderate: {
    bg: 'bg-blue-100', 
    text: 'text-blue-800',
    bar: 'bg-blue-400',
    border: 'border-blue-300',
    badge: 'bg-blue-200',
    label: 'Moderate Prediction'
  },
  // Good confidence (45-60%): Strong blue - solid prediction
  good: {
    bg: 'bg-blue-600',
    text: 'text-white',
    bar: 'bg-blue-500',
    border: 'border-blue-500',
    badge: 'bg-blue-500',
    label: 'Confident Prediction'
  },
  // Excellent confidence (60%+): Green - highest confidence level
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
 * Determine confidence level category based on numerical score
 * 
 * Converts a confidence score (0-1) into one of four categorical levels
 * used for color scheme selection and UI display.
 * 
 * Thresholds:
 * - excellent: 60%+ (0.60+)
 * - good: 45-60% (0.45-0.59)  
 * - moderate: 30-45% (0.30-0.44)
 * - low: 0-30% (0.00-0.29)
 * 
 * @param confidence - Confidence score as decimal (0.0 to 1.0)
 * @returns Confidence level key for CONFIDENCE_COLORS lookup
 */
export function getConfidenceLevel(confidence: number): keyof typeof CONFIDENCE_COLORS {
  if (confidence >= 0.60) return 'excellent'  // 60%+ - High confidence
  if (confidence >= 0.45) return 'good'       // 45-60% - Good confidence  
  if (confidence >= 0.30) return 'moderate'   // 30-45% - Moderate confidence
  return 'low'                                // 0-30% - Low confidence
}

/**
 * Get complete color scheme for a given confidence level
 * 
 * Convenience function that combines confidence level determination
 * and color scheme lookup in a single call.
 * 
 * @param confidence - Confidence score as decimal (0.0 to 1.0)
 * @returns Complete color scheme object with all UI color classes
 */
export function getConfidenceColors(confidence: number): ConfidenceColorScheme {
  const level = getConfidenceLevel(confidence)
  return CONFIDENCE_COLORS[level]
}

/**
 * Get progress bar color based on confidence score
 * 
 * Returns specific Tailwind CSS background color class for progress bars.
 * Uses same thresholds as getConfidenceLevel but optimized for progress visualization.
 * 
 * @param confidence - Confidence score as decimal (0.0 to 1.0)
 * @returns Tailwind CSS background color class
 */
export function getProgressBarColor(confidence: number): string {
  const percentage = confidence * 100
  
  if (percentage >= 60) return 'bg-green-500'   // High confidence - green
  if (percentage >= 45) return 'bg-blue-600'    // Good confidence - strong blue
  if (percentage >= 30) return 'bg-blue-400'    // Moderate confidence - medium blue
  return 'bg-blue-300'                          // Low confidence - light blue
}

/**
 * Format confidence score as a percentage string
 * 
 * Converts decimal confidence to human-readable percentage
 * with one decimal place precision.
 * 
 * @param confidence - Confidence score as decimal (0.0 to 1.0)
 * @returns Formatted percentage string (e.g., "67.5%")
 */
export function formatConfidence(confidence: number): string {
  return `${(confidence * 100).toFixed(1)}%`
}