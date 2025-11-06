/**
 * Utility Functions for Oddsy Frontend
 * ===================================
 * 
 * This module provides common utility functions used across the frontend.
 * Main purpose is CSS class name manipulation for Tailwind CSS.
 */

// clsx: Library for constructing className strings conditionally
// ClassValue: TypeScript type for clsx input values
import { type ClassValue, clsx } from 'clsx'

// twMerge: Tailwind CSS class merging utility to handle conflicts
import { twMerge } from 'tailwind-merge'

/**
 * Combine and merge CSS class names intelligently
 * 
 * This function combines clsx for conditional class construction
 * with twMerge for Tailwind CSS class conflict resolution.
 * 
 * Example:
 * cn('px-2 py-1', 'px-4', isActive && 'bg-blue-500')
 * // Returns: 'px-4 py-1 bg-blue-500' (px-2 is overridden by px-4)
 * 
 * @param inputs - Variable number of class values (strings, objects, arrays, etc.)
 * @returns Merged and optimized class string
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}