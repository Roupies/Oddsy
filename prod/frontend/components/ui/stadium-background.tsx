'use client'

import { useState, useEffect } from 'react'
import { clsx } from 'clsx'

interface StadiumBackgroundProps {
  homeTeam: string
  className?: string
  overlayOpacity?: number
}

// Mapping équipes → images de stades (équipe à domicile)
const STADIUM_MAP: Record<string, string> = {
  // Premier League Teams 2024-25 - Toutes images disponibles ✅
  'Arsenal': '/images/Stades/Arsenal.avif',
  'Aston Villa': '/images/Stades/Aston_Villa.jpg',
  'Bournemouth': '/images/Stades/Bournemouth.webp',
  'Brentford': '/images/Stades/Brentford.avif',
  'Brighton and Hove Albion': '/images/Stades/Brighton.jpg',
  'Burnley': '/images/Stades/Burnley.jpg',
  'Chelsea': '/images/Stades/Chelsea.webp',
  'Crystal Palace': '/images/Stades/Crystal_Palace.avif',
  'Everton': '/images/Stades/Everton.avif',
  'Fulham': '/images/Stades/Fulham.png',
  'Leeds United': '/images/Stades/Leeds_United.webp',
  'Liverpool': '/images/Stades/Liverpool.webp',
  'Manchester City': '/images/Stades/Manchester_City.jpg',
  'Manchester United': '/images/Stades/Manchester_United.avif',
  'Newcastle United': '/images/Stades/Newcastle.jpg',
  'Nottingham Forest': '/images/Stades/Nottingham_Forest.jpg',
  'Sunderland': '/images/Stades/Sunderland.jpg',
  'Tottenham Hotspur': '/images/Stades/Tottenham.jpg',
  'West Ham United': '/images/Stades/West_Ham.avif',
  'Wolverhampton Wanderers': '/images/Stades/Wolverhampton.jpeg',
  
  // Alias pour compatibilité API
  'Man City': '/images/Stades/Manchester_City.jpg',
  'Man United': '/images/Stades/Manchester_United.avif',
  'Brighton': '/images/Stades/Brighton.jpg',
  'Spurs': '/images/Stades/Tottenham.jpg',
  'Wolves': '/images/Stades/Wolverhampton.jpeg',
  'West Ham': '/images/Stades/West_Ham.avif',
  // Alias pour prédictions J9
  'Leeds': '/images/Stades/Leeds_United.webp',
  'Man Utd': '/images/Stades/Manchester_United.avif',
  'Newcastle': '/images/Stades/Newcastle.jpg',
  'Nott\'m Forest': '/images/Stades/Nottingham_Forest.jpg',
  'Crystal Palace': '/images/Stades/Crystal_Palace.avif'
}

// Version statique du cache busting pour éviter les re-renders constants
const CACHE_VERSION = "v20251020"

export function StadiumBackground({ 
  homeTeam, 
  className = '',
  overlayOpacity = 0.15 
}: StadiumBackgroundProps) {
  const [imageError, setImageError] = useState(false)
  const [imageLoaded, setImageLoaded] = useState(false)
  const stadiumImage = STADIUM_MAP[homeTeam]
  
  // Preload de l'image pour détecter les erreurs
  useEffect(() => {
    if (stadiumImage) {
      // Reset states when stadium image changes
      setImageLoaded(false)
      setImageError(false)
      
      const img = new window.Image()
      img.onload = () => {
        setImageLoaded(true)
        setImageError(false)
      }
      img.onerror = () => {
        setImageError(true)
        setImageLoaded(false)
      }
      // Use static cache version to force reload
      img.src = `${stadiumImage}?${CACHE_VERSION}`
    }
  }, [stadiumImage])
  
  // Si pas d'image de stade disponible, utiliser un fond dégradé équipe-spécifique
  if (!stadiumImage || imageError) {
    const teamColors = getTeamGradient(homeTeam)
    
    return (
      <div className={clsx(
        'absolute inset-0 rounded-2xl',
        teamColors,
        className
      )} />
    )
  }
  
  return (
    <div className={clsx('absolute inset-0 overflow-hidden rounded-2xl', className)}>
      {/* Placeholder gradient pendant le chargement */}
      {!imageLoaded && (
        <div className={clsx(
          'absolute inset-0 rounded-2xl transition-opacity duration-300',
          getTeamGradient(homeTeam)
        )} />
      )}
      
      {/* Image de stade */}
      <div 
        className={clsx(
          'absolute inset-0 bg-cover bg-center rounded-2xl transition-opacity duration-500',
          imageLoaded ? 'opacity-100' : 'opacity-0'
        )}
        style={{ backgroundImage: `url(${stadiumImage}?${CACHE_VERSION})` }}
      />
      
      {/* Overlay pour la lisibilité (seulement si image chargée) */}
      {imageLoaded && (
        <div 
          className="absolute inset-0 bg-black transition-opacity duration-300 rounded-2xl"
          style={{ opacity: overlayOpacity }}
        />
      )}
      
      {/* Gradient subtil pour le contraste (seulement si image chargée) */}
      {imageLoaded && (
        <div className="absolute inset-0 bg-gradient-to-t from-black/20 via-transparent via-70% to-black/10 rounded-2xl" />
      )}
    </div>
  )
}

// Fonction pour générer des dégradés basés sur les couleurs des équipes
function getTeamGradient(teamName: string): string {
  const gradients: Record<string, string> = {
    'Arsenal': 'bg-gradient-to-br from-red-600 via-red-500 to-red-400',
    'Aston Villa': 'bg-gradient-to-br from-purple-700 via-purple-600 to-blue-700',
    'Bournemouth': 'bg-gradient-to-br from-red-700 via-black to-red-600',
    'Brentford': 'bg-gradient-to-br from-red-600 via-yellow-500 to-red-500',
    'Brighton and Hove Albion': 'bg-gradient-to-br from-blue-600 via-blue-500 to-white',
    'Burnley': 'bg-gradient-to-br from-purple-800 via-blue-800 to-purple-700',
    'Chelsea': 'bg-gradient-to-br from-blue-700 via-blue-600 to-blue-500',
    'Crystal Palace': 'bg-gradient-to-br from-blue-700 via-red-600 to-blue-600',
    'Everton': 'bg-gradient-to-br from-blue-800 via-blue-700 to-blue-600',
    'Fulham': 'bg-gradient-to-br from-black via-white to-black',
    'Leeds United': 'bg-gradient-to-br from-white via-yellow-400 to-blue-600',
    'Liverpool': 'bg-gradient-to-br from-red-700 via-red-600 to-red-500',
    'Manchester City': 'bg-gradient-to-br from-sky-500 via-sky-400 to-sky-300',
    'Manchester United': 'bg-gradient-to-br from-red-700 via-red-600 to-yellow-500',
    'Newcastle United': 'bg-gradient-to-br from-black via-white to-black',
    'Nottingham Forest': 'bg-gradient-to-br from-red-700 via-red-600 to-red-500',
    'Sunderland': 'bg-gradient-to-br from-red-700 via-red-600 to-white',
    'Tottenham Hotspur': 'bg-gradient-to-br from-white via-blue-800 to-white',
    'West Ham United': 'bg-gradient-to-br from-purple-800 via-blue-800 to-purple-700',
    'Wolverhampton Wanderers': 'bg-gradient-to-br from-yellow-600 via-black to-yellow-500',
    // Alias pour prédictions J9
    'Leeds': 'bg-gradient-to-br from-white via-yellow-400 to-blue-600',
    'Man Utd': 'bg-gradient-to-br from-red-700 via-red-600 to-yellow-500',
    'Man City': 'bg-gradient-to-br from-sky-500 via-sky-400 to-sky-300',
    'Newcastle': 'bg-gradient-to-br from-black via-white to-black',
    'Nott\'m Forest': 'bg-gradient-to-br from-red-700 via-red-600 to-red-500',
    'Spurs': 'bg-gradient-to-br from-white via-blue-800 to-white',
    'Brighton': 'bg-gradient-to-br from-blue-600 via-blue-500 to-white',
    'West Ham': 'bg-gradient-to-br from-purple-800 via-blue-800 to-purple-700',
    'Wolves': 'bg-gradient-to-br from-yellow-600 via-black to-yellow-500',
    'Crystal Palace': 'bg-gradient-to-br from-blue-700 via-red-600 to-blue-600'
  }
  
  return gradients[teamName] || 'bg-gradient-to-br from-blue-100 via-gray-50 to-blue-50'
}