'use client'

import { AlertTriangle, Wifi, WifiOff, RefreshCw } from 'lucide-react'
import { useState, useEffect } from 'react'

interface DegradedModeBannerProps {
  /**
   * Mode de dégradation détecté
   */
  mode?: 'static_fallback' | 'backend_unavailable' | 'partial_data'
  
  /**
   * Message personnalisé à afficher
   */
  message?: string
  
  /**
   * Callback appelé quand l'utilisateur clique sur "Réessayer"
   */
  onRetry?: () => void
  
  /**
   * Si true, affiche un bouton de retry
   */
  showRetry?: boolean
  
  /**
   * Si true, la bannière peut être fermée
   */
  dismissible?: boolean
  
  /**
   * Métadonnées additionnelles pour debugging
   */
  metadata?: {
    source?: string
    backend_status?: string
    last_update?: string
    [key: string]: any
  }
}

export function DegradedModeBanner({
  mode = 'backend_unavailable',
  message,
  onRetry,
  showRetry = true,
  dismissible = true,
  metadata
}: DegradedModeBannerProps) {
  const [isDismissed, setIsDismissed] = useState(false)
  const [isRetrying, setIsRetrying] = useState(false)

  // Auto-retry après 30 secondes en arrière-plan
  useEffect(() => {
    if (onRetry && !isDismissed) {
      const timer = setTimeout(() => {
        if (!isRetrying) {
          onRetry()
        }
      }, 30000)
      
      return () => clearTimeout(timer)
    }
  }, [onRetry, isDismissed, isRetrying])

  if (isDismissed) {
    return null
  }

  const handleRetry = async () => {
    if (onRetry && !isRetrying) {
      setIsRetrying(true)
      try {
        await onRetry()
      } finally {
        setIsRetrying(false)
      }
    }
  }

  const getIcon = () => {
    switch (mode) {
      case 'static_fallback':
        return <WifiOff className="h-5 w-5" />
      case 'backend_unavailable':
        return <Wifi className="h-5 w-5 text-red-500" />
      case 'partial_data':
        return <AlertTriangle className="h-5 w-5 text-yellow-500" />
      default:
        return <AlertTriangle className="h-5 w-5" />
    }
  }

  const getDefaultMessage = () => {
    switch (mode) {
      case 'static_fallback':
        return "Mode dégradé : Données statiques affichées"
      case 'backend_unavailable':
        return "Service principal temporairement indisponible"
      case 'partial_data':
        return "Données partielles - Certaines informations peuvent être manquantes"
      default:
        return "Mode dégradé activé"
    }
  }

  const getBannerStyle = () => {
    switch (mode) {
      case 'static_fallback':
        return "bg-orange-50 border-orange-200 text-orange-800"
      case 'backend_unavailable':
        return "bg-red-50 border-red-200 text-red-800"
      case 'partial_data':
        return "bg-yellow-50 border-yellow-200 text-yellow-800"
      default:
        return "bg-gray-50 border-gray-200 text-gray-800"
    }
  }

  return (
    <div className={`
      border rounded-lg p-4 mb-4 shadow-sm
      ${getBannerStyle()}
      transition-all duration-300 ease-in-out
      ${isDismissed ? 'opacity-0 transform -translate-y-2' : 'opacity-100'}
    `}>
      <div className="flex items-start justify-between">
        <div className="flex items-start space-x-3 flex-1">
          <div className="flex-shrink-0 pt-0.5">
            {getIcon()}
          </div>
          
          <div className="flex-1 min-w-0">
            <div className="font-medium text-sm">
              {message || getDefaultMessage()}
            </div>
            
            {metadata && (
              <div className="mt-1 text-xs opacity-75">
                {metadata.source && (
                  <span>Source: {metadata.source}</span>
                )}
                {metadata.last_update && (
                  <span className="ml-3">
                    Dernière mise à jour: {new Date(metadata.last_update).toLocaleTimeString()}
                  </span>
                )}
              </div>
            )}
            
            {mode === 'static_fallback' && (
              <div className="mt-2 text-xs opacity-90">
                Les prédictions affichées sont les dernières données disponibles. 
                Le système tente de rétablir la connexion automatiquement.
              </div>
            )}
          </div>
        </div>
        
        <div className="flex items-center space-x-2 ml-4">
          {showRetry && onRetry && (
            <button
              onClick={handleRetry}
              disabled={isRetrying}
              className={`
                inline-flex items-center px-3 py-1 text-xs font-medium rounded-md
                transition-colors duration-200
                ${mode === 'static_fallback' 
                  ? 'bg-orange-100 hover:bg-orange-200 text-orange-800' 
                  : mode === 'backend_unavailable'
                  ? 'bg-red-100 hover:bg-red-200 text-red-800'
                  : 'bg-yellow-100 hover:bg-yellow-200 text-yellow-800'
                }
                disabled:opacity-50 disabled:cursor-not-allowed
                focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-orange-500
              `}
            >
              <RefreshCw className={`h-3 w-3 mr-1 ${isRetrying ? 'animate-spin' : ''}`} />
              {isRetrying ? 'Reconnexion...' : 'Réessayer'}
            </button>
          )}
          
          {dismissible && (
            <button
              onClick={() => setIsDismissed(true)}
              className={`
                inline-flex items-center justify-center w-6 h-6 rounded-md
                transition-colors duration-200
                ${mode === 'static_fallback' 
                  ? 'hover:bg-orange-200 text-orange-600' 
                  : mode === 'backend_unavailable'
                  ? 'hover:bg-red-200 text-red-600'
                  : 'hover:bg-yellow-200 text-yellow-600'
                }
                focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-orange-500
              `}
              aria-label="Fermer la notification"
            >
              <span className="text-lg">×</span>
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

/**
 * Hook pour détecter automatiquement le mode dégradé depuis les headers HTTP
 */
export function useDegradedModeDetection() {
  const [degradedMode, setDegradedMode] = useState<{
    active: boolean
    mode?: 'static_fallback' | 'backend_unavailable' | 'partial_data'
    metadata?: any
  }>({ active: false })

  const checkDegradedMode = (response: Response) => {
    const fallbackMode = response.headers.get('X-Fallback-Mode')
    const dataSource = response.headers.get('X-Data-Source')
    const backendStatus = response.headers.get('X-Backend-Status')
    
    if (fallbackMode === 'true' || dataSource === 'static_fallback') {
      setDegradedMode({
        active: true,
        mode: 'static_fallback',
        metadata: {
          source: dataSource,
          backend_status: backendStatus,
          last_update: new Date().toISOString()
        }
      })
    } else if (backendStatus === 'unavailable') {
      setDegradedMode({
        active: true,
        mode: 'backend_unavailable',
        metadata: {
          source: dataSource,
          backend_status: backendStatus,
          last_update: new Date().toISOString()
        }
      })
    } else {
      setDegradedMode({ active: false })
    }
  }

  const resetDegradedMode = () => {
    setDegradedMode({ active: false })
  }

  return {
    degradedMode,
    checkDegradedMode,
    resetDegradedMode
  }
}