'use client'

import { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { clsx } from 'clsx'
import { X, CheckCircle, AlertCircle, AlertTriangle, RefreshCw } from 'lucide-react'

export type ToastType = 'success' | 'error' | 'warning' | 'info'

export interface Toast {
  id: string
  type: ToastType
  title: string
  message?: string
  duration?: number
  action?: {
    label: string
    onClick: () => void
  }
  onRetry?: () => void
}

interface ToastContextType {
  toasts: Toast[]
  addToast: (toast: Omit<Toast, 'id'>) => string
  removeToast: (id: string) => void
  clearAll: () => void
}

const ToastContext = createContext<ToastContextType | undefined>(undefined)

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])

  const addToast = (toastData: Omit<Toast, 'id'>): string => {
    const id = Math.random().toString(36).substr(2, 9)
    const toast: Toast = {
      id,
      duration: 5000,
      ...toastData
    }
    
    setToasts(prev => [...prev, toast])
    
    // Auto-remove après duration
    if (toast.duration && toast.duration > 0) {
      setTimeout(() => {
        removeToast(id)
      }, toast.duration)
    }
    
    return id
  }

  const removeToast = (id: string) => {
    setToasts(prev => prev.filter(toast => toast.id !== id))
  }

  const clearAll = () => {
    setToasts([])
  }

  return (
    <ToastContext.Provider value={{ toasts, addToast, removeToast, clearAll }}>
      {children}
      <ToastContainer />
    </ToastContext.Provider>
  )
}

export function useToast() {
  const context = useContext(ToastContext)
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider')
  }
  return context
}

function ToastContainer() {
  const { toasts } = useToast()

  return (
    <div className="fixed top-4 right-4 z-50 space-y-2 max-w-sm w-full">
      {toasts.map((toast, index) => (
        <ToastItem 
          key={toast.id} 
          toast={toast} 
          index={index}
        />
      ))}
    </div>
  )
}

interface ToastItemProps {
  toast: Toast
  index: number
}

function ToastItem({ toast, index }: ToastItemProps) {
  const { removeToast } = useToast()
  const [isVisible, setIsVisible] = useState(false)
  const [isRemoving, setIsRemoving] = useState(false)

  useEffect(() => {
    // Animation d'entrée
    const timer = setTimeout(() => setIsVisible(true), 50)
    return () => clearTimeout(timer)
  }, [])

  const handleRemove = () => {
    setIsRemoving(true)
    setTimeout(() => removeToast(toast.id), 300)
  }

  const handleRetry = () => {
    if (toast.onRetry) {
      toast.onRetry()
      handleRemove()
    }
  }

  const getIcon = () => {
    switch (toast.type) {
      case 'success':
        return <CheckCircle className="w-5 h-5" />
      case 'error':
        return <AlertCircle className="w-5 h-5" />
      case 'warning':
        return <AlertTriangle className="w-5 h-5" />
      case 'info':
      default:
        return <AlertCircle className="w-5 h-5" />
    }
  }

  const getStyles = () => {
    const baseStyles = "border-l-4 shadow-lg backdrop-blur-sm"
    
    switch (toast.type) {
      case 'success':
        return `${baseStyles} bg-green-50/95 border-green-400 text-green-800`
      case 'error':
        return `${baseStyles} bg-red-50/95 border-red-400 text-red-800`
      case 'warning':
        return `${baseStyles} bg-yellow-50/95 border-yellow-400 text-yellow-800`
      case 'info':
      default:
        return `${baseStyles} bg-blue-50/95 border-blue-400 text-blue-800`
    }
  }

  return (
    <div
      className={clsx(
        'relative rounded-lg p-4 pr-12 transition-all duration-300 ease-out',
        'transform-gpu will-change-transform',
        getStyles(),
        isVisible && !isRemoving ? 'translate-x-0 opacity-100' : 'translate-x-full opacity-0',
        isRemoving && 'scale-95'
      )}
      style={{
        animationDelay: `${index * 100}ms`,
        transitionDelay: isVisible ? `${index * 50}ms` : '0ms'
      }}
    >
      {/* Close button */}
      <button
        onClick={handleRemove}
        className="absolute top-2 right-2 p-1 rounded-full hover:bg-black/10 transition-colors"
        aria-label="Fermer"
      >
        <X className="w-4 h-4" />
      </button>

      {/* Content */}
      <div className="flex items-start space-x-3">
        {/* Icon */}
        <div className="flex-shrink-0 mt-0.5">
          {getIcon()}
        </div>
        
        {/* Text content */}
        <div className="flex-1 min-w-0">
          <div className="font-semibold text-sm">
            {toast.title}
          </div>
          {toast.message && (
            <div className="text-sm opacity-90 mt-1">
              {toast.message}
            </div>
          )}
          
          {/* Actions */}
          <div className="flex items-center space-x-3 mt-3">
            {toast.onRetry && (
              <button
                onClick={handleRetry}
                className="inline-flex items-center space-x-1 text-xs font-medium hover:underline"
              >
                <RefreshCw className="w-3 h-3" />
                <span>Retry</span>
              </button>
            )}
            
            {toast.action && (
              <button
                onClick={() => {
                  toast.action!.onClick()
                  handleRemove()
                }}
                className="text-xs font-medium hover:underline"
              >
                {toast.action.label}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

// Utility functions pour usage rapide
export const useToastHelpers = () => {
  const { addToast } = useToast()
  
  return {
    success: (title: string, message?: string) => 
      addToast({ type: 'success', title, message }),
    
    error: (title: string, message?: string, onRetry?: () => void) => 
      addToast({ type: 'error', title, message, onRetry }),
    
    warning: (title: string, message?: string) => 
      addToast({ type: 'warning', title, message }),
    
    info: (title: string, message?: string) => 
      addToast({ type: 'info', title, message }),
      
    // Helper spécifique pour erreurs API avec retry
    apiError: (operation: string, onRetry?: () => void) => 
      addToast({
        type: 'error',
        title: 'API Error',
        message: `Failed to ${operation}. Please try again.`,
        onRetry,
        duration: 0 // Pas d'auto-dismiss pour les erreurs importantes
      }),
      
    // Helper pour succès API
    apiSuccess: (operation: string) => 
      addToast({
        type: 'success',
        title: 'Success',
        message: `${operation} completed successfully.`,
        duration: 3000
      })
  }
}