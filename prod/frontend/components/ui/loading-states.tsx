'use client'

import React from 'react'
import { cn } from '@/lib/utils'

interface LoadingSpinnerProps {
  className?: string
  size?: 'sm' | 'default' | 'lg' | 'xl'
  color?: 'primary' | 'secondary' | 'white' | 'gray'
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ 
  className, 
  size = 'default',
  color = 'primary'
}) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    default: 'w-6 h-6',
    lg: 'w-8 h-8',
    xl: 'w-12 h-12'
  }

  const colorClasses = {
    primary: 'border-oddsy-primary',
    secondary: 'border-oddsy-secondary',
    white: 'border-white',
    gray: 'border-gray-400'
  }

  return (
    <div
      className={cn(
        'animate-spin rounded-full border-2 border-transparent',
        sizeClasses[size],
        `${colorClasses[color]} border-t-current`,
        className
      )}
      role="status"
      aria-label="Loading"
    >
      <span className="sr-only">Loading...</span>
    </div>
  )
}

interface SkeletonProps {
  className?: string
  variant?: 'text' | 'rectangular' | 'circular' | 'rounded'
  animation?: 'pulse' | 'wave' | 'breathing'
}

export const Skeleton: React.FC<SkeletonProps> = ({ 
  className, 
  variant = 'rectangular',
  animation = 'pulse'
}) => {
  const variantClasses = {
    text: 'h-4 rounded',
    rectangular: 'rounded-lg',
    circular: 'rounded-full',
    rounded: 'rounded-xl'
  }

  const animationClasses = {
    pulse: 'animate-pulse',
    wave: 'animate-pulse',
    breathing: 'animate-pulse-slow'
  }

  return (
    <div
      className={cn(
        'bg-gradient-to-r from-gray-200 via-gray-300 to-gray-200 bg-[length:200%_100%] animate-pulse',
        variantClasses[variant],
        animationClasses[animation],
        className
      )}
    />
  )
}

interface LoadingCardProps {
  className?: string
  showAvatar?: boolean
  lines?: number
}

export const LoadingCard: React.FC<LoadingCardProps> = ({ 
  className, 
  showAvatar = false,
  lines = 3
}) => {
  return (
    <div className={cn('bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20', className)}>
      <div className="animate-pulse">
        {showAvatar && (
          <div className="flex items-center space-x-4 mb-4">
            <Skeleton variant="circular" className="w-12 h-12" />
            <div className="flex-1">
              <Skeleton className="h-4 w-3/4 mb-2" />
              <Skeleton className="h-3 w-1/2" />
            </div>
          </div>
        )}
        
        <div className="space-y-3">
          {Array.from({ length: lines }).map((_, i) => (
            <Skeleton 
              key={i} 
              className={cn(
                'h-4',
                i === lines - 1 ? 'w-2/3' : 'w-full'
              )} 
            />
          ))}
        </div>
      </div>
    </div>
  )
}

interface LoadingDotsProps {
  className?: string
  size?: 'sm' | 'default' | 'lg'
  color?: 'primary' | 'secondary' | 'white'
}

export const LoadingDots: React.FC<LoadingDotsProps> = ({ 
  className, 
  size = 'default',
  color = 'primary'
}) => {
  const sizeClasses = {
    sm: 'w-1 h-1',
    default: 'w-2 h-2',
    lg: 'w-3 h-3'
  }

  const colorClasses = {
    primary: 'bg-oddsy-primary',
    secondary: 'bg-oddsy-secondary',
    white: 'bg-white'
  }

  return (
    <div className={cn('flex space-x-1', className)}>
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className={cn(
            'rounded-full animate-bounce',
            sizeClasses[size],
            colorClasses[color]
          )}
          style={{
            animationDelay: `${i * 0.15}s`,
            animationDuration: '0.6s'
          }}
        />
      ))}
    </div>
  )
}

interface LoadingPulseProps {
  className?: string
  children?: React.ReactNode
}

export const LoadingPulse: React.FC<LoadingPulseProps> = ({ className, children }) => {
  return (
    <div className={cn('relative', className)}>
      {children}
      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-pulse" />
    </div>
  )
}

interface LoadingOverlayProps {
  className?: string
  message?: string
  variant?: 'spinner' | 'dots' | 'pulse'
}

export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({ 
  className, 
  message = 'Loading...',
  variant = 'spinner'
}) => {
  return (
    <div className={cn(
      'absolute inset-0 bg-white/10 backdrop-blur-sm flex items-center justify-center z-50',
      className
    )}>
      <div className="text-center">
        {variant === 'spinner' && <LoadingSpinner size="lg" color="white" className="mx-auto mb-4" />}
        {variant === 'dots' && <LoadingDots size="lg" color="white" className="justify-center mb-4" />}
        {variant === 'pulse' && (
          <div className="w-16 h-16 bg-white/20 rounded-full animate-pulse-slow mx-auto mb-4" />
        )}
        <p className="text-white font-medium">{message}</p>
      </div>
    </div>
  )
}