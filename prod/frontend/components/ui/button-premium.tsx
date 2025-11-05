'use client'

import React from 'react'
import { cn } from '@/lib/utils'

interface ButtonPremiumProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'epl' | 'glass' | 'accent'
  size?: 'sm' | 'default' | 'lg' | 'xl'
  loading?: boolean
  icon?: React.ReactNode
  rightIcon?: React.ReactNode
}

const getVariantClasses = (variant: string = 'primary') => {
  const variants = {
    primary: 'bg-gradient-to-r from-oddsy-secondary to-oddsy-accent hover:from-oddsy-accent hover:to-oddsy-secondary text-oddsy-primary shadow-lg hover:shadow-2xl transform hover:scale-105',
    secondary: 'border-2 border-white/30 hover:border-oddsy-secondary bg-white/10 hover:bg-oddsy-secondary/20 backdrop-blur-sm text-white hover:shadow-xl',
    ghost: 'bg-transparent hover:bg-white/10 text-white border border-transparent hover:border-white/20',
    epl: 'bg-gradient-to-r from-epl-purple via-epl-green to-epl-pink hover:shadow-2xl text-white transform hover:scale-105',
    glass: 'bg-white/10 backdrop-blur-2xl border border-white/20 text-white hover:bg-white/20 hover:border-white/40 shadow-3xl',
    accent: 'bg-oddsy-accent hover:bg-oddsy-accent/90 text-white shadow-lg hover:shadow-xl transform hover:scale-105'
  }
  return variants[variant as keyof typeof variants] || variants.primary
}

const getSizeClasses = (size: string = 'default') => {
  const sizes = {
    sm: 'h-9 px-4 py-2 text-sm',
    default: 'h-12 px-6 py-3 text-base',
    lg: 'h-14 px-8 py-4 text-lg',
    xl: 'h-16 px-10 py-5 text-xl'
  }
  return sizes[size as keyof typeof sizes] || sizes.default
}

const ButtonPremium = React.forwardRef<HTMLButtonElement, ButtonPremiumProps>(
  ({ className, variant = 'primary', size = 'default', loading, icon, rightIcon, children, ...props }, ref) => {
    const baseClasses = 'inline-flex items-center justify-center rounded-xl font-semibold transition-all duration-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 relative overflow-hidden group'
    
    return (
      <button
        className={cn(baseClasses, getVariantClasses(variant), getSizeClasses(size), className)}
        ref={ref}
        disabled={loading || props.disabled}
        {...props}
      >
        {/* Shimmer Effect for Primary/EPL variants */}
        {(variant === 'primary' || variant === 'epl') && (
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700" />
        )}
        
        {/* Loading Spinner */}
        {loading && (
          <svg className="animate-spin -ml-1 mr-3 h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
        )}
        
        {/* Left Icon */}
        {icon && !loading && (
          <span className="mr-2 flex-shrink-0">
            {icon}
          </span>
        )}
        
        {/* Content */}
        <span className="relative z-10 flex items-center">
          {children}
        </span>
        
        {/* Right Icon */}
        {rightIcon && !loading && (
          <span className="ml-2 flex-shrink-0 group-hover:translate-x-1 transition-transform duration-200">
            {rightIcon}
          </span>
        )}
      </button>
    )
  }
)

ButtonPremium.displayName = 'ButtonPremium'

export { ButtonPremium }