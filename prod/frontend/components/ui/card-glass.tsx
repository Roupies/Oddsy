'use client'

import React from 'react'
import { cn } from '@/lib/utils'

interface CardGlassProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'glass' | 'premium' | 'epl' | 'solid'
  size?: 'sm' | 'default' | 'lg' | 'xl'
  hover?: 'none' | 'lift' | 'glow' | 'scale' | 'premium'
  children: React.ReactNode
  glow?: boolean
  shimmer?: boolean
}

const getVariantClasses = (variant: string = 'default') => {
  const variants = {
    default: 'bg-white/10 backdrop-blur-xl border border-white/20 hover:bg-white/15 hover:border-white/30',
    glass: 'bg-white/5 backdrop-blur-2xl border border-white/10 hover:bg-white/10 hover:border-white/20',
    premium: 'bg-gradient-to-br from-white/20 via-white/10 to-white/5 backdrop-blur-2xl border border-white/30 hover:from-white/30 hover:to-white/10',
    epl: 'bg-gradient-to-br from-oddsy-primary/20 via-oddsy-secondary/10 to-oddsy-accent/5 backdrop-blur-xl border border-oddsy-secondary/30 hover:border-oddsy-secondary/50',
    solid: 'bg-white border border-gray-200 hover:border-gray-300 shadow-sm hover:shadow-md'
  }
  return variants[variant as keyof typeof variants] || variants.default
}

const getSizeClasses = (size: string = 'default') => {
  const sizes = {
    sm: 'p-4 rounded-lg',
    default: 'p-6 rounded-xl',
    lg: 'p-8 rounded-2xl',
    xl: 'p-10 rounded-3xl'
  }
  return sizes[size as keyof typeof sizes] || sizes.default
}

const getHoverClasses = (hover: string = 'lift') => {
  const hovers = {
    none: '',
    lift: 'hover:-translate-y-1 hover:shadow-2xl',
    glow: 'hover:shadow-2xl hover:shadow-oddsy-secondary/20',
    scale: 'hover:scale-105',
    premium: 'hover:-translate-y-2 hover:shadow-3xl hover:shadow-oddsy-primary/30'
  }
  return hovers[hover as keyof typeof hovers] || hovers.lift
}

const CardGlass = React.forwardRef<HTMLDivElement, CardGlassProps>(
  ({ className, variant = 'default', size = 'default', hover = 'lift', glow, shimmer, children, ...props }, ref) => {
    const baseClasses = 'relative overflow-hidden transition-all duration-300 group'
    
    return (
      <div
        ref={ref}
        className={cn(baseClasses, getVariantClasses(variant), getSizeClasses(size), getHoverClasses(hover), className)}
        {...props}
      >
        {/* Shimmer Effect */}
        {shimmer && (
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000" />
        )}
        
        {/* Glow Effect */}
        {glow && (
          <div className="absolute inset-0 bg-gradient-to-r from-oddsy-secondary/0 via-oddsy-secondary/20 to-oddsy-accent/0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 blur-xl" />
        )}
        
        {/* EPL Corner Accent */}
        {variant === 'epl' && (
          <div className="absolute top-0 right-0 w-20 h-20 bg-gradient-to-bl from-oddsy-secondary/30 to-transparent rounded-bl-full" />
        )}
        
        {/* Content */}
        <div className="relative z-10">
          {children}
        </div>
      </div>
    )
  }
)

CardGlass.displayName = 'CardGlass'

export { CardGlass }