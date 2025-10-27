'use client'

import { clsx } from 'clsx'

interface SkeletonProps {
  className?: string
  width?: string
  height?: string
}

export function Skeleton({ className, width, height }: SkeletonProps) {
  return (
    <div 
      className={clsx(
        'animate-pulse bg-gray-200 rounded',
        className
      )}
      style={{ width, height }}
    />
  )
}

interface MatchCardSkeletonProps {
  compact?: boolean
  className?: string
}

export function MatchCardSkeleton({ compact = false, className }: MatchCardSkeletonProps) {
  return (
    <div className={clsx(
      'prediction-card overflow-hidden relative',
      'bg-white rounded-lg shadow-sm border border-gray-100',
      compact ? 'p-4' : 'p-6',
      className
    )}>
      {/* Ko2h Badge skeleton */}
      <div className="absolute top-4 right-4">
        <Skeleton className="w-20 h-6 rounded-full" />
      </div>
      
      {/* Header avec logos skeleton */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-4">
          {/* Home team */}
          <div className="flex items-center space-x-3">
            <Skeleton className="w-12 h-12 rounded-full" /> {/* Logo */}
            <div>
              <Skeleton className="w-24 h-5 mb-1" /> {/* Nom équipe */}
              <Skeleton className="w-12 h-3" />      {/* HOME label */}
            </div>
          </div>
          
          {/* VS section */}
          <div className="flex flex-col items-center mx-4">
            <Skeleton className="w-8 h-6 mb-1" />    {/* VS */}
            <Skeleton className="w-20 h-3" />        {/* Date */}
          </div>
          
          {/* Away team */}
          <div className="flex items-center space-x-3">
            <div>
              <Skeleton className="w-24 h-5 mb-1" /> {/* Nom équipe */}
              <Skeleton className="w-12 h-3" />      {/* AWAY label */}
            </div>
            <Skeleton className="w-12 h-12 rounded-full" /> {/* Logo */}
          </div>
        </div>
      </div>
      
      {/* Odds info skeleton */}
      <div className="flex items-center gap-2 mb-4">
        <Skeleton className="w-16 h-6 rounded-full" />
        <Skeleton className="w-20 h-6 rounded-full" />
      </div>
      
      {/* Ensemble prediction skeleton */}
      <div className="p-6 bg-gray-50 rounded-lg mb-4">
        <div className="flex items-center space-x-3 mb-4">
          <Skeleton className="w-8 h-8 rounded" />    {/* Icon */}
          <div>
            <Skeleton className="w-16 h-6 mb-1" />    {/* Label */}
            <Skeleton className="w-32 h-4" />         {/* Model name */}
          </div>
        </div>
        
        {/* Confidence gauge skeleton */}
        <div className="space-y-2">
          <div className="flex justify-between">
            <Skeleton className="w-12 h-5" />         {/* Percentage */}
            <Skeleton className="w-16 h-6 rounded-full" /> {/* Badge */}
          </div>
          <Skeleton className="w-full h-3 rounded-full" /> {/* Progress bar */}
          <div className="flex justify-between">
            <Skeleton className="w-6 h-3" />
            <Skeleton className="w-10 h-3" />
            <Skeleton className="w-8 h-3" />
          </div>
        </div>
      </div>
      
      {/* Probabilities skeleton */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        {[1, 2, 3].map((i) => (
          <div key={i} className="text-center p-4 bg-gray-50 rounded-lg">
            <Skeleton className="w-8 h-8 mx-auto mb-2" />    {/* Icon */}
            <Skeleton className="w-12 h-5 mx-auto mb-1" />   {/* Percentage */}
            <Skeleton className="w-8 h-3 mx-auto mb-2" />    {/* Label */}
            <Skeleton className="w-full h-1 rounded-full" />  {/* Progress bar */}
          </div>
        ))}
      </div>
      
      {/* Model info skeleton */}
      {!compact && (
        <div className="pt-4 border-t border-gray-100">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Skeleton className="w-20 h-4" />
              <Skeleton className="w-24 h-5 rounded-full" />
            </div>
            <Skeleton className="w-16 h-4" />
          </div>
        </div>
      )}
    </div>
  )
}

interface MatchGridSkeletonProps {
  count?: number
  compact?: boolean
  className?: string
}

export function MatchGridSkeleton({ 
  count = 10, 
  compact = false, 
  className 
}: MatchGridSkeletonProps) {
  return (
    <div className={clsx(
      'grid grid-cols-1 lg:grid-cols-2 gap-6',
      className
    )}>
      {Array.from({ length: count }).map((_, index) => (
        <div
          key={index}
          style={{ 
            animationDelay: `${index * 50}ms`,
            animationDuration: '1.5s'
          }}
          className="animate-fade-in"
        >
          <MatchCardSkeleton compact={compact} />
        </div>
      ))}
    </div>
  )
}

interface StatsSkeletonProps {
  className?: string
}

export function StatsCardsSkeleton({ className }: StatsSkeletonProps) {
  return (
    <div className={clsx('grid grid-cols-1 md:grid-cols-4 gap-4', className)}>
      {[1, 2, 3, 4].map((i) => (
        <div key={i} className="p-4 bg-white rounded-lg border border-gray-100">
          <Skeleton className="w-8 h-8 mb-2" />      {/* Number */}
          <Skeleton className="w-16 h-4" />          {/* Label */}
        </div>
      ))}
    </div>
  )
}

interface HeaderSkeletonProps {
  className?: string
}

export function HeaderSkeleton({ className }: HeaderSkeletonProps) {
  return (
    <div className={clsx('mb-8', className)}>
      <div className="flex items-center justify-between mb-4">
        <Skeleton className="w-48 h-8" />            {/* Title */}
        <Skeleton className="w-20 h-6" />            {/* Match count */}
      </div>
      
      <Skeleton className="w-64 h-5 mb-6" />        {/* Description */}
      
      <div className="flex flex-wrap gap-4">
        {[1, 2, 3].map((i) => (
          <Skeleton key={i} className="w-32 h-6 rounded-full" /> /* Badges */
        ))}
      </div>
    </div>
  )
}

interface LoadingPageSkeletonProps {
  className?: string
}

export function LoadingPageSkeleton({ className }: LoadingPageSkeletonProps) {
  return (
    <div className={clsx('container mx-auto px-4 py-8', className)}>
      {/* Header skeleton */}
      <HeaderSkeleton className="mb-8" />
      
      {/* Performance summary skeleton */}
      <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-lg p-6 mb-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[1, 2, 3].map((i) => (
            <div key={i} className="text-center">
              <Skeleton className="w-16 h-8 mx-auto mb-1" />
              <Skeleton className="w-24 h-4 mx-auto" />
            </div>
          ))}
        </div>
      </div>
      
      {/* Stats cards skeleton */}
      <StatsCardsSkeleton className="mb-8" />
      
      {/* Match grid skeleton */}
      <MatchGridSkeleton count={10} />
    </div>
  )
}