'use client'

import React, { useState, useEffect, useRef } from 'react'
import { useSmartHeroVideo } from '@/hooks/useSmartHeroVideo'

interface HeroCinematicProps {
  className?: string
}

export const HeroCinematic: React.FC<HeroCinematicProps> = ({ 
  className = ''
}) => {
  const { videoSource, isLoading, deviceInfo, canPlayVideo, shouldAutoplay } = useSmartHeroVideo()
  const videoRef = useRef<HTMLVideoElement>(null)
  const [isMounted, setIsMounted] = useState(false)
  const [isVideoReady, setIsVideoReady] = useState(false)
  const [isPaused, setIsPaused] = useState(false)

  useEffect(() => {
    setIsMounted(true)
  }, [])

  // Handle video ready state
  useEffect(() => {
    if (videoRef.current && canPlayVideo && videoSource?.type !== 'poster') {
      const video = videoRef.current
      
      const handleCanPlay = () => {
        console.log('✅ Video can play')
        setIsVideoReady(true)
      }
      
      const handleError = (e: Event) => {
        console.error('❌ Video error:', e)
        setIsVideoReady(false)
      }
      
      video.addEventListener('canplay', handleCanPlay)
      video.addEventListener('error', handleError)
      
      return () => {
        video.removeEventListener('canplay', handleCanPlay)
        video.removeEventListener('error', handleError)
      }
    }
  }, [canPlayVideo, videoSource])

  // Auto-play when ready
  useEffect(() => {
    if (videoRef.current && isVideoReady && shouldAutoplay && !isPaused) {
      videoRef.current.play().catch((error) => {
        console.log('⚠️ Autoplay prevented:', error)
      })
    }
  }, [isVideoReady, shouldAutoplay, isPaused])

  const toggleVideo = () => {
    if (videoRef.current && isVideoReady) {
      if (isPaused) {
        videoRef.current.play()
        setIsPaused(false)
      } else {
        videoRef.current.pause()
        setIsPaused(true)
      }
    }
  }

  // État de chargement initial (hydration)
  if (!isMounted) {
    return (
      <section className={`relative min-h-screen flex items-center justify-center overflow-hidden ${className}`}>
        <div 
          className="absolute inset-0 w-full h-full bg-cover bg-center bg-no-repeat"
          style={{ backgroundImage: `url('/videos/oddsy-hero-poster.svg')` }}
        />
        <div className="absolute inset-0 bg-gradient-to-br from-oddsy-primary/90 via-oddsy-primary/70 to-transparent" />
        <div className="relative z-10 container mx-auto px-4 text-center text-white">
          <div className="animate-pulse">
            <div className="h-16 bg-white/20 rounded mb-6 mx-auto max-w-2xl" />
            <div className="h-8 bg-white/20 rounded mb-8 mx-auto max-w-lg" />
            <div className="h-12 bg-white/20 rounded mx-auto w-48" />
          </div>
        </div>
      </section>
    )
  }

  return (
    <section className={`relative min-h-screen flex items-center justify-center overflow-hidden ${className}`}>
      {/* Video Background */}
      {canPlayVideo && videoSource && videoSource.type !== 'poster' && (
        <video
          ref={videoRef}
          className="absolute inset-0 w-full h-full object-cover scale-105"
          autoPlay={shouldAutoplay}
          muted
          loop
          playsInline
          poster="/videos/oddsy-hero-poster.svg"
          src={videoSource.url}
        />
      )}

      {/* Fallback Background Image */}
      {(!canPlayVideo || !videoSource || videoSource.type === 'poster') && (
        <div 
          className="absolute inset-0 w-full h-full bg-cover bg-center bg-no-repeat scale-105 animate-float"
          style={{ backgroundImage: `url('/videos/oddsy-hero-poster.svg')` }}
        />
      )}

      {/* Dynamic Gradient Overlay */}
      <div className="absolute inset-0 bg-gradient-to-br from-oddsy-primary/80 via-oddsy-primary/60 to-transparent" />
      
      {/* Animated Particles */}
      {!deviceInfo?.prefersReducedMotion && (
        <div className="absolute inset-0 opacity-30">
          <div className="absolute top-1/4 left-1/4 w-2 h-2 bg-oddsy-secondary rounded-full animate-pulse-slow" />
          <div className="absolute top-1/3 right-1/3 w-1 h-1 bg-oddsy-accent rounded-full animate-pulse delay-500" />
          <div className="absolute bottom-1/3 left-1/3 w-3 h-3 bg-oddsy-secondary/50 rounded-full animate-pulse delay-1000" />
          <div className="absolute top-2/3 right-1/4 w-1 h-1 bg-oddsy-accent/70 rounded-full animate-pulse delay-1500" />
        </div>
      )}

      {/* Content Container */}
      <div className="relative z-10 container mx-auto px-4 text-center text-white">
        {/* Main Content */}
        {!isLoading && (
          <div className="animate-fadeUp">
            {/* Hero Badge */}
            <div className="inline-flex items-center space-x-2 bg-white/10 backdrop-blur-sm rounded-full px-6 py-3 mb-8 border border-white/20">
              <span className="w-2 h-2 bg-oddsy-secondary rounded-full animate-pulse" />
              <span className="text-sm font-medium">Pipeline Durci v1.0 Active</span>
              <span className="text-oddsy-secondary">✨</span>
            </div>

            {/* Main Headline */}
            <h1 className="text-5xl md:text-7xl lg:text-8xl font-bold mb-6 leading-tight">
              <span className="block text-white animate-slideUp">Predict the</span>
              <span className="block bg-gradient-to-r from-oddsy-secondary via-white to-oddsy-accent bg-clip-text text-transparent animate-slideUp delay-200">
                Premier League
              </span>
            </h1>

            {/* Subtitle */}
            <p className="text-xl md:text-2xl lg:text-3xl mb-8 text-white/90 max-w-3xl mx-auto font-light animate-slideUp delay-400">
              AI-powered predictions with{' '}
              <span className="text-oddsy-secondary font-semibold">53.5% accuracy</span>
              {' '}validated on real EPL data
            </p>

            {/* Trust Indicators */}
            <div className="flex flex-wrap justify-center gap-6 mb-12 animate-slideUp delay-600">
              <div className="flex items-center space-x-2 bg-white/10 backdrop-blur-sm rounded-lg px-4 py-2">
                <span className="text-oddsy-secondary">⚽</span>
                <span className="text-sm font-medium">840+ Predictions</span>
              </div>
              <div className="flex items-center space-x-2 bg-white/10 backdrop-blur-sm rounded-lg px-4 py-2">
                <span className="text-oddsy-secondary">🎯</span>
                <span className="text-sm font-medium">Real EPL Accuracy: 51.7%</span>
              </div>
              <div className="flex items-center space-x-2 bg-white/10 backdrop-blur-sm rounded-lg px-4 py-2">
                <span className="text-oddsy-secondary">🔬</span>
                <span className="text-sm font-medium">Validated on 60 Matches</span>
              </div>
            </div>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center animate-slideUp delay-800">
              <button className="group relative overflow-hidden bg-gradient-to-r from-oddsy-secondary to-oddsy-accent hover:from-oddsy-accent hover:to-oddsy-secondary px-8 py-4 rounded-xl font-semibold text-oddsy-primary transition-all duration-300 transform hover:scale-105 hover:shadow-2xl">
                <span className="relative z-10">View GW8 Predictions</span>
                <div className="absolute inset-0 bg-white/20 translate-x-full group-hover:translate-x-0 transition-transform duration-300" />
              </button>
              
              <button className="group border-2 border-white/30 hover:border-oddsy-secondary bg-white/10 hover:bg-oddsy-secondary/20 backdrop-blur-sm px-8 py-4 rounded-xl font-semibold text-white transition-all duration-300 hover:shadow-xl">
                <span className="flex items-center space-x-2">
                  <span>Pipeline Status</span>
                  <span className="text-oddsy-secondary group-hover:animate-pulse">📊</span>
                </span>
              </button>
            </div>

            {/* Scroll Indicator */}
            <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-bounce">
              <div className="w-6 h-10 border-2 border-white/50 rounded-full flex justify-center">
                <div className="w-1 h-3 bg-white/70 rounded-full mt-2 animate-pulse" />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Video Controls */}
      {isVideoReady && !deviceInfo?.prefersReducedMotion && (
        <button
          onClick={toggleVideo}
          className="absolute bottom-6 right-6 bg-black/50 hover:bg-black/70 backdrop-blur-sm text-white p-3 rounded-full transition-all duration-200 hover:scale-110 focus:outline-none focus:ring-2 focus:ring-oddsy-secondary z-20"
          aria-label={isPaused ? 'Play video' : 'Pause video'}
        >
          {isPaused ? (
            <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
            </svg>
          ) : (
            <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM7 8a1 1 0 012 0v4a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v4a1 1 0 002 0V8a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
          )}
        </button>
      )}

      {/* Debug Panel amélioré */}
      {process.env.NODE_ENV === 'development' && (
        <div className="absolute top-4 left-4 bg-black/90 text-white p-3 rounded-lg text-xs space-y-1 z-20 font-mono border border-white/20">
          <div className="font-bold text-oddsy-secondary mb-2">🎬 Video Debug</div>
          <div>Can Play: {canPlayVideo ? '✅' : '❌'}</div>
          <div>Video Ready: {isVideoReady ? '✅' : '❌'}</div>
          <div>Source: {videoSource?.quality || 'N/A'}</div>
          <div>Type: {videoSource?.type || 'N/A'}</div>
          <div>URL: {videoSource?.url ? '✅' : '❌'}</div>
          <div>Device: {deviceInfo?.isMobile ? '📱 Mobile' : '💻 Desktop'}</div>
          <div>Mounted: {isMounted ? '✅' : '❌'}</div>
        </div>
      )}
    </section>
  )
}
