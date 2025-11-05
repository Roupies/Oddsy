'use client'

import { useState, useEffect, useCallback } from 'react'

interface VideoSource {
  type: 'webm' | 'mp4' | 'poster'
  url: string
  quality: '4k' | '1080p' | '720p' | 'poster'
}

interface DeviceInfo {
  isMobile: boolean
  isTablet: boolean
  isDesktop: boolean
  connectionType: 'slow' | 'fast' | 'wifi'
  prefersReducedMotion: boolean
  isLowPowerMode: boolean
}

export function useSmartHeroVideo() {
  const [videoSource, setVideoSource] = useState<VideoSource | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [hasError, setHasError] = useState(false)
  const [deviceInfo, setDeviceInfo] = useState<DeviceInfo | null>(null)

  // Device and connection detection
  const detectDevice = useCallback((): DeviceInfo => {
    const userAgent = navigator.userAgent
    const isMobile = /iPhone|iPad|iPod|Android/i.test(userAgent)
    const isTablet = /iPad|Android(?=.*Tablet)/i.test(userAgent)
    const isDesktop = !isMobile && !isTablet
    
    // Connection detection
    const connection = (navigator as any).connection || (navigator as any).mozConnection || (navigator as any).webkitConnection
    let connectionType: 'slow' | 'fast' | 'wifi' = 'fast'
    
    if (connection) {
      if (connection.effectiveType === '2g' || connection.effectiveType === 'slow-2g') {
        connectionType = 'slow'
      } else if (connection.effectiveType === '4g' || connection.type === 'wifi') {
        connectionType = 'wifi'
      }
    }

    // Reduced motion preference
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches

    // Low power mode detection (approximation for mobile)
    const isLowPowerMode = isMobile && (
      (navigator as any).getBattery?.then?.((battery: any) => battery.level < 0.2) || 
      prefersReducedMotion
    )

    return {
      isMobile,
      isTablet,
      isDesktop,
      connectionType,
      prefersReducedMotion,
      isLowPowerMode: Boolean(isLowPowerMode)
    }
  }, [])

  // Smart source selection based on device capabilities
  const selectOptimalSource = useCallback((device: DeviceInfo): VideoSource => {
    // Respect reduced motion preference
    if (device.prefersReducedMotion) {
      return {
        type: 'poster',
        url: '/videos/oddsy-hero-poster.svg',
        quality: 'poster'
      }
    }

    // Mobile strategy - utiliser la vraie vidéo mobile
    if (device.isMobile) {
      if (device.connectionType === 'slow' || device.isLowPowerMode) {
        return {
          type: 'poster',
          url: '/videos/oddsy-hero-poster.svg',
          quality: 'poster'
        }
      }
      return {
        type: 'mp4',
        url: '/videos/oddsy-bg-1080p-mobile.mp4',
        quality: '1080p'
      }
    }

    // Desktop strategy - utiliser la vraie vidéo WebM
    if (device.isDesktop) {
      // Prefer WebM for better compression on modern browsers
      if (canPlayWebM()) {
        return {
          type: 'webm',
          url: '/videos/oddsy-bg-1080p.webm',
          quality: '1080p'
        }
      }
      
      // Fallback to MP4
      return {
        type: 'mp4',
        url: '/videos/oddsy-bg-1080p-mobile.mp4',
        quality: '1080p'
      }
    }

    // Tablet fallback
    return {
      type: 'mp4',
      url: '/videos/oddsy-bg-1080p-mobile.mp4',
      quality: '1080p'
    }
  }, [])

  // WebM support detection
  const canPlayWebM = useCallback((): boolean => {
    const video = document.createElement('video')
    return video.canPlayType('video/webm') !== ''
  }, [])

  // Preload video with error handling
  const preloadVideo = useCallback(async (source: VideoSource): Promise<boolean> => {
    if (source.type === 'poster') return true

    return new Promise((resolve) => {
      const video = document.createElement('video')
      video.preload = 'metadata'
      video.muted = true
      
      const onLoad = () => {
        cleanup()
        resolve(true)
      }
      
      const onError = () => {
        cleanup()
        resolve(false)
      }
      
      const cleanup = () => {
        video.removeEventListener('loadedmetadata', onLoad)
        video.removeEventListener('error', onError)
      }
      
      video.addEventListener('loadedmetadata', onLoad)
      video.addEventListener('error', onError)
      video.src = source.url
      
      // Timeout after 5 seconds
      setTimeout(() => {
        cleanup()
        resolve(false)
      }, 5000)
    })
  }, [])

  // Initialize video source
  useEffect(() => {
    const initializeVideo = async () => {
      try {
        setIsLoading(true)
        const device = detectDevice()
        setDeviceInfo(device)
        
        const optimalSource = selectOptimalSource(device)
        
        // Try to preload the optimal source
        const canLoad = await preloadVideo(optimalSource)
        
        if (canLoad) {
          setVideoSource(optimalSource)
        } else {
          // Fallback to poster
          setVideoSource({
            type: 'poster',
            url: '/videos/oddsy-hero-poster.jpg',
            quality: 'poster'
          })
        }
        
        setHasError(false)
      } catch (error) {
        console.error('Video loading error:', error)
        setHasError(true)
        setVideoSource({
          type: 'poster',
          url: '/videos/oddsy-hero-poster.jpg',
          quality: 'poster'
        })
      } finally {
        setIsLoading(false)
      }
    }

    initializeVideo()
  }, [detectDevice, selectOptimalSource, preloadVideo])

  // Handle connection changes
  useEffect(() => {
    const handleConnectionChange = () => {
      if (deviceInfo) {
        const updatedDevice = detectDevice()
        const newSource = selectOptimalSource(updatedDevice)
        
        // Only update if source quality should change
        if (videoSource && newSource.quality !== videoSource.quality) {
          setVideoSource(newSource)
        }
      }
    }

    const connection = (navigator as any).connection
    if (connection) {
      connection.addEventListener('change', handleConnectionChange)
      return () => connection.removeEventListener('change', handleConnectionChange)
    }
  }, [deviceInfo, videoSource, detectDevice, selectOptimalSource])

  return {
    videoSource,
    isLoading,
    hasError,
    deviceInfo,
    // Utility functions for component
    canPlayVideo: videoSource?.type !== 'poster',
    shouldAutoplay: deviceInfo?.prefersReducedMotion === false,
    isOptimized: !hasError && !isLoading
  }
}