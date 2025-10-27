'use client'

import React, { useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import { motion, useScroll, useTransform } from 'framer-motion'

export const HeroSplitPremium = () => {
  const videoRef = useRef<HTMLVideoElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [isMounted, setIsMounted] = useState(false)
  
  const { scrollYProgress } = useScroll({
    target: isMounted ? containerRef : undefined,
    offset: ["start start", "end start"]
  })

  const videoScale = useTransform(scrollYProgress, [0, 1], [1, 1.05])
  const contentY = useTransform(scrollYProgress, [0, 1], [0, -50])

  useEffect(() => {
    setIsMounted(true)
  }, [])

  if (!isMounted) {
    return (
      <section className="h-screen bg-neutral-950 flex items-center justify-center">
        <div className="text-neutral-100 text-sm font-mono">Loading...</div>
      </section>
    )
  }

  return (
    <section 
      ref={containerRef}
      className="relative h-screen overflow-hidden bg-neutral-950 z-10"
    >
      <div className="flex flex-col md:flex-row h-full">
        
        <div className="relative w-full md:w-[60%] h-[45vh] md:h-full overflow-hidden">
          <motion.div 
            className="absolute inset-0 w-full h-full overflow-hidden"
            style={isMounted ? { scale: videoScale, transformOrigin: 'center top' } : {}}
          >
            <div 
              className="absolute inset-0 w-full h-full bg-cover bg-center bg-no-repeat"
              style={{ backgroundImage: `url('/videos/oddsy-hero-poster.svg')` }}
            />
            
            <video
              ref={videoRef}
              className="absolute inset-0 w-full h-full object-cover opacity-0"
              autoPlay
              muted
              loop
              playsInline
              poster="/videos/oddsy-hero-poster.svg"
              onLoadedData={() => {
                if (videoRef.current) {
                  videoRef.current.style.opacity = '1'
                }
              }}
            >
              <source src="/videos/oddsy-bg-1080p.webm" type="video/webm" />
              <source src="/videos/oddsy-bg-1080p-mobile.mp4" type="video/mp4" />
            </video>
          </motion.div>

          <div className="absolute inset-y-0 right-0 w-32 md:w-48 bg-gradient-to-l from-neutral-950 via-neutral-950/50 to-transparent pointer-events-none" />
          
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.2, duration: 0.6 }}
            className="absolute bottom-8 left-8 hidden md:block"
          >
            <div className="inline-flex items-center gap-3 bg-black/40 backdrop-blur-sm border border-white/10 rounded-full px-4 py-2">
              <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
              <span className="text-white text-sm font-medium tracking-tight">
                Live Predictions
              </span>
            </div>
          </motion.div>
        </div>

        <motion.div 
          className="flex-1 bg-neutral-950 text-neutral-100 p-8 md:p-16 lg:p-20 flex items-center justify-center"
          style={isMounted ? { y: contentY } : {}}
        >
          <div className="max-w-xl w-full">
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="mb-8 md:mb-12"
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="w-1 h-8 bg-emerald-500" />
                <span className="text-neutral-400 text-xs font-mono uppercase tracking-wider">
                  Prediction Engine
                </span>
              </div>
              <h1 className="text-[clamp(3rem,8vw,6rem)] leading-[0.9] font-black tracking-tighter">
                ODDSY
              </h1>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2, duration: 0.6 }}
              className="mb-8 md:mb-12"
            >
              <h2 className="text-[clamp(2rem,4.5vw,3.5rem)] leading-[1.1] font-bold tracking-tight mb-4">
                Predict the
                <br />
                <span className="relative inline-block">
                  Premier League
                  <span className="absolute -bottom-2 left-0 w-full h-1 bg-emerald-500" />
                </span>
              </h2>
              <p className="text-neutral-400 text-lg md:text-xl leading-relaxed max-w-md">
                Machine learning models trained on real match data. No guessing, just numbers.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4, duration: 0.6 }}
              className="grid grid-cols-2 gap-4 md:gap-6 mb-10 md:mb-14 max-w-lg"
            >
              <StatCard 
                value="51.7%" 
                label="Real Match Accuracy" 
                trend="+2.3%"
                delay={0.5}
              />
              <StatCard 
                value="840+" 
                label="Predictions Made" 
                trend="Season 24/25"
                delay={0.6}
              />
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7, duration: 0.6 }}
              className="flex flex-col sm:flex-row gap-4"
            >
              <Link href="/predictions/latest">
                <button className="group relative overflow-hidden bg-emerald-500 hover:bg-emerald-400 text-neutral-950 font-bold px-8 py-4 transition-all duration-300 hover:-translate-y-0.5 w-full">
                  <span className="relative z-10 flex items-center gap-2 justify-center">
                    View Latest Predictions
                    <svg className="w-5 h-5 transition-transform group-hover:translate-x-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                    </svg>
                  </span>
                </button>
              </Link>

              <Link href="/pipeline">
                <button className="group border border-neutral-700 hover:border-neutral-500 bg-transparent text-neutral-100 font-semibold px-8 py-4 transition-all duration-300 hover:-translate-y-0.5 w-full">
                  <span className="flex items-center gap-2 justify-center">
                    Pipeline Status
                    <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
                  </span>
                </button>
              </Link>
            </motion.div>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1, duration: 0.6 }}
              className="mt-10 md:mt-14 pt-8 border-t border-neutral-800"
            >
              <p className="text-neutral-500 text-sm font-mono">
                Validated on 70 real EPL matches • Season 2024/25
              </p>
            </motion.div>
          </div>
        </motion.div>
      </div>
    </section>
  )
}

const StatCard = ({ 
  value, 
  label, 
  trend, 
  delay 
}: { 
  value: string
  label: string
  trend?: string
  delay: number 
}) => {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay, duration: 0.5 }}
      className="group relative overflow-hidden bg-black/60 border border-neutral-800 hover:border-emerald-500/50 p-6 transition-all duration-300 hover:-translate-y-1 backdrop-blur-md"
    >
      <div className="absolute top-0 left-0 w-full h-0.5 bg-gradient-to-r from-emerald-500 to-transparent" />
      
      <div className="text-[clamp(2rem,3.5vw,2.5rem)] font-black text-neutral-100 mb-2 leading-none">
        {value}
      </div>
      <div className="text-xs text-neutral-400 font-medium uppercase tracking-wider mb-3">
        {label}
      </div>
      {trend && (
        <div className="text-xs text-emerald-500 font-mono">
          {trend}
        </div>
      )}

      <div className="absolute inset-0 bg-gradient-to-tr from-emerald-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none" />
    </motion.div>
  )
}