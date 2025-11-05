'use client'

import React, { useEffect, useState } from 'react'
import Link from 'next/link'
import { motion, useScroll, useTransform } from 'framer-motion'

interface HeaderProps {
  className?: string
}

export function Header({ className = '' }: HeaderProps) {
  const [isMounted, setIsMounted] = useState(false)
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  
  const { scrollY } = useScroll()
  const headerBackground = useTransform(
    scrollY,
    [0, 100],
    ['rgba(0, 0, 0, 0.6)', 'rgba(0, 0, 0, 0.95)']
  )
  
  const headerBorder = useTransform(
    scrollY,
    [0, 100],
    ['rgba(255, 255, 255, 0.1)', 'rgba(255, 255, 255, 0.2)']
  )

  useEffect(() => {
    setIsMounted(true)
  }, [])

  if (!isMounted) {
    return (
      <header className={`fixed top-0 left-0 right-0 z-50 h-20 bg-black/60 ${className}`}>
        <div className="max-w-7xl mx-auto px-4 md:px-8 lg:px-16 h-full flex items-center justify-between">
          <div className="text-white text-2xl font-black">ODDSY</div>
        </div>
      </header>
    )
  }

  return (
    <motion.header 
      className={`fixed top-0 left-0 right-0 z-[100] transition-all duration-300 backdrop-blur-xl ${className}`}
      style={{ 
        backgroundColor: headerBackground,
        borderBottom: `1px solid ${headerBorder}`
      }}
    >
      <div className="max-w-7xl mx-auto px-4 md:px-8 lg:px-16">
        <div className="flex items-center justify-between h-20">
          
          {/* Logo - Editorial Style */}
          <Link href="/" className="group flex items-center gap-3">
            <div className="flex items-center gap-2">
              <div className="w-1 h-8 bg-emerald-500 transition-all duration-300 group-hover:h-10" />
              <div>
                <div className="text-white text-2xl font-black tracking-tight transition-colors group-hover:text-emerald-400">
                  ODDSY
                </div>
                <div className="text-neutral-400 text-xs font-mono uppercase tracking-wider">
                  Prediction Engine
                </div>
              </div>
            </div>
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center gap-8">
            <Link 
              href="/matchday" 
              className="text-neutral-300 hover:text-white transition-colors text-sm font-medium uppercase tracking-wide"
            >
              Predictions
            </Link>
            <a 
              href="#fortress" 
              className="text-neutral-300 hover:text-white transition-colors text-sm font-medium uppercase tracking-wide"
            >
              Fortress Analysis
            </a>
            <Link 
              href="/models" 
              className="text-neutral-300 hover:text-white transition-colors text-sm font-medium uppercase tracking-wide"
            >
              Our Models
            </Link>
            <Link 
              href="/pipeline" 
              className="text-neutral-300 hover:text-white transition-colors text-sm font-medium uppercase tracking-wide"
            >
              Pipeline
            </Link>
          </nav>

          {/* Status & Performance - Editorial Style */}
          <div className="hidden lg:flex items-center gap-4">
            {/* Pipeline Status */}
            <div className="flex items-center gap-2 bg-black/40 backdrop-blur-sm border border-white/10 rounded-full px-4 py-2">
              <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
              <span className="text-white text-sm font-medium">
                Pipeline Active
              </span>
            </div>
            
            {/* Performance Metric with Tooltip */}
            <div className="group relative">
              <div className="bg-black/40 backdrop-blur-sm border border-emerald-500/30 rounded px-3 py-1 cursor-help">
                <span className="text-emerald-400 text-sm font-mono font-bold">
                  51.3%
                </span>
                <span className="text-neutral-400 text-xs ml-1">
                  v3.0
                </span>
              </div>
              
              {/* Tooltip */}
              <div className="absolute top-full left-1/2 -translate-x-1/2 mt-2 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                <div className="bg-black/90 backdrop-blur-sm border border-white/20 rounded px-3 py-2 text-xs text-white whitespace-nowrap">
                  Model accuracy on EPL 2024/25
                  <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-2 h-2 bg-black/90 border-l border-t border-white/20 rotate-45" />
                </div>
              </div>
            </div>
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setIsMenuOpen(!isMenuOpen)}
            className="md:hidden relative w-6 h-6 flex flex-col justify-center items-center gap-1 transition-all duration-300"
            aria-label="Toggle menu"
          >
            <span 
              className={`block w-6 h-0.5 bg-white transition-all duration-300 ${
                isMenuOpen ? 'rotate-45 translate-y-1.5' : ''
              }`} 
            />
            <span 
              className={`block w-6 h-0.5 bg-white transition-all duration-300 ${
                isMenuOpen ? 'opacity-0' : ''
              }`} 
            />
            <span 
              className={`block w-6 h-0.5 bg-white transition-all duration-300 ${
                isMenuOpen ? '-rotate-45 -translate-y-1.5' : ''
              }`} 
            />
          </button>
        </div>

        {/* Mobile Menu */}
        <motion.div
          initial={false}
          animate={{ 
            height: isMenuOpen ? 'auto' : 0,
            opacity: isMenuOpen ? 1 : 0
          }}
          transition={{ duration: 0.3 }}
          className="md:hidden overflow-hidden"
        >
          <nav className="py-6 space-y-4 border-t border-white/10">
            <Link 
              href="/matchday"
              onClick={() => setIsMenuOpen(false)}
              className="block text-neutral-300 hover:text-white transition-colors text-sm font-medium uppercase tracking-wide"
            >
              Predictions
            </Link>
            <a 
              href="#fortress"
              onClick={() => setIsMenuOpen(false)}
              className="block text-neutral-300 hover:text-white transition-colors text-sm font-medium uppercase tracking-wide"
            >
              Fortress Analysis
            </a>
            <Link 
              href="/models"
              onClick={() => setIsMenuOpen(false)}
              className="block text-neutral-300 hover:text-white transition-colors text-sm font-medium uppercase tracking-wide"
            >
              Our Models
            </Link>
            <Link 
              href="/pipeline"
              onClick={() => setIsMenuOpen(false)}
              className="block text-neutral-300 hover:text-white transition-colors text-sm font-medium uppercase tracking-wide"
            >
              Pipeline
            </Link>
            
            {/* Mobile Status */}
            <div className="pt-4 border-t border-white/10">
              <div className="flex items-center gap-2 text-sm">
                <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
                <span className="text-white">Pipeline Active</span>
                <span className="text-emerald-400 font-mono ml-auto">51.3%</span>
              </div>
            </div>
          </nav>
        </motion.div>
      </div>
    </motion.header>
  )
}