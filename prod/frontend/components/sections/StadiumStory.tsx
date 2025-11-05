'use client'

import React from 'react'
import { motion } from 'framer-motion'
import Image from 'next/image'

interface StadiumStoryProps {
  stadium: string
  team: string
  narrative: string
  imageUrl: string
  imagePosition?: 'left' | 'right'
  delay?: number
}

export const StadiumStory: React.FC<StadiumStoryProps> = ({
  stadium,
  team,
  narrative,
  imageUrl,
  imagePosition = 'left',
  delay = 0
}) => {
  const [isMounted, setIsMounted] = React.useState(false)
  const isImageRight = imagePosition === 'right'

  React.useEffect(() => {
    setIsMounted(true)
  }, [])

  if (!isMounted) {
    return (
      <div className={`
        grid grid-cols-1 md:grid-cols-[75%_25%] gap-6 md:gap-8 opacity-0
        ${isImageRight ? 'md:grid-cols-[25%_75%]' : ''}
      `}>
        {/* Placeholder content */}
      </div>
    )
  }

  return (
    <motion.div 
      initial={{ opacity: 0, y: 40 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8, delay }}
      viewport={{ once: true }}
      className={`
        grid grid-cols-1 md:grid-cols-[75%_25%] gap-6 md:gap-8
        ${isImageRight ? 'md:grid-cols-[25%_75%]' : ''}
      `}
    >
      {/* Image stade */}
      <div className={`
        relative h-[50vh] md:h-[70vh] overflow-hidden group rounded-lg
        ${isImageRight ? 'md:order-2' : ''}
      `}>
        <div 
          className="stadium-img w-full h-full bg-cover bg-center bg-no-repeat transition-all duration-700 group-hover:scale-102"
          style={{ backgroundImage: `url('${imageUrl}')` }}
        />
        
        {/* Vignette effect */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent" />
        
        {/* Statistiques supprimées pour design plus épuré */}

      </div>

      {/* Content narratif */}
      <div className={`
        flex flex-col justify-center 
        ${isImageRight ? 'md:order-1 md:text-right md:items-end' : ''}
      `}>
        <motion.div 
          initial={{ opacity: 0, x: isImageRight ? 20 : -20 }}
          whileInView={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: delay + 0.2 }}
          viewport={{ once: true }}
          className={`
            flex items-center gap-3 mb-4 
            ${isImageRight ? 'md:justify-end md:flex-row-reverse' : ''}
          `}
        >
          <div className="w-1 h-12 bg-emerald-500" />
          <span className="text-xs font-mono uppercase tracking-wider text-neutral-500">
            Venue Analysis
          </span>
        </motion.div>
        
        <motion.h3 
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: delay + 0.4 }}
          viewport={{ once: true }}
          className="text-[clamp(2rem,4vw,3rem)] font-black tracking-tight mb-4 leading-[1.1] text-white"
        >
          {stadium.toUpperCase()}
        </motion.h3>
        
        <motion.p 
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: delay + 0.5 }}
          viewport={{ once: true }}
          className={`
            text-neutral-400 text-lg leading-relaxed mb-6 max-w-md
            ${isImageRight ? 'md:text-right' : ''}
          `}
        >
          {narrative}
        </motion.p>

        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: delay + 0.6 }}
          viewport={{ once: true }}
          className={`
            flex flex-wrap gap-4 text-sm
            ${isImageRight ? 'md:justify-end' : ''}
          `}
        >
          <div className="bg-neutral-900 border border-neutral-800 rounded px-4 py-2 text-neutral-300">
            🏟️ {team}
          </div>
        </motion.div>
      </div>
    </motion.div>
  )
}