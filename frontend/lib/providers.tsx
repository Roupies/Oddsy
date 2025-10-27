'use client'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { useState } from 'react'
import { ToastProvider } from '@/components/ui/toast'

// Cache configuration based on data freshness requirements
const queryClientConfig = {
  defaultOptions: {
    queries: {
      // Conservative defaults
      staleTime: 5 * 60 * 1000, // 5 minutes
      gcTime: 15 * 60 * 1000, // 15 minutes (was cacheTime in v4)
      refetchOnWindowFocus: false,
      refetchOnReconnect: 'always' as const,
      retry: (failureCount: number, error: unknown) => {
        // Don't retry on 404s (round not available)
        if (error instanceof Error && error.message.includes('404')) {
          return false
        }
        return failureCount < 2
      }
    },
    mutations: {
      retry: 1,
      gcTime: 5 * 60 * 1000 // 5 minutes
    }
  }
}

export function Providers({ children }: { children: React.ReactNode }) {
  // Create query client in state to ensure stable instance
  const [queryClient] = useState(() => new QueryClient(queryClientConfig))
  
  return (
    <QueryClientProvider client={queryClient}>
      <ToastProvider>
        {children}
        {process.env.NODE_ENV === 'development' && (
          <ReactQueryDevtools 
            initialIsOpen={false}
            buttonPosition="bottom-right"
          />
        )}
      </ToastProvider>
    </QueryClientProvider>
  )
}