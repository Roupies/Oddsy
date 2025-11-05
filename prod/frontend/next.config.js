/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    // Enable TypeScript strict mode
    typedRoutes: true,
  },
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  },
  images: {
    unoptimized: true,
    formats: ['image/webp', 'image/avif'],
    deviceSizes: [640, 750, 828, 1080, 1200, 1920],
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
  },
  async redirects() {
    return [
      {
        source: '/matchday',
        destination: '/predictions/latest',
        permanent: true
      },
      {
        source: '/matchday/:round',
        destination: '/predictions/:round',
        permanent: true
      }
    ];
  },
  async rewrites() {
    return [
      // Proxy API system et gameweeks vers le backend
      {
        source: '/api/system/:path*',
        destination: `${process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'}/api/system/:path*`
      },
      {
        source: '/api/gameweeks/:path*',
        destination: `${process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'}/api/gameweeks/:path*`
      },
      // Legacy aliases
      {
        source: '/api/v1/:path*',
        destination: `${process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'}/api/v1/:path*`
      }
    ];
  }
};

module.exports = nextConfig;