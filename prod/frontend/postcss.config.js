/**
 * PostCSS Configuration for Oddsy Frontend
 * ========================================
 * 
 * Configures CSS processing pipeline for the Next.js application.
 * Includes Tailwind CSS for utility-first styling and Autoprefixer
 * for cross-browser compatibility.
 */

module.exports = {
  plugins: {
    // Tailwind CSS: Utility-first CSS framework for rapid UI development
    tailwindcss: {},
    // Autoprefixer: Adds vendor prefixes for cross-browser compatibility
    autoprefixer: {},
  },
}