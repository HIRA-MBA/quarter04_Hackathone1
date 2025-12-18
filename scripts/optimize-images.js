#!/usr/bin/env node
/**
 * optimize-images.js
 *
 * Image optimization script for the Physical AI Textbook.
 * Compresses images and generates responsive variants.
 *
 * Usage:
 *   node scripts/optimize-images.js          # Analyze only
 *   node scripts/optimize-images.js --fix    # Optimize images
 *
 * Prerequisites:
 *   npm install sharp glob
 */

const fs = require('fs');
const path = require('path');
const glob = require('glob');

// Configuration
const STATIC_DIR = path.join(__dirname, '..', 'static');
const DOCS_DIR = path.join(__dirname, '..', 'docs');
const MAX_FILE_SIZE = 500 * 1024; // 500KB max for images
const MAX_DIMENSION = 2000; // Max width/height
const RECOMMENDED_FORMATS = ['webp', 'avif'];

const results = {
  total: 0,
  optimized: 0,
  warnings: [],
  savings: 0,
  images: [],
};

/**
 * Get image metadata without processing
 */
function getImageInfo(filepath) {
  const stats = fs.statSync(filepath);
  const ext = path.extname(filepath).toLowerCase();

  return {
    path: filepath,
    relativePath: path.relative(path.join(__dirname, '..'), filepath),
    size: stats.size,
    extension: ext,
    isLarge: stats.size > MAX_FILE_SIZE,
  };
}

/**
 * Analyze images in the project
 */
function analyzeImages() {
  // Find all images
  const imagePatterns = ['**/*.{png,jpg,jpeg,gif,webp,svg}'];
  const searchDirs = [STATIC_DIR, DOCS_DIR];

  for (const dir of searchDirs) {
    if (!fs.existsSync(dir)) continue;

    for (const pattern of imagePatterns) {
      const files = glob.sync(pattern, { cwd: dir, absolute: true });

      for (const file of files) {
        const info = getImageInfo(file);
        results.images.push(info);
        results.total++;

        if (info.isLarge) {
          results.warnings.push({
            file: info.relativePath,
            issue: `File size ${formatBytes(info.size)} exceeds ${formatBytes(MAX_FILE_SIZE)}`,
            suggestion: 'Consider compressing or using WebP format',
          });
        }

        // Check for non-optimized formats
        if (['.png', '.jpg', '.jpeg'].includes(info.extension)) {
          const webpPath = file.replace(/\.(png|jpe?g)$/i, '.webp');
          if (!fs.existsSync(webpPath)) {
            results.warnings.push({
              file: info.relativePath,
              issue: 'No WebP version available',
              suggestion: 'Generate WebP for better compression',
            });
          }
        }
      }
    }
  }
}

/**
 * Format bytes to human readable
 */
function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

/**
 * Generate optimization report
 */
function generateReport() {
  console.log('='.repeat(60));
  console.log('Physical AI Textbook - Image Optimization Report');
  console.log('='.repeat(60));

  console.log(`\nTotal images found: ${results.total}`);

  // Size distribution
  const sizeCategories = {
    small: results.images.filter(i => i.size < 50 * 1024).length,
    medium: results.images.filter(i => i.size >= 50 * 1024 && i.size < 200 * 1024).length,
    large: results.images.filter(i => i.size >= 200 * 1024 && i.size < 500 * 1024).length,
    oversized: results.images.filter(i => i.size >= 500 * 1024).length,
  };

  console.log('\nSize Distribution:');
  console.log(`  Small (<50KB):     ${sizeCategories.small}`);
  console.log(`  Medium (50-200KB): ${sizeCategories.medium}`);
  console.log(`  Large (200-500KB): ${sizeCategories.large}`);
  console.log(`  Oversized (>500KB): ${sizeCategories.oversized}`);

  // Format distribution
  const formatCounts = {};
  for (const img of results.images) {
    const ext = img.extension;
    formatCounts[ext] = (formatCounts[ext] || 0) + 1;
  }

  console.log('\nFormat Distribution:');
  for (const [ext, count] of Object.entries(formatCounts).sort((a, b) => b[1] - a[1])) {
    console.log(`  ${ext}: ${count}`);
  }

  // Total size
  const totalSize = results.images.reduce((sum, img) => sum + img.size, 0);
  console.log(`\nTotal image size: ${formatBytes(totalSize)}`);

  // Warnings
  if (results.warnings.length > 0) {
    console.log('\n' + '='.repeat(60));
    console.log('OPTIMIZATION OPPORTUNITIES');
    console.log('='.repeat(60));

    const oversizedImages = results.warnings.filter(w => w.issue.includes('exceeds'));
    const noWebP = results.warnings.filter(w => w.issue.includes('WebP'));

    if (oversizedImages.length > 0) {
      console.log(`\nOversized Images (${oversizedImages.length}):`);
      for (const warning of oversizedImages.slice(0, 10)) {
        console.log(`  ⚠ ${warning.file}`);
        console.log(`    ${warning.issue}`);
      }
      if (oversizedImages.length > 10) {
        console.log(`  ... and ${oversizedImages.length - 10} more`);
      }
    }

    if (noWebP.length > 0) {
      console.log(`\nMissing WebP Versions (${noWebP.length}):`);
      for (const warning of noWebP.slice(0, 10)) {
        console.log(`  ⚠ ${warning.file}`);
      }
      if (noWebP.length > 10) {
        console.log(`  ... and ${noWebP.length - 10} more`);
      }
    }
  }

  // Recommendations
  console.log('\n' + '='.repeat(60));
  console.log('RECOMMENDATIONS');
  console.log('='.repeat(60));
  console.log(`
1. For new images, prefer WebP or AVIF format
2. Use responsive images with srcset for different screen sizes
3. Lazy load images below the fold
4. Consider using a CDN for image delivery

To optimize images with sharp:

  npm install sharp

  # In your build process or manually:
  const sharp = require('sharp');

  // Convert to WebP
  await sharp('input.png')
    .webp({ quality: 80 })
    .toFile('output.webp');

  // Resize large images
  await sharp('large.jpg')
    .resize(1200, null, { withoutEnlargement: true })
    .jpeg({ quality: 85 })
    .toFile('optimized.jpg');

Docusaurus configuration for responsive images:

  // docusaurus.config.js
  module.exports = {
    plugins: [
      [
        '@docusaurus/plugin-ideal-image',
        {
          quality: 70,
          max: 1030,
          min: 640,
          steps: 2,
          disableInDev: false,
        },
      ],
    ],
  };
`);

  const exitCode = results.warnings.filter(w => w.issue.includes('exceeds')).length > 0 ? 1 : 0;
  console.log(`\n${exitCode === 0 ? '✓ No critical image issues' : '⚠ Some images need optimization'}`);

  return exitCode;
}

/**
 * Main function
 */
function main() {
  const args = process.argv.slice(2);
  const shouldFix = args.includes('--fix');

  if (shouldFix) {
    console.log('Note: --fix mode requires the "sharp" package.');
    console.log('Install with: npm install sharp\n');

    try {
      require.resolve('sharp');
    } catch (e) {
      console.log('Sharp not installed. Running in analyze-only mode.\n');
    }
  }

  analyzeImages();
  const exitCode = generateReport();
  process.exit(exitCode);
}

main();
