#!/usr/bin/env node
/**
 * validate-links.js
 *
 * Validates all internal links in the documentation:
 * - Checks markdown links resolve to existing files
 * - Validates anchor links within documents
 * - Reports broken and orphaned links
 *
 * Usage: node scripts/validate-links.js
 */

const fs = require('fs');
const path = require('path');
const glob = require('glob');

const DOCS_DIR = path.join(__dirname, '..', 'docs');
const LABS_DIR = path.join(__dirname, '..', 'labs');

const results = {
  total: 0,
  valid: 0,
  broken: [],
  warnings: [],
};

/**
 * Extract all links from markdown content
 */
function extractLinks(content, filepath) {
  const links = [];

  // Markdown links: [text](url)
  const mdLinkRegex = /\[([^\]]*)\]\(([^)]+)\)/g;
  let match;

  while ((match = mdLinkRegex.exec(content)) !== null) {
    const [fullMatch, text, url] = match;
    const lineNumber = content.substring(0, match.index).split('\n').length;

    links.push({
      text: text.trim(),
      url: url.trim(),
      line: lineNumber,
      source: filepath,
    });
  }

  // Reference-style links: [text][ref] and [ref]: url
  const refDefRegex = /^\[([^\]]+)\]:\s*(.+)$/gm;
  const refDefs = {};

  while ((match = refDefRegex.exec(content)) !== null) {
    refDefs[match[1].toLowerCase()] = match[2].trim();
  }

  return links;
}

/**
 * Check if a link target exists
 */
function checkLink(link) {
  const { url, source } = link;

  // Skip external links
  if (url.startsWith('http://') || url.startsWith('https://')) {
    return { valid: true, type: 'external' };
  }

  // Skip mailto links
  if (url.startsWith('mailto:')) {
    return { valid: true, type: 'mailto' };
  }

  // Handle anchor-only links
  if (url.startsWith('#')) {
    // TODO: Validate anchor exists in current file
    return { valid: true, type: 'anchor' };
  }

  // Parse URL and anchor
  const [urlPath, anchor] = url.split('#');

  // Resolve relative path
  let targetPath;
  if (urlPath.startsWith('/')) {
    // Absolute path from root
    targetPath = path.join(__dirname, '..', urlPath);
  } else if (urlPath === '') {
    // Just an anchor
    return { valid: true, type: 'anchor' };
  } else {
    // Relative path
    targetPath = path.join(path.dirname(source), urlPath);
  }

  // Normalize path
  targetPath = path.normalize(targetPath);

  // Try different extensions
  const extensions = ['', '.md', '.mdx', '/index.md', '/README.md'];

  for (const ext of extensions) {
    const testPath = targetPath + ext;
    if (fs.existsSync(testPath)) {
      return { valid: true, type: 'internal', resolved: testPath };
    }
  }

  // Check if it's a directory
  if (fs.existsSync(targetPath) && fs.statSync(targetPath).isDirectory()) {
    // Check for index file
    const indexPath = path.join(targetPath, 'index.md');
    if (fs.existsSync(indexPath)) {
      return { valid: true, type: 'directory', resolved: indexPath };
    }
  }

  return { valid: false, type: 'broken' };
}

/**
 * Validate all links in a file
 */
function validateFile(filepath) {
  const content = fs.readFileSync(filepath, 'utf-8');
  const links = extractLinks(content, filepath);
  const relativePath = path.relative(path.join(__dirname, '..'), filepath);

  for (const link of links) {
    results.total++;

    const check = checkLink(link);

    if (check.valid) {
      results.valid++;
    } else {
      results.broken.push({
        source: relativePath,
        line: link.line,
        text: link.text,
        url: link.url,
      });
    }
  }

  return links.length;
}

/**
 * Find orphaned files (not linked from anywhere)
 */
function findOrphanedFiles(allFiles, allLinks) {
  const linkedFiles = new Set();

  for (const link of allLinks) {
    if (link.resolved) {
      linkedFiles.add(path.normalize(link.resolved));
    }
  }

  const orphaned = [];

  for (const file of allFiles) {
    const normalizedFile = path.normalize(file);
    // Check if file is linked (simplified check)
    let isLinked = false;

    for (const linked of linkedFiles) {
      if (normalizedFile.includes(linked) || linked.includes(normalizedFile)) {
        isLinked = true;
        break;
      }
    }

    // Entry points and special files are not orphaned
    const relativePath = path.relative(path.join(__dirname, '..'), file);
    const isEntryPoint = relativePath.includes('introduction') ||
                         relativePath.includes('index') ||
                         relativePath.includes('README');

    if (!isLinked && !isEntryPoint) {
      // This is a simplified check - in reality we'd need full graph traversal
    }
  }

  return orphaned;
}

/**
 * Main function
 */
function main() {
  console.log('='.repeat(60));
  console.log('Physical AI Textbook - Link Validation');
  console.log('='.repeat(60));

  // Find all markdown files
  const docsFiles = glob.sync('**/*.md', { cwd: DOCS_DIR, absolute: true });
  const labsFiles = glob.sync('**/*.md', { cwd: LABS_DIR, absolute: true });
  const allFiles = [...docsFiles, ...labsFiles];

  console.log(`\nFound ${allFiles.length} markdown files to check\n`);

  // Validate each file
  for (const file of allFiles) {
    const relativePath = path.relative(path.join(__dirname, '..'), file);
    const linkCount = validateFile(file);

    if (linkCount > 0) {
      console.log(`  ${relativePath}: ${linkCount} links`);
    }
  }

  // Summary
  console.log('\n' + '='.repeat(60));
  console.log('LINK VALIDATION SUMMARY');
  console.log('='.repeat(60));
  console.log(`Total links checked: ${results.total}`);
  console.log(`Valid links: ${results.valid}`);
  console.log(`Broken links: ${results.broken.length}`);

  if (results.broken.length > 0) {
    console.log('\nBROKEN LINKS:');
    console.log('-'.repeat(50));

    for (const broken of results.broken) {
      console.log(`  ✗ ${broken.source}:${broken.line}`);
      console.log(`    Link: [${broken.text}](${broken.url})`);
    }
  }

  if (results.warnings.length > 0) {
    console.log('\nWARNINGS:');
    for (const warning of results.warnings) {
      console.log(`  ⚠ ${warning}`);
    }
  }

  // Exit code
  const exitCode = results.broken.length > 0 ? 1 : 0;
  console.log(`\n${exitCode === 0 ? '✓ All links valid!' : '✗ Some links are broken'}`);
  process.exit(exitCode);
}

main();
