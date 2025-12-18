#!/usr/bin/env node
/**
 * validate-content.js
 *
 * Validates chapter content for the Physical AI Textbook:
 * - Word count within specification
 * - Required sections present
 * - Internal links valid
 * - Code blocks syntax check
 *
 * Usage: node scripts/validate-content.js [--fix]
 */

const fs = require('fs');
const path = require('path');
const glob = require('glob');

// Configuration
const DOCS_DIR = path.join(__dirname, '..', 'docs');
const WORD_COUNT_LIMITS = {
  regular: { min: 1700, max: 2500 },
  capstone: { min: 3000, max: 3500 },
  frontmatter: { min: 500, max: 2000 },
  appendix: { min: 600, max: 2000 },
};

const CAPSTONE_CHAPTERS = ['ch05', 'ch07', 'ch14'];

const REQUIRED_SECTIONS = [
  'Learning Objectives',
  'Prerequisites',
  'Summary',
];

const REQUIRED_SECTIONS_CAPSTONE = [
  ...REQUIRED_SECTIONS,
  'Grading Rubric',
];

// Results tracking
const results = {
  passed: 0,
  failed: 0,
  warnings: 0,
  errors: [],
  warnings_list: [],
};

/**
 * Count words in markdown content (excluding code blocks)
 */
function countWords(content) {
  // Remove frontmatter
  const withoutFrontmatter = content.replace(/^---[\s\S]*?---\n/, '');

  // Remove code blocks
  const withoutCode = withoutFrontmatter.replace(/```[\s\S]*?```/g, '');

  // Remove HTML comments
  const withoutComments = withoutCode.replace(/<!--[\s\S]*?-->/g, '');

  // Remove Mermaid blocks
  const withoutMermaid = withoutComments.replace(/```mermaid[\s\S]*?```/g, '');

  // Count words
  const words = withoutMermaid
    .replace(/[#*_`\[\]()]/g, ' ')
    .split(/\s+/)
    .filter(word => word.length > 0);

  return words.length;
}

/**
 * Check for required sections
 */
function checkRequiredSections(content, filename, isCapstone) {
  const sections = isCapstone ? REQUIRED_SECTIONS_CAPSTONE : REQUIRED_SECTIONS;
  const missing = [];

  for (const section of sections) {
    const regex = new RegExp(`##\\s+${section}`, 'i');
    if (!regex.test(content)) {
      missing.push(section);
    }
  }

  return missing;
}

/**
 * Extract and validate internal links
 */
function extractInternalLinks(content) {
  const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
  const links = [];
  let match;

  while ((match = linkRegex.exec(content)) !== null) {
    const [, text, url] = match;
    if (!url.startsWith('http') && !url.startsWith('#')) {
      links.push({ text, url });
    }
  }

  return links;
}

/**
 * Validate a single markdown file
 */
function validateFile(filepath) {
  const filename = path.basename(filepath);
  const relativePath = path.relative(DOCS_DIR, filepath);
  const content = fs.readFileSync(filepath, 'utf-8');

  console.log(`\nValidating: ${relativePath}`);

  const fileErrors = [];
  const fileWarnings = [];

  // Determine chapter type
  const isCapstone = CAPSTONE_CHAPTERS.some(ch => filename.includes(ch));
  const isFrontmatter = relativePath.includes('front-matter');
  const isAppendix = relativePath.includes('back-matter') && filename.includes('appendix');
  const isGlossary = filename.includes('glossary');
  const isIndex = filename === 'index.md';

  // Get appropriate word count limits
  let limits;
  if (isFrontmatter) {
    limits = WORD_COUNT_LIMITS.frontmatter;
  } else if (isAppendix) {
    limits = WORD_COUNT_LIMITS.appendix;
  } else if (isCapstone) {
    limits = WORD_COUNT_LIMITS.capstone;
  } else if (isGlossary || isIndex) {
    limits = { min: 0, max: 10000 }; // No strict limit for glossary/index
  } else {
    limits = WORD_COUNT_LIMITS.regular;
  }

  // Word count validation
  const wordCount = countWords(content);
  console.log(`  Word count: ${wordCount} (limit: ${limits.min}-${limits.max})`);

  if (wordCount < limits.min) {
    fileWarnings.push(`Word count ${wordCount} below minimum ${limits.min}`);
  } else if (wordCount > limits.max) {
    fileWarnings.push(`Word count ${wordCount} above maximum ${limits.max}`);
  } else {
    console.log(`  ✓ Word count OK`);
  }

  // Required sections (only for chapter files)
  if (relativePath.includes('module-') && filename.startsWith('ch')) {
    const missingSections = checkRequiredSections(content, filename, isCapstone);
    if (missingSections.length > 0) {
      fileWarnings.push(`Missing sections: ${missingSections.join(', ')}`);
    } else {
      console.log(`  ✓ Required sections present`);
    }
  }

  // Internal links
  const internalLinks = extractInternalLinks(content);
  let brokenLinks = 0;

  for (const link of internalLinks) {
    // Resolve relative path
    let targetPath;
    if (link.url.startsWith('/')) {
      targetPath = path.join(DOCS_DIR, '..', link.url);
    } else {
      targetPath = path.join(path.dirname(filepath), link.url);
    }

    // Remove any .md extension handling for Docusaurus
    if (!targetPath.includes('.')) {
      targetPath += '.md';
    }

    // Check if file exists (simplified check)
    const basePath = targetPath.replace(/\.md$/, '');
    const exists = fs.existsSync(targetPath) ||
                   fs.existsSync(basePath + '.md') ||
                   fs.existsSync(basePath + '.mdx') ||
                   fs.existsSync(basePath);

    if (!exists && !link.url.includes('github.com')) {
      brokenLinks++;
      fileWarnings.push(`Potentially broken link: ${link.url}`);
    }
  }

  if (brokenLinks === 0 && internalLinks.length > 0) {
    console.log(`  ✓ ${internalLinks.length} internal links checked`);
  }

  // Frontmatter check
  if (!content.startsWith('---')) {
    fileErrors.push('Missing frontmatter');
  } else {
    console.log(`  ✓ Frontmatter present`);
  }

  // Code block check (basic syntax)
  const codeBlockCount = (content.match(/```/g) || []).length;
  if (codeBlockCount % 2 !== 0) {
    fileErrors.push('Unclosed code block detected');
  } else if (codeBlockCount > 0) {
    console.log(`  ✓ ${codeBlockCount / 2} code blocks properly closed`);
  }

  // Report results
  if (fileErrors.length > 0) {
    console.log(`  ✗ ERRORS:`);
    fileErrors.forEach(err => {
      console.log(`    - ${err}`);
      results.errors.push(`${relativePath}: ${err}`);
    });
    results.failed++;
  } else if (fileWarnings.length > 0) {
    console.log(`  ⚠ WARNINGS:`);
    fileWarnings.forEach(warn => {
      console.log(`    - ${warn}`);
      results.warnings_list.push(`${relativePath}: ${warn}`);
    });
    results.warnings++;
    results.passed++;
  } else {
    results.passed++;
  }

  return { errors: fileErrors, warnings: fileWarnings, wordCount };
}

/**
 * Main validation function
 */
function main() {
  console.log('='.repeat(60));
  console.log('Physical AI Textbook - Content Validation');
  console.log('='.repeat(60));

  // Find all markdown files
  const files = glob.sync('**/*.md', { cwd: DOCS_DIR, absolute: true });

  console.log(`\nFound ${files.length} markdown files to validate\n`);

  const wordCounts = {};

  for (const file of files) {
    const result = validateFile(file);
    const relativePath = path.relative(DOCS_DIR, file);
    wordCounts[relativePath] = result.wordCount;
  }

  // Summary
  console.log('\n' + '='.repeat(60));
  console.log('VALIDATION SUMMARY');
  console.log('='.repeat(60));
  console.log(`Total files: ${files.length}`);
  console.log(`Passed: ${results.passed}`);
  console.log(`Warnings: ${results.warnings}`);
  console.log(`Failed: ${results.failed}`);

  if (results.errors.length > 0) {
    console.log('\nERRORS:');
    results.errors.forEach(err => console.log(`  ✗ ${err}`));
  }

  if (results.warnings_list.length > 0) {
    console.log('\nWARNINGS:');
    results.warnings_list.forEach(warn => console.log(`  ⚠ ${warn}`));
  }

  // Word count summary
  console.log('\nWORD COUNT SUMMARY:');
  console.log('-'.repeat(50));
  const sortedFiles = Object.entries(wordCounts)
    .filter(([file]) => file.includes('module-') || file.includes('front-matter'))
    .sort((a, b) => a[0].localeCompare(b[0]));

  for (const [file, count] of sortedFiles) {
    const status = count < 500 ? '⚠' : '✓';
    console.log(`  ${status} ${file}: ${count} words`);
  }

  // Exit code
  process.exit(results.failed > 0 ? 1 : 0);
}

main();
