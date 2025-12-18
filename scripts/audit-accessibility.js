#!/usr/bin/env node
/**
 * audit-accessibility.js
 *
 * Accessibility audit script for the Physical AI Textbook.
 * Checks for common accessibility issues in the documentation.
 *
 * Usage: node scripts/audit-accessibility.js [--fix]
 *
 * For full browser-based auditing, run:
 *   npm run build && npx serve build -l 3000
 *   npx pa11y-ci --config .pa11yci.json
 */

const fs = require('fs');
const path = require('path');
const glob = require('glob');

const DOCS_DIR = path.join(__dirname, '..', 'docs');
const SRC_DIR = path.join(__dirname, '..', 'src');

const results = {
  total: 0,
  passed: 0,
  warnings: [],
  errors: [],
};

/**
 * WCAG 2.1 AA compliance checks for markdown content
 */
const accessibilityChecks = [
  {
    name: 'Images have alt text',
    pattern: /!\[([^\]]*)\]\([^)]+\)/g,
    check: (match, content, file) => {
      const altText = match[1];
      if (!altText || altText.trim() === '') {
        return { pass: false, message: `Image missing alt text: ${match[0]}` };
      }
      if (altText.length < 5) {
        return { pass: false, message: `Alt text too short: "${altText}"`, warning: true };
      }
      return { pass: true };
    },
  },
  {
    name: 'Heading hierarchy is sequential',
    check: (content, file) => {
      const headings = content.match(/^#{1,6}\s/gm) || [];
      let lastLevel = 0;

      for (const heading of headings) {
        const level = heading.trim().length - 1; // Count # symbols
        if (level > lastLevel + 1 && lastLevel !== 0) {
          return {
            pass: false,
            message: `Heading skips level: h${lastLevel} to h${level}`,
          };
        }
        lastLevel = level;
      }
      return { pass: true };
    },
  },
  {
    name: 'Links have descriptive text',
    pattern: /\[([^\]]+)\]\([^)]+\)/g,
    check: (match, content, file) => {
      const linkText = match[1].toLowerCase();
      const vaguePhrases = ['click here', 'here', 'link', 'read more', 'more'];

      if (vaguePhrases.includes(linkText.trim())) {
        return {
          pass: false,
          message: `Non-descriptive link text: "${match[1]}"`,
          warning: true,
        };
      }
      return { pass: true };
    },
  },
  {
    name: 'Code blocks have language specified',
    pattern: /```(\w*)\n/g,
    check: (match, content, file) => {
      const language = match[1];
      if (!language || language === '') {
        return {
          pass: false,
          message: 'Code block missing language specification',
          warning: true,
        };
      }
      return { pass: true };
    },
  },
  {
    name: 'Tables have headers',
    check: (content, file) => {
      const tablePattern = /\|[^|]+\|/;
      const headerPattern = /\|[\s-:]+\|/;

      if (tablePattern.test(content) && !headerPattern.test(content)) {
        return {
          pass: false,
          message: 'Table may be missing header row',
          warning: true,
        };
      }
      return { pass: true };
    },
  },
  {
    name: 'No empty links',
    pattern: /\[\s*\]\([^)]+\)/g,
    check: (match, content, file) => {
      return {
        pass: false,
        message: `Empty link found: ${match[0]}`,
      };
    },
  },
  {
    name: 'Color contrast in admonitions',
    check: (content, file) => {
      // Check for custom color definitions that might have contrast issues
      const colorPattern = /style=["'][^"']*color:\s*#([0-9a-fA-F]{3,6})/g;
      const issues = [];
      let match;

      while ((match = colorPattern.exec(content)) !== null) {
        // Flag any inline color styles for manual review
        issues.push(`Inline color style found - verify contrast: ${match[0]}`);
      }

      if (issues.length > 0) {
        return { pass: true, message: issues.join('; '), warning: true };
      }
      return { pass: true };
    },
  },
  {
    name: 'Form inputs have labels',
    pattern: /<input[^>]*>/gi,
    check: (match, content, file) => {
      const input = match[0];
      if (!input.includes('aria-label') && !input.includes('id=')) {
        return {
          pass: false,
          message: 'Input missing label or aria-label',
          warning: true,
        };
      }
      return { pass: true };
    },
  },
];

/**
 * Run accessibility checks on a file
 */
function auditFile(filepath) {
  const content = fs.readFileSync(filepath, 'utf-8');
  const relativePath = path.relative(path.join(__dirname, '..'), filepath);
  const fileResults = { errors: [], warnings: [] };

  for (const check of accessibilityChecks) {
    results.total++;

    if (check.pattern) {
      // Pattern-based check
      let match;
      const regex = new RegExp(check.pattern.source, check.pattern.flags);

      while ((match = regex.exec(content)) !== null) {
        const result = check.check(match, content, filepath);

        if (!result.pass) {
          const issue = {
            file: relativePath,
            check: check.name,
            message: result.message,
            line: content.substring(0, match.index).split('\n').length,
          };

          if (result.warning) {
            fileResults.warnings.push(issue);
          } else {
            fileResults.errors.push(issue);
          }
        }
      }
    } else {
      // Content-based check
      const result = check.check(content, filepath);

      if (!result.pass) {
        const issue = {
          file: relativePath,
          check: check.name,
          message: result.message,
        };

        if (result.warning) {
          fileResults.warnings.push(issue);
        } else {
          fileResults.errors.push(issue);
        }
      } else if (result.message) {
        // Warning even on pass
        fileResults.warnings.push({
          file: relativePath,
          check: check.name,
          message: result.message,
        });
      }
    }
  }

  results.errors.push(...fileResults.errors);
  results.warnings.push(...fileResults.warnings);

  if (fileResults.errors.length === 0 && fileResults.warnings.length === 0) {
    results.passed++;
  }

  return fileResults;
}

/**
 * Check React components for accessibility
 */
function auditComponents() {
  const componentFiles = glob.sync('**/*.{tsx,jsx}', { cwd: SRC_DIR, absolute: true });

  for (const file of componentFiles) {
    const content = fs.readFileSync(file, 'utf-8');
    const relativePath = path.relative(path.join(__dirname, '..'), file);

    // Check for common React accessibility issues
    const checks = [
      {
        pattern: /<img[^>]*(?!alt=)[^>]*>/g,
        message: 'Image missing alt attribute',
      },
      {
        pattern: /onClick=\{[^}]+\}(?![^>]*(?:onKeyDown|onKeyPress|role=))/g,
        message: 'onClick handler may need keyboard equivalent',
        warning: true,
      },
      {
        pattern: /<button[^>]*(?!type=)[^>]*>/g,
        message: 'Button missing type attribute',
        warning: true,
      },
      {
        pattern: /tabIndex=["']-1["']/g,
        message: 'Negative tabIndex removes element from tab order',
        warning: true,
      },
    ];

    for (const check of checks) {
      let match;
      while ((match = check.pattern.exec(content)) !== null) {
        const issue = {
          file: relativePath,
          check: 'React component',
          message: check.message,
          line: content.substring(0, match.index).split('\n').length,
        };

        if (check.warning) {
          results.warnings.push(issue);
        } else {
          results.errors.push(issue);
        }
      }
    }
  }
}

/**
 * Main function
 */
function main() {
  console.log('='.repeat(60));
  console.log('Physical AI Textbook - Accessibility Audit');
  console.log('='.repeat(60));
  console.log('\nWCAG 2.1 AA Compliance Check\n');

  // Find all markdown files
  const mdFiles = glob.sync('**/*.md', { cwd: DOCS_DIR, absolute: true });
  console.log(`Checking ${mdFiles.length} markdown files...\n`);

  // Audit markdown files
  for (const file of mdFiles) {
    const result = auditFile(file);
    const relativePath = path.relative(path.join(__dirname, '..'), file);

    if (result.errors.length > 0 || result.warnings.length > 0) {
      console.log(`${relativePath}:`);
      result.errors.forEach(e => console.log(`  ✗ ${e.message}`));
      result.warnings.forEach(w => console.log(`  ⚠ ${w.message}`));
    }
  }

  // Audit React components
  console.log('\nChecking React components...\n');
  auditComponents();

  // Summary
  console.log('\n' + '='.repeat(60));
  console.log('ACCESSIBILITY AUDIT SUMMARY');
  console.log('='.repeat(60));
  console.log(`Files checked: ${mdFiles.length}`);
  console.log(`Files passed: ${results.passed}`);
  console.log(`Errors: ${results.errors.length}`);
  console.log(`Warnings: ${results.warnings.length}`);

  if (results.errors.length > 0) {
    console.log('\nERRORS (must fix):');
    console.log('-'.repeat(50));
    for (const error of results.errors) {
      console.log(`  ✗ ${error.file}${error.line ? `:${error.line}` : ''}`);
      console.log(`    ${error.check}: ${error.message}`);
    }
  }

  if (results.warnings.length > 0) {
    console.log('\nWARNINGS (should review):');
    console.log('-'.repeat(50));
    for (const warning of results.warnings.slice(0, 20)) {
      console.log(`  ⚠ ${warning.file}${warning.line ? `:${warning.line}` : ''}`);
      console.log(`    ${warning.check}: ${warning.message}`);
    }
    if (results.warnings.length > 20) {
      console.log(`  ... and ${results.warnings.length - 20} more warnings`);
    }
  }

  // Recommendations
  console.log('\n' + '='.repeat(60));
  console.log('RECOMMENDATIONS');
  console.log('='.repeat(60));
  console.log(`
For full accessibility testing, also run:

1. Browser-based audit (after building):
   npm run build
   npx serve build -l 3000
   npx pa11y-ci --config .pa11yci.json

2. Lighthouse accessibility audit:
   npx lighthouse http://localhost:3000 --only-categories=accessibility

3. Manual testing:
   - Test with screen reader (NVDA, VoiceOver)
   - Test keyboard navigation
   - Test with browser zoom at 200%
   - Test color contrast with browser tools
`);

  // Exit code
  const exitCode = results.errors.length > 0 ? 1 : 0;
  console.log(`\n${exitCode === 0 ? '✓ No critical accessibility issues' : '✗ Accessibility issues found'}`);
  process.exit(exitCode);
}

main();
