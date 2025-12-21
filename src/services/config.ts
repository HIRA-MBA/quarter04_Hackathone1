/**
 * Configuration service for accessing environment-specific settings.
 * Uses Docusaurus custom fields for server-side configuration.
 */

// Default API URL - will be overridden by Docusaurus context in browser
let apiUrl = 'https://quarter04-hackathone1.onrender.com';

/**
 * Initialize configuration from Docusaurus context.
 * Call this from a React component that has access to useDocusaurusContext.
 */
export function initConfig(customFields: Record<string, unknown>): void {
  if (customFields?.apiUrl && typeof customFields.apiUrl === 'string') {
    apiUrl = customFields.apiUrl;
  }
}

/**
 * Get the API base URL.
 */
export function getApiUrl(): string {
  // In browser, check if we're on localhost
  if (typeof window !== 'undefined') {
    const hostname = window.location.hostname;
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
      return 'http://localhost:8000';
    }
  }
  return apiUrl;
}

/**
 * Check if we're in development mode.
 */
export function isDevelopment(): boolean {
  if (typeof window !== 'undefined') {
    const hostname = window.location.hostname;
    return hostname === 'localhost' || hostname === '127.0.0.1';
  }
  return false;
}
