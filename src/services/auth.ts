/**
 * Authentication API client for user signup, signin, and session management.
 */

import { getApiUrl } from './config';

// Token storage keys
const ACCESS_TOKEN_KEY = 'physical_ai_access_token';
const REFRESH_TOKEN_KEY = 'physical_ai_refresh_token';

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface UserResponse {
  id: string;
  email: string;
  full_name: string | null;
  is_verified: boolean;
  oauth_provider: string | null;
}

export interface SignupRequest {
  email: string;
  password: string;
  full_name?: string;
}

export interface SigninRequest {
  email: string;
  password: string;
}

class AuthApiClient {
  private get baseUrl(): string {
    return getApiUrl();
  }

  /**
   * Get the stored access token.
   */
  getAccessToken(): string | null {
    return localStorage.getItem(ACCESS_TOKEN_KEY);
  }

  /**
   * Get the stored refresh token.
   */
  getRefreshToken(): string | null {
    return localStorage.getItem(REFRESH_TOKEN_KEY);
  }

  /**
   * Store tokens in localStorage.
   */
  private storeTokens(tokens: TokenResponse): void {
    localStorage.setItem(ACCESS_TOKEN_KEY, tokens.access_token);
    localStorage.setItem(REFRESH_TOKEN_KEY, tokens.refresh_token);
  }

  /**
   * Clear stored tokens.
   */
  clearTokens(): void {
    localStorage.removeItem(ACCESS_TOKEN_KEY);
    localStorage.removeItem(REFRESH_TOKEN_KEY);
  }

  /**
   * Check if user is authenticated.
   */
  isAuthenticated(): boolean {
    return !!this.getAccessToken();
  }

  /**
   * Get authorization headers for API requests.
   */
  getAuthHeaders(): Record<string, string> {
    const token = this.getAccessToken();
    if (!token) return {};
    return { Authorization: `Bearer ${token}` };
  }

  /**
   * Register a new user.
   */
  async signup(data: SignupRequest): Promise<TokenResponse> {
    const response = await fetch(`${this.baseUrl}/api/auth/signup`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Signup failed' }));
      throw new Error(error.detail || 'Signup failed');
    }

    const tokens: TokenResponse = await response.json();
    this.storeTokens(tokens);
    return tokens;
  }

  /**
   * Sign in with email and password.
   */
  async signin(data: SigninRequest): Promise<TokenResponse> {
    const response = await fetch(`${this.baseUrl}/api/auth/signin`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Invalid credentials' }));
      throw new Error(error.detail || 'Invalid credentials');
    }

    const tokens: TokenResponse = await response.json();
    this.storeTokens(tokens);
    return tokens;
  }

  /**
   * Sign out and invalidate session.
   */
  async signout(): Promise<void> {
    const token = this.getAccessToken();
    if (token) {
      try {
        await fetch(`${this.baseUrl}/api/auth/signout`, {
          method: 'POST',
          headers: {
            ...this.getAuthHeaders(),
          },
        });
      } catch (e) {
        // Ignore signout errors
      }
    }
    this.clearTokens();
  }

  /**
   * Refresh the access token.
   */
  async refreshToken(): Promise<TokenResponse | null> {
    const refreshToken = this.getRefreshToken();
    if (!refreshToken) return null;

    try {
      const response = await fetch(`${this.baseUrl}/api/auth/refresh`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ refresh_token: refreshToken }),
      });

      if (!response.ok) {
        this.clearTokens();
        return null;
      }

      const tokens: TokenResponse = await response.json();
      this.storeTokens(tokens);
      return tokens;
    } catch (e) {
      this.clearTokens();
      return null;
    }
  }

  /**
   * Get current user information.
   */
  async getCurrentUser(): Promise<UserResponse | null> {
    const token = this.getAccessToken();
    if (!token) return null;

    try {
      const response = await fetch(`${this.baseUrl}/api/auth/me`, {
        headers: {
          ...this.getAuthHeaders(),
        },
      });

      if (!response.ok) {
        if (response.status === 401) {
          // Try to refresh token
          const refreshed = await this.refreshToken();
          if (refreshed) {
            return this.getCurrentUser();
          }
          return null;
        }
        return null;
      }

      return response.json();
    } catch (e) {
      return null;
    }
  }

  /**
   * Request password reset.
   */
  async requestPasswordReset(email: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/api/auth/password-reset`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email }),
    });

    if (!response.ok) {
      throw new Error('Failed to request password reset');
    }
  }

  /**
   * Reset password with token.
   */
  async resetPassword(token: string, newPassword: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/api/auth/password-reset/confirm`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ token, new_password: newPassword }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Password reset failed' }));
      throw new Error(error.detail || 'Password reset failed');
    }
  }

  /**
   * Get OAuth redirect URL from backend and redirect user.
   */
  getOAuthUrl(provider: 'google' | 'github'): string {
    // This returns the API endpoint - the actual redirect happens via the backend
    return `${this.baseUrl}/api/auth/oauth/${provider}`;
  }

  /**
   * Initiate OAuth flow - fetches the OAuth URL and redirects.
   */
  async initiateOAuth(provider: 'google' | 'github'): Promise<void> {
    const response = await fetch(`${this.baseUrl}/api/auth/oauth/${provider}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'OAuth not configured' }));
      throw new Error(error.detail || 'Failed to initiate OAuth');
    }

    const data = await response.json();
    // Redirect to the OAuth provider
    window.location.href = data.url;
  }

  /**
   * Handle OAuth callback - extract tokens from URL params.
   */
  handleOAuthCallback(): { accessToken: string; refreshToken: string } | null {
    const params = new URLSearchParams(window.location.search);
    const accessToken = params.get('access_token');
    const refreshToken = params.get('refresh_token');
    const error = params.get('error');

    if (error) {
      throw new Error(error);
    }

    if (accessToken && refreshToken) {
      this.storeTokens({
        access_token: accessToken,
        refresh_token: refreshToken,
        token_type: 'bearer',
        expires_in: 1800,
      });
      return { accessToken, refreshToken };
    }

    return null;
  }
}

// Export singleton instance
export const authApi = new AuthApiClient();

// Export class for testing
export { AuthApiClient };
