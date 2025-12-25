/**
 * OAuth callback page - handles tokens from OAuth redirect.
 */

import React, { useEffect, useState } from 'react';
import { useHistory } from '@docusaurus/router';
import Layout from '@theme/Layout';
import styles from './auth.module.css';
import { authApi } from '../../services/auth';

export default function OAuthCallbackPage(): React.ReactElement {
  const history = useHistory();
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState('Processing login...');

  useEffect(() => {
    const handleCallback = async () => {
      try {
        // Check for error in URL
        const params = new URLSearchParams(window.location.search);
        const errorParam = params.get('error');

        if (errorParam) {
          setError(errorParam);
          return;
        }

        // Handle the OAuth callback (stores tokens)
        const result = authApi.handleOAuthCallback();

        if (result) {
          setStatus('Login successful! Redirecting...');
          // Redirect to home or questionnaire for new users
          setTimeout(() => {
            history.push('/');
          }, 1000);
        } else {
          setError('No authentication tokens received');
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Authentication failed');
      }
    };

    handleCallback();
  }, [history]);

  return (
    <Layout title="Signing In" description="Completing authentication">
      <div className={styles.authContainer}>
        <div className={styles.authCard}>
          <h1>{error ? 'Authentication Failed' : 'Signing In'}</h1>

          {error ? (
            <>
              <div className={styles.error}>{error}</div>
              <p className={styles.subtitle}>
                Something went wrong during authentication. Please try again.
              </p>
              <a href="/auth/signin" className={styles.submitButton}>
                Back to Sign In
              </a>
            </>
          ) : (
            <>
              <div className={styles.loadingSpinner} />
              <p className={styles.subtitle}>{status}</p>
            </>
          )}
        </div>
      </div>
    </Layout>
  );
}
