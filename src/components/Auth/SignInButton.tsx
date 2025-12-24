/**
 * Sign-in button component for the navbar.
 * Shows "Sign In" when not authenticated, user dropdown when authenticated.
 */

import React, { useState, useRef, useEffect } from 'react';
import Link from '@docusaurus/Link';
import { useAuth } from '../../contexts/AuthContext';
import styles from './SignInButton.module.css';

export default function SignInButton(): React.ReactElement | null {
  const { user, isAuthenticated, isLoading, signout } = useAuth();
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setDropdownOpen(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  if (isLoading) {
    return (
      <div className={styles.container}>
        <span className={styles.loading}>...</span>
      </div>
    );
  }

  if (!isAuthenticated || !user) {
    return (
      <div className={styles.container}>
        <Link to="/auth/signin" className={styles.signInButton}>
          Sign In
        </Link>
      </div>
    );
  }

  const handleSignout = async () => {
    setDropdownOpen(false);
    await signout();
  };

  const userInitial = user.full_name
    ? user.full_name.charAt(0).toUpperCase()
    : user.email.charAt(0).toUpperCase();

  return (
    <div className={styles.container} ref={dropdownRef}>
      <button
        type="button"
        className={styles.userButton}
        onClick={() => setDropdownOpen(!dropdownOpen)}
        aria-expanded={dropdownOpen}
        aria-haspopup="true"
      >
        <span className={styles.avatar}>{userInitial}</span>
        <span className={styles.userName}>
          {user.full_name || user.email.split('@')[0]}
        </span>
        <svg
          className={`${styles.chevron} ${dropdownOpen ? styles.chevronUp : ''}`}
          width="12"
          height="12"
          viewBox="0 0 12 12"
          fill="none"
        >
          <path
            d="M2.5 4.5L6 8L9.5 4.5"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>

      {dropdownOpen && (
        <div className={styles.dropdown}>
          <div className={styles.dropdownHeader}>
            <span className={styles.dropdownEmail}>{user.email}</span>
          </div>
          <div className={styles.dropdownDivider} />
          <Link
            to="/auth/questionnaire"
            className={styles.dropdownItem}
            onClick={() => setDropdownOpen(false)}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
              <circle cx="12" cy="7" r="4" />
            </svg>
            My Profile
          </Link>
          <Link
            to="/auth/questionnaire"
            className={styles.dropdownItem}
            onClick={() => setDropdownOpen(false)}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
              <circle cx="12" cy="12" r="3" />
            </svg>
            Preferences
          </Link>
          <div className={styles.dropdownDivider} />
          <button
            type="button"
            className={styles.dropdownItem}
            onClick={handleSignout}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
              <polyline points="16,17 21,12 16,7" />
              <line x1="21" y1="12" x2="9" y2="12" />
            </svg>
            Sign Out
          </button>
        </div>
      )}
    </div>
  );
}
