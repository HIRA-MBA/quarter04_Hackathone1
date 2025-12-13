/**
 * Authentication context for managing user state across the application.
 */

import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import { authApi, UserResponse } from '../services/auth';
import { userApi, UserPreferences } from '../services/user';

interface AuthContextType {
  user: UserResponse | null;
  preferences: UserPreferences | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  signin: (email: string, password: string) => Promise<void>;
  signup: (email: string, password: string, fullName?: string) => Promise<void>;
  signout: () => Promise<void>;
  refreshUser: () => Promise<void>;
  updatePreferences: (prefs: Partial<{ language: string; theme: string; font_size: string }>) => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<UserResponse | null>(null);
  const [preferences, setPreferences] = useState<UserPreferences | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const refreshUser = useCallback(async () => {
    try {
      const userData = await authApi.getCurrentUser();
      setUser(userData);

      if (userData) {
        const prefs = await userApi.getPreferences();
        setPreferences(prefs);
      } else {
        setPreferences(null);
      }
    } catch (error) {
      setUser(null);
      setPreferences(null);
    }
  }, []);

  useEffect(() => {
    const initAuth = async () => {
      setIsLoading(true);
      await refreshUser();
      setIsLoading(false);
    };

    initAuth();
  }, [refreshUser]);

  const signin = async (email: string, password: string) => {
    await authApi.signin({ email, password });
    await refreshUser();
  };

  const signup = async (email: string, password: string, fullName?: string) => {
    await authApi.signup({ email, password, full_name: fullName });
    await refreshUser();
  };

  const signout = async () => {
    await authApi.signout();
    setUser(null);
    setPreferences(null);
  };

  const updatePreferences = async (
    prefs: Partial<{ language: string; theme: string; font_size: string }>
  ) => {
    const updated = await userApi.updatePreferences(prefs);
    setPreferences(updated);
  };

  const value: AuthContextType = {
    user,
    preferences,
    isLoading,
    isAuthenticated: !!user,
    signin,
    signup,
    signout,
    refreshUser,
    updatePreferences,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
