import React from 'react';
import ChatBot from '@site/src/components/ChatBot';
import UrduToggle from '@site/src/components/Translation/UrduToggle';
import { TranslationProvider } from '@site/src/context/TranslationContext';

interface RootProps {
  children: React.ReactNode;
}

// This wrapper adds global components to all pages
export default function Root({ children }: RootProps): React.ReactElement {
  return (
    <TranslationProvider>
      {children}
      <ChatBot />
      <div
        style={{
          position: 'fixed',
          bottom: '90px',
          right: '20px',
          zIndex: 999,
        }}
      >
        <UrduToggle />
      </div>
    </TranslationProvider>
  );
}
