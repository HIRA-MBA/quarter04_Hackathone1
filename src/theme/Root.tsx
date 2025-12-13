import React from 'react';
import ChatBot from '@site/src/components/ChatBot';

interface RootProps {
  children: React.ReactNode;
}

// This wrapper adds the ChatBot to all pages
export default function Root({ children }: RootProps): JSX.Element {
  return (
    <>
      {children}
      <ChatBot />
    </>
  );
}
