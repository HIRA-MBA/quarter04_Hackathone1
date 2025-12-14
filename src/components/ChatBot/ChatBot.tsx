import React, { useState, useRef, useEffect, useCallback } from 'react';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import { chatApi, ChatSource } from '../../services/api';
import styles from './ChatBot.module.css';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: ChatSource[];
}

export interface ChatBotProps {
  chapter?: string;
  defaultOpen?: boolean;
}

export default function ChatBot({
  chapter,
  defaultOpen = false,
}: ChatBotProps): React.ReactElement {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isLoading]);

  // Generate unique message ID
  const generateId = () => `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

  const handleSend = useCallback(async (content: string) => {
    setError(null);

    // Add user message
    const userMessage: Message = {
      id: generateId(),
      role: 'user',
      content,
    };
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response = await chatApi.sendMessage(content, chapter);

      // Add assistant response
      const assistantMessage: Message = {
        id: generateId(),
        role: 'assistant',
        content: response.message,
        sources: response.sources,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An error occurred';
      setError(errorMessage);

      // Add error message to chat
      const errorAssistantMessage: Message = {
        id: generateId(),
        role: 'assistant',
        content: `Sorry, I encountered an error: ${errorMessage}. Please try again.`,
      };
      setMessages((prev) => [...prev, errorAssistantMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [chapter]);

  const handleClear = useCallback(async () => {
    try {
      await chatApi.clearSession();
      setMessages([]);
      setError(null);
    } catch (err) {
      // Silently ignore clear errors
    }
  }, []);

  const toggleOpen = () => setIsOpen((prev) => !prev);

  return (
    <div className={`${styles.chatBot} ${isOpen ? styles.open : ''}`}>
      {/* Toggle Button */}
      <button
        className={styles.toggleButton}
        onClick={toggleOpen}
        aria-label={isOpen ? 'Close chat' : 'Open chat'}
        aria-expanded={isOpen}
      >
        {isOpen ? (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
        ) : (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
          </svg>
        )}
      </button>

      {/* Chat Panel */}
      {isOpen && (
        <div className={styles.chatPanel}>
          {/* Header */}
          <div className={styles.header}>
            <div className={styles.headerTitle}>
              <span className={styles.headerIcon}>🤖</span>
              <span>AI Tutor</span>
            </div>
            <button
              className={styles.clearButton}
              onClick={handleClear}
              title="Clear conversation"
              aria-label="Clear conversation"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="3 6 5 6 21 6"></polyline>
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
              </svg>
            </button>
          </div>

          {/* Messages Area */}
          <div className={styles.messagesArea}>
            {messages.length === 0 && !isLoading && (
              <div className={styles.welcomeMessage}>
                <p>Hi! I&apos;m your AI tutor for the Physical AI textbook.</p>
                <p>Ask me questions about:</p>
                <ul>
                  <li>ROS 2 and robotics concepts</li>
                  <li>Digital twin simulations</li>
                  <li>NVIDIA Isaac Sim</li>
                  <li>Vision-Language-Action models</li>
                </ul>
              </div>
            )}

            {messages.map((message) => (
              <ChatMessage
                key={message.id}
                role={message.role}
                content={message.content}
                sources={message.sources}
              />
            ))}

            {isLoading && (
              <ChatMessage
                role="assistant"
                content=""
                isLoading
              />
            )}

            {error && (
              <div className={styles.errorBanner}>
                {error}
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className={styles.inputArea}>
            <ChatInput onSend={handleSend} disabled={isLoading} />
          </div>

          {/* Footer */}
          <div className={styles.footer}>
            <span>Answers based on textbook content only</span>
          </div>
        </div>
      )}
    </div>
  );
}
