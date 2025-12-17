/**
 * API client for the RAG chatbot backend.
 */

import { getApiUrl } from './config';

export interface ChatSource {
  chapter: string;
  section: string;
  score: number;
  content?: string;
}

export interface ChatResponse {
  message: string;
  sources: ChatSource[];
  session_id: string;
  tokens_used: number;
}

export interface ChatRequest {
  message: string;
  session_id?: string;
  chapter?: string;
  stream?: boolean;
}

export interface SearchResult {
  content: string;
  chapter: string;
  section: string;
  score: number;
  metadata: Record<string, unknown>;
}

export interface CollectionStats {
  collection_name: string;
  vectors_count?: number;
  points_count?: number;
  status?: string;
  error?: string;
}

class ChatApiClient {
  private sessionId: string | null = null;

  private get baseUrl(): string {
    return getApiUrl();
  }

  /**
   * Send a chat message and get a response.
   */
  async sendMessage(
    message: string,
    chapter?: string
  ): Promise<ChatResponse> {
    const response = await fetch(`${this.baseUrl}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        session_id: this.sessionId,
        chapter,
        stream: false,
      } as ChatRequest),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Chat API error: ${response.status} - ${error}`);
    }

    const data: ChatResponse = await response.json();
    this.sessionId = data.session_id;
    return data;
  }

  /**
   * Send a chat message and receive a streaming response.
   */
  async *sendMessageStream(
    message: string,
    chapter?: string
  ): AsyncGenerator<string, void, unknown> {
    const response = await fetch(`${this.baseUrl}/api/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        session_id: this.sessionId,
        chapter,
      } as ChatRequest),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Chat API error: ${response.status} - ${error}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Response body is not readable');
    }

    const decoder = new TextDecoder();

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value, { stream: true });
        yield text;
      }
    } finally {
      reader.releaseLock();
    }
  }

  /**
   * Search the textbook content.
   */
  async search(
    query: string,
    limit: number = 5,
    chapter?: string
  ): Promise<{ query: string; results: SearchResult[]; count: number }> {
    const params = new URLSearchParams({
      q: query,
      limit: limit.toString(),
    });
    if (chapter) {
      params.append('chapter', chapter);
    }

    const response = await fetch(`${this.baseUrl}/api/chat/search?${params}`);

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Search API error: ${response.status} - ${error}`);
    }

    return response.json();
  }

  /**
   * Get collection statistics.
   */
  async getStats(): Promise<CollectionStats> {
    const response = await fetch(`${this.baseUrl}/api/chat/stats`);

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Stats API error: ${response.status} - ${error}`);
    }

    return response.json();
  }

  /**
   * Clear the current chat session.
   */
  async clearSession(): Promise<void> {
    if (!this.sessionId) return;

    const response = await fetch(`${this.baseUrl}/api/chat/${this.sessionId}`, {
      method: 'DELETE',
    });

    if (!response.ok && response.status !== 404) {
      const error = await response.text();
      throw new Error(`Clear session error: ${response.status} - ${error}`);
    }

    this.sessionId = null;
  }

  /**
   * Submit feedback for a response.
   */
  async submitFeedback(
    messageIndex: number,
    rating: number,
    comment?: string
  ): Promise<void> {
    if (!this.sessionId) {
      throw new Error('No active session');
    }

    const response = await fetch(`${this.baseUrl}/api/chat/feedback`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        session_id: this.sessionId,
        message_index: messageIndex,
        rating,
        comment,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Feedback API error: ${response.status} - ${error}`);
    }
  }

  /**
   * Get the current session ID.
   */
  getSessionId(): string | null {
    return this.sessionId;
  }

  /**
   * Set a specific session ID (for restoring sessions).
   */
  setSessionId(sessionId: string): void {
    this.sessionId = sessionId;
  }
}

// Export singleton instance
export const chatApi = new ChatApiClient();

// Export class for testing or custom instances
export { ChatApiClient };
