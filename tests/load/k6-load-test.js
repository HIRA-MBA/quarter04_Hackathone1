/**
 * k6 Load Testing Script for Physical AI Textbook
 *
 * Tests the backend API endpoints under load to validate:
 * - NFR-001: RAG chatbot responses within 3 seconds (p95)
 * - General API performance and stability
 *
 * Prerequisites:
 *   - Install k6: https://k6.io/docs/get-started/installation/
 *   - Backend running: uvicorn app.main:app --host 0.0.0.0 --port 8000
 *
 * Usage:
 *   k6 run tests/load/k6-load-test.js
 *   k6 run --vus 50 --duration 2m tests/load/k6-load-test.js
 *   k6 run --out json=results.json tests/load/k6-load-test.js
 *
 * Environment Variables:
 *   BASE_URL: Backend API URL (default: http://localhost:8000)
 */

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const chatLatency = new Trend('chat_latency', true);
const searchLatency = new Trend('search_latency', true);
const errorRate = new Rate('errors');

// Configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

// Test scenarios
export const options = {
  scenarios: {
    // Smoke test - minimal load to verify system works
    smoke: {
      executor: 'constant-vus',
      vus: 1,
      duration: '30s',
      startTime: '0s',
      tags: { scenario: 'smoke' },
    },
    // Load test - normal expected traffic
    load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '1m', target: 20 },  // Ramp up to 20 users
        { duration: '3m', target: 20 },  // Stay at 20 users
        { duration: '1m', target: 0 },   // Ramp down
      ],
      startTime: '30s',
      tags: { scenario: 'load' },
    },
    // Stress test - beyond normal capacity
    stress: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '1m', target: 50 },  // Ramp up
        { duration: '2m', target: 50 },  // Stay at peak
        { duration: '30s', target: 0 },  // Ramp down
      ],
      startTime: '6m',
      tags: { scenario: 'stress' },
    },
  },
  thresholds: {
    // NFR-001: RAG chatbot p95 latency < 3 seconds
    'chat_latency': ['p(95)<3000'],
    // General API response times
    'http_req_duration': ['p(95)<2000', 'p(99)<5000'],
    // Error rate should be below 1%
    'errors': ['rate<0.01'],
    // Health check must be fast
    'http_req_duration{endpoint:health}': ['p(99)<500'],
  },
};

// Sample chat queries for realistic load testing
const CHAT_QUERIES = [
  'What is ROS 2?',
  'How do I create a ROS 2 node?',
  'Explain the publisher-subscriber pattern in ROS 2',
  'What is URDF and how is it used?',
  'How do I set up Gazebo simulation?',
  'What is NVIDIA Isaac Sim?',
  'Explain Vision-Language-Action models',
  'How do I implement a balance controller for a humanoid robot?',
  'What sensors are commonly used in robotics?',
  'How do I deploy ROS 2 on edge devices?',
];

// Sample search queries
const SEARCH_QUERIES = [
  'ROS 2 node',
  'sensor fusion',
  'URDF model',
  'Gazebo physics',
  'Isaac Sim',
  'navigation stack',
  'reinforcement learning',
  'humanoid locomotion',
  'grasp planning',
  'VLA architecture',
];

/**
 * Get a random item from an array
 */
function randomItem(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

/**
 * Generate a unique session ID
 */
function generateSessionId() {
  return `load-test-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Test health endpoint
 */
function testHealth() {
  const res = http.get(`${BASE_URL}/health`, {
    tags: { endpoint: 'health' },
  });

  const success = check(res, {
    'health status is 200': (r) => r.status === 200,
    'health response has status': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.status === 'healthy';
      } catch {
        return false;
      }
    },
  });

  if (!success) {
    errorRate.add(1);
  }

  return success;
}

/**
 * Test chat endpoint (RAG chatbot)
 */
function testChat() {
  const query = randomItem(CHAT_QUERIES);
  const sessionId = generateSessionId();

  const payload = JSON.stringify({
    message: query,
    session_id: sessionId,
    chapter: 'ch01-welcome-first-node',
    stream: false,
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
    tags: { endpoint: 'chat' },
  };

  const startTime = Date.now();
  const res = http.post(`${BASE_URL}/api/chat`, payload, params);
  const duration = Date.now() - startTime;

  chatLatency.add(duration);

  const success = check(res, {
    'chat status is 200': (r) => r.status === 200,
    'chat response has message': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.message && body.message.length > 0;
      } catch {
        return false;
      }
    },
    'chat response has sources': (r) => {
      try {
        const body = JSON.parse(r.body);
        return Array.isArray(body.sources);
      } catch {
        return false;
      }
    },
    'chat latency under 3s': () => duration < 3000,
  });

  if (!success) {
    errorRate.add(1);
  }

  return success;
}

/**
 * Test search endpoint
 */
function testSearch() {
  const query = randomItem(SEARCH_QUERIES);

  const res = http.get(`${BASE_URL}/api/chat/search?q=${encodeURIComponent(query)}&limit=5`, {
    tags: { endpoint: 'search' },
  });

  const startTime = Date.now();
  const duration = Date.now() - startTime;
  searchLatency.add(duration);

  const success = check(res, {
    'search status is 200': (r) => r.status === 200,
    'search has results': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.results && Array.isArray(body.results);
      } catch {
        return false;
      }
    },
  });

  if (!success) {
    errorRate.add(1);
  }

  return success;
}

/**
 * Test collection stats endpoint
 */
function testStats() {
  const res = http.get(`${BASE_URL}/api/chat/stats`, {
    tags: { endpoint: 'stats' },
  });

  const success = check(res, {
    'stats status is 200': (r) => r.status === 200,
    'stats has collection name': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.collection_name !== undefined;
      } catch {
        return false;
      }
    },
  });

  if (!success) {
    errorRate.add(1);
  }

  return success;
}

/**
 * Main test function - executed for each virtual user
 */
export default function () {
  group('Health Check', () => {
    testHealth();
    sleep(0.5);
  });

  group('RAG Chatbot', () => {
    testChat();
    sleep(1); // Simulate user thinking time
  });

  group('Content Search', () => {
    testSearch();
    sleep(0.5);
  });

  group('Stats', () => {
    testStats();
    sleep(0.5);
  });

  // Add some randomness to simulate real user behavior
  sleep(Math.random() * 2 + 1);
}

/**
 * Setup function - runs once before the test
 */
export function setup() {
  console.log(`Starting load test against ${BASE_URL}`);

  // Verify backend is reachable
  const res = http.get(`${BASE_URL}/health`);
  if (res.status !== 200) {
    throw new Error(`Backend not reachable at ${BASE_URL}`);
  }

  console.log('Backend health check passed');

  return {
    startTime: new Date().toISOString(),
    baseUrl: BASE_URL,
  };
}

/**
 * Teardown function - runs once after the test
 */
export function teardown(data) {
  console.log(`Load test completed`);
  console.log(`Started: ${data.startTime}`);
  console.log(`Finished: ${new Date().toISOString()}`);
}

/**
 * Handle summary - custom summary output
 */
export function handleSummary(data) {
  const summary = {
    timestamp: new Date().toISOString(),
    baseUrl: BASE_URL,
    metrics: {
      // Overall metrics
      totalRequests: data.metrics.http_reqs?.values?.count || 0,
      failedRequests: data.metrics.http_req_failed?.values?.passes || 0,

      // Latency metrics
      httpReqDuration: {
        avg: data.metrics.http_req_duration?.values?.avg || 0,
        p50: data.metrics.http_req_duration?.values?.['p(50)'] || 0,
        p95: data.metrics.http_req_duration?.values?.['p(95)'] || 0,
        p99: data.metrics.http_req_duration?.values?.['p(99)'] || 0,
        max: data.metrics.http_req_duration?.values?.max || 0,
      },

      // Chat-specific metrics (NFR-001)
      chatLatency: {
        avg: data.metrics.chat_latency?.values?.avg || 0,
        p95: data.metrics.chat_latency?.values?.['p(95)'] || 0,
        p99: data.metrics.chat_latency?.values?.['p(99)'] || 0,
        max: data.metrics.chat_latency?.values?.max || 0,
      },

      // Error rate
      errorRate: data.metrics.errors?.values?.rate || 0,
    },
    thresholds: {
      passed: Object.values(data.root_group?.checks || {}).filter((c) => c.passes > 0).length,
      failed: Object.values(data.root_group?.checks || {}).filter((c) => c.fails > 0).length,
    },
  };

  // NFR-001 validation
  const chatP95 = summary.metrics.chatLatency.p95;
  const nfr001Passed = chatP95 < 3000;

  console.log('\n' + '='.repeat(60));
  console.log('LOAD TEST SUMMARY');
  console.log('='.repeat(60));
  console.log(`Total Requests: ${summary.metrics.totalRequests}`);
  console.log(`Error Rate: ${(summary.metrics.errorRate * 100).toFixed(2)}%`);
  console.log(`\nHTTP Request Duration:`);
  console.log(`  Average: ${summary.metrics.httpReqDuration.avg.toFixed(0)}ms`);
  console.log(`  P95: ${summary.metrics.httpReqDuration.p95.toFixed(0)}ms`);
  console.log(`  P99: ${summary.metrics.httpReqDuration.p99.toFixed(0)}ms`);
  console.log(`\nChat Latency (NFR-001 Target: p95 < 3000ms):`);
  console.log(`  P95: ${chatP95.toFixed(0)}ms ${nfr001Passed ? '✓ PASS' : '✗ FAIL'}`);
  console.log(`  P99: ${summary.metrics.chatLatency.p99.toFixed(0)}ms`);
  console.log('='.repeat(60));

  return {
    'tests/load/results.json': JSON.stringify(summary, null, 2),
    stdout: '', // Suppress default stdout output
  };
}
