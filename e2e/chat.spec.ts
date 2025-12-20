import { test, expect } from '@playwright/test';

test.describe('RAG Chatbot', () => {
  test('chat widget is visible', async ({ page }) => {
    await page.goto('/module-1-ros2/ch01-welcome-first-node');
    await expect(page.locator('[data-testid="chat-widget"]')).toBeVisible();
  });

  test('chat opens on click', async ({ page }) => {
    await page.goto('/module-1-ros2/ch01-welcome-first-node');
    await page.click('[data-testid="chat-widget"]');
    await expect(page.locator('[data-testid="chat-panel"]')).toBeVisible();
  });

  test('can send message', async ({ page }) => {
    await page.goto('/module-1-ros2/ch01-welcome-first-node');
    await page.click('[data-testid="chat-widget"]');
    await page.fill('[data-testid="chat-input"]', 'What is ROS 2?');
    await page.click('[data-testid="chat-send"]');
    await expect(page.locator('[data-testid="chat-message"]')).toBeVisible({ timeout: 10000 });
  });
});
