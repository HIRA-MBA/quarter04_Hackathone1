import { test, expect } from '@playwright/test';

test.describe('Authentication', () => {
  test('signin page loads', async ({ page }) => {
    await page.goto('/auth/signin');
    await expect(page.locator('text=Sign In')).toBeVisible();
  });

  test('OAuth buttons visible', async ({ page }) => {
    await page.goto('/auth/signin');
    await expect(page.locator('text=Google')).toBeVisible();
    await expect(page.locator('text=GitHub')).toBeVisible();
  });

  test('redirects unauthenticated users', async ({ page }) => {
    await page.goto('/profile');
    await expect(page).toHaveURL(/signin/);
  });
});
