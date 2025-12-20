import { test, expect } from '@playwright/test';

test.describe('Navigation', () => {
  test('homepage loads correctly', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveTitle(/Physical AI/);
  });

  test('can navigate to Chapter 1', async ({ page }) => {
    await page.goto('/');
    await page.click('text=ROS 2');
    await expect(page.locator('h1')).toContainText(/Welcome|ROS 2/);
  });

  test('sidebar navigation works', async ({ page }) => {
    await page.goto('/module-1-ros2/ch01-welcome-first-node');
    await expect(page.locator('.menu__list')).toBeVisible();
  });

  test('chapter pagination works', async ({ page }) => {
    await page.goto('/module-1-ros2/ch01-welcome-first-node');
    const nextLink = page.locator('.pagination-nav__link--next');
    await expect(nextLink).toBeVisible();
  });
});

test.describe('Content', () => {
  test('code blocks render correctly', async ({ page }) => {
    await page.goto('/module-1-ros2/ch01-welcome-first-node');
    await expect(page.locator('pre code')).toBeVisible();
  });

  test('learning objectives visible', async ({ page }) => {
    await page.goto('/module-1-ros2/ch01-welcome-first-node');
    await expect(page.locator('text=Learning Objectives')).toBeVisible();
  });
});

test.describe('Accessibility', () => {
  test('has proper heading hierarchy', async ({ page }) => {
    await page.goto('/module-1-ros2/ch01-welcome-first-node');
    const h1 = await page.locator('h1').count();
    expect(h1).toBe(1);
  });

  test('images have alt text', async ({ page }) => {
    await page.goto('/');
    const images = page.locator('img');
    const count = await images.count();
    for (let i = 0; i < count; i++) {
      const alt = await images.nth(i).getAttribute('alt');
      expect(alt).toBeTruthy();
    }
  });
});

test.describe('RTL Support', () => {
  test('Urdu locale switches to RTL', async ({ page }) => {
    await page.goto('/ur/');
    const html = page.locator('html');
    await expect(html).toHaveAttribute('dir', 'rtl');
  });
});
