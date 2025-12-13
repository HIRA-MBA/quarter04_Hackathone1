import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'Embodied Intelligence: From ROS 2 to Vision-Language-Action Models',
  favicon: 'img/favicon.ico',

  // Custom head tags for Urdu font support
  headTags: [
    {
      tagName: 'link',
      attributes: {
        rel: 'preconnect',
        href: 'https://fonts.googleapis.com',
      },
    },
    {
      tagName: 'link',
      attributes: {
        rel: 'preconnect',
        href: 'https://fonts.gstatic.com',
        crossorigin: 'anonymous',
      },
    },
    {
      tagName: 'link',
      attributes: {
        rel: 'stylesheet',
        href: 'https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;500;600;700&display=swap',
      },
    },
  ],

  future: {
    v4: true,
  },

  url: 'https://physical-ai-textbook.vercel.app',
  baseUrl: '/',

  organizationName: 'physical-ai-textbook',
  projectName: 'physical-ai-textbook',

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'ur'],
    localeConfigs: {
      en: {
        label: 'English',
        direction: 'ltr',
      },
      ur: {
        label: 'اردو',
        direction: 'rtl',
      },
    },
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          routeBasePath: '/',
          showLastUpdateTime: true,
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/social-card.jpg',
    colorMode: {
      defaultMode: 'light',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'Physical AI Textbook',
      logo: {
        alt: 'Physical AI Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'frontMatterSidebar',
          position: 'left',
          label: 'Start Here',
        },
        {
          type: 'docSidebar',
          sidebarId: 'module1Sidebar',
          position: 'left',
          label: 'ROS 2',
        },
        {
          type: 'docSidebar',
          sidebarId: 'module2Sidebar',
          position: 'left',
          label: 'Digital Twin',
        },
        {
          type: 'docSidebar',
          sidebarId: 'module3Sidebar',
          position: 'left',
          label: 'Isaac',
        },
        {
          type: 'docSidebar',
          sidebarId: 'module4Sidebar',
          position: 'left',
          label: 'VLA',
        },
        {
          type: 'docSidebar',
          sidebarId: 'backMatterSidebar',
          position: 'left',
          label: 'Reference',
        },
        {
          type: 'localeDropdown',
          position: 'right',
        },
        {
          href: 'https://github.com/physical-ai-textbook/physical-ai-textbook',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Modules',
          items: [
            {
              label: 'Module 1: ROS 2 Fundamentals',
              to: '/module-1-ros2/ch01-welcome-first-node',
            },
            {
              label: 'Module 2: Digital Twin',
              to: '/module-2-digital-twin/ch06-gazebo-physics',
            },
            {
              label: 'Module 3: NVIDIA Isaac',
              to: '/module-3-isaac/ch08-isaac-sim',
            },
            {
              label: 'Module 4: VLA & Capstone',
              to: '/module-4-vla/ch11-humanoid-locomotion',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'Hardware Setup',
              to: '/front-matter/hardware-lab-setup',
            },
            {
              label: 'Glossary',
              to: '/back-matter/glossary',
            },
            {
              label: 'Troubleshooting',
              to: '/back-matter/appendix-d-troubleshooting',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/physical-ai-textbook/physical-ai-textbook',
            },
            {
              label: 'Discussions',
              href: 'https://github.com/physical-ai-textbook/physical-ai-textbook/discussions',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Textbook. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'yaml', 'cmake', 'markup'],
    },
    tableOfContents: {
      minHeadingLevel: 2,
      maxHeadingLevel: 4,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
