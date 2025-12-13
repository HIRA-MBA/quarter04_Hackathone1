import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  frontMatterSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: [
        'front-matter/foreword',
        'front-matter/preface',
        'front-matter/introduction',
        'front-matter/hardware-lab-setup',
      ],
    },
  ],

  module1Sidebar: [
    {
      type: 'category',
      label: 'Module 1: ROS 2 Fundamentals',
      collapsed: false,
      items: [
        'module-1-ros2/ch01-welcome-first-node',
        'module-1-ros2/ch02-sensors-perception',
        'module-1-ros2/ch03-ros2-architecture',
        'module-1-ros2/ch04-urdf-humanoid',
        'module-1-ros2/ch05-edge-capstone',
      ],
    },
  ],

  module2Sidebar: [
    {
      type: 'category',
      label: 'Module 2: Digital Twin',
      collapsed: false,
      items: [
        'module-2-digital-twin/ch06-gazebo-physics',
        'module-2-digital-twin/ch07-unity-capstone',
      ],
    },
  ],

  module3Sidebar: [
    {
      type: 'category',
      label: 'Module 3: NVIDIA Isaac',
      collapsed: false,
      items: [
        'module-3-isaac/ch08-isaac-sim',
        'module-3-isaac/ch09-isaac-ros-gpu',
        'module-3-isaac/ch10-nav-rl-sim2real',
      ],
    },
  ],

  module4Sidebar: [
    {
      type: 'category',
      label: 'Module 4: VLA & Capstone',
      collapsed: false,
      items: [
        'module-4-vla/ch11-humanoid-locomotion',
        'module-4-vla/ch12-dexterous-manipulation',
        'module-4-vla/ch13-vision-language-action',
        'module-4-vla/ch14-capstone-humanoid',
      ],
    },
  ],

  backMatterSidebar: [
    {
      type: 'category',
      label: 'Appendices',
      collapsed: false,
      items: [
        'back-matter/appendix-a-student-kit',
        'back-matter/appendix-b-cloud-setup',
        'back-matter/appendix-c-repos-docker',
        'back-matter/appendix-d-troubleshooting',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      collapsed: false,
      items: [
        'back-matter/safety-ethics',
        'back-matter/glossary',
        'back-matter/index',
      ],
    },
  ],
};

export default sidebars;
