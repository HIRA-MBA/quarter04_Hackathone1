/**
 * Custom navbar item types for Docusaurus.
 * This file registers custom navbar components.
 */

import ComponentTypes from '@theme-original/NavbarItem/ComponentTypes';
import SignInButton from '@site/src/components/Auth/SignInButton';

export default {
  ...ComponentTypes,
  'custom-signInButton': SignInButton,
};
