/**
 * Questionnaire page that wraps the BackgroundQuestionnaire component.
 */

import React from 'react';
import Layout from '@theme/Layout';
import BackgroundQuestionnaire from '../../components/Personalization/BackgroundQuestionnaire';
import styles from './auth.module.css';

export default function QuestionnairePage(): React.ReactElement {
  return (
    <Layout title="Complete Your Profile" description="Set up your learning preferences">
      <div className={styles.authContainer}>
        <BackgroundQuestionnaire />
      </div>
    </Layout>
  );
}
