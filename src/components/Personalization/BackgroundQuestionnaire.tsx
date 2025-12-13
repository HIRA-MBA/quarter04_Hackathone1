/**
 * Multi-step background questionnaire for user personalization.
 */

import React, { useState } from 'react';
import { useHistory } from '@docusaurus/router';
import styles from '../../pages/auth/auth.module.css';
import { userApi, QuestionnaireData } from '../../services/user';

interface Step {
  id: string;
  title: string;
  description: string;
}

const STEPS: Step[] = [
  {
    id: 'experience',
    title: 'Experience Level',
    description: 'What is your current experience with robotics and programming?',
  },
  {
    id: 'background',
    title: 'Your Background',
    description: 'Tell us about your educational and professional background.',
  },
  {
    id: 'goals',
    title: 'Learning Goals',
    description: 'What do you hope to achieve with this course?',
  },
];

const EXPERIENCE_OPTIONS = [
  {
    value: 'beginner',
    label: 'Beginner',
    description: 'New to robotics, some programming experience',
  },
  {
    value: 'intermediate',
    label: 'Intermediate',
    description: 'Familiar with ROS or robotics concepts',
  },
  {
    value: 'advanced',
    label: 'Advanced',
    description: 'Professional experience in robotics or AI',
  },
];

const LEARNING_STYLE_OPTIONS = [
  { value: 'visual', label: 'Visual learner (diagrams, videos)' },
  { value: 'hands-on', label: 'Hands-on learner (labs, projects)' },
  { value: 'reading', label: 'Reading-focused (detailed text)' },
  { value: 'mixed', label: 'Mixed approach' },
];

export default function BackgroundQuestionnaire(): React.ReactElement {
  const history = useHistory();
  const [currentStep, setCurrentStep] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');

  const [formData, setFormData] = useState<QuestionnaireData>({
    experience_level: '' as 'beginner' | 'intermediate' | 'advanced',
    background: '',
    goals: '',
    programming_experience: '',
    robotics_experience: '',
    preferred_learning_style: '',
  });

  const handleExperienceSelect = (value: 'beginner' | 'intermediate' | 'advanced') => {
    setFormData({ ...formData, experience_level: value });
  };

  const handleInputChange = (field: keyof QuestionnaireData, value: string) => {
    setFormData({ ...formData, [field]: value });
  };

  const canProceed = (): boolean => {
    switch (currentStep) {
      case 0:
        return !!formData.experience_level;
      case 1:
        return formData.background.length >= 20;
      case 2:
        return formData.goals.length >= 20;
      default:
        return true;
    }
  };

  const handleNext = () => {
    if (currentStep < STEPS.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handleBack = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleSubmit = async () => {
    setIsSubmitting(true);
    setError('');

    try {
      await userApi.submitQuestionnaire(formData);
      history.push('/');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save preferences');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleSkip = () => {
    history.push('/');
  };

  const renderStep = () => {
    switch (currentStep) {
      case 0:
        return (
          <div className={styles.options}>
            {EXPERIENCE_OPTIONS.map((option) => (
              <button
                key={option.value}
                type="button"
                className={`${styles.optionButton} ${
                  formData.experience_level === option.value ? styles.selected : ''
                }`}
                onClick={() => handleExperienceSelect(option.value as 'beginner' | 'intermediate' | 'advanced')}
              >
                <div className={styles.optionContent}>
                  <strong>{option.label}</strong>
                  <span>{option.description}</span>
                </div>
              </button>
            ))}
          </div>
        );

      case 1:
        return (
          <div className={styles.form}>
            <div className={styles.formGroup}>
              <label htmlFor="background">Educational/Professional Background</label>
              <textarea
                id="background"
                value={formData.background}
                onChange={(e) => handleInputChange('background', e.target.value)}
                placeholder="e.g., Computer Science student, Mechanical Engineer with 3 years experience..."
                rows={3}
              />
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="programmingExp">Programming Languages (optional)</label>
              <input
                type="text"
                id="programmingExp"
                value={formData.programming_experience || ''}
                onChange={(e) => handleInputChange('programming_experience', e.target.value)}
                placeholder="e.g., Python, C++, ROS 2"
              />
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="roboticsExp">Previous Robotics Experience (optional)</label>
              <input
                type="text"
                id="roboticsExp"
                value={formData.robotics_experience || ''}
                onChange={(e) => handleInputChange('robotics_experience', e.target.value)}
                placeholder="e.g., Arduino projects, ROS 1, simulation work"
              />
            </div>
          </div>
        );

      case 2:
        return (
          <div className={styles.form}>
            <div className={styles.formGroup}>
              <label htmlFor="goals">What are your learning goals?</label>
              <textarea
                id="goals"
                value={formData.goals}
                onChange={(e) => handleInputChange('goals', e.target.value)}
                placeholder="e.g., Build a humanoid robot for my thesis, transition to robotics career, understand physical AI concepts..."
                rows={4}
              />
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="learningStyle">Preferred Learning Style (optional)</label>
              <select
                id="learningStyle"
                value={formData.preferred_learning_style || ''}
                onChange={(e) => handleInputChange('preferred_learning_style', e.target.value)}
              >
                <option value="">Select your preference</option>
                {LEARNING_STYLE_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className={styles.authCard} style={{ maxWidth: '520px' }}>
      <h1>Personalize Your Experience</h1>
      <p className={styles.subtitle}>
        Help us tailor the content to your needs. This takes about 2 minutes.
      </p>

      {/* Step indicator */}
      <div className={styles.stepIndicator}>
        {STEPS.map((step, index) => (
          <div
            key={step.id}
            className={`${styles.step} ${
              index === currentStep ? styles.active : ''
            } ${index < currentStep ? styles.completed : ''}`}
          />
        ))}
      </div>

      {/* Current step info */}
      <div style={{ marginBottom: '1.5rem' }}>
        <h3 style={{ margin: '0 0 0.5rem' }}>{STEPS[currentStep].title}</h3>
        <p style={{ color: 'var(--ifm-color-emphasis-600)', margin: 0 }}>
          {STEPS[currentStep].description}
        </p>
      </div>

      {error && <div className={styles.error}>{error}</div>}

      {renderStep()}

      {/* Navigation buttons */}
      <div className={styles.navigationButtons}>
        {currentStep > 0 && (
          <button
            type="button"
            className={styles.backButton}
            onClick={handleBack}
          >
            Back
          </button>
        )}

        {currentStep < STEPS.length - 1 ? (
          <button
            type="button"
            className={`${styles.submitButton} ${styles.nextButton}`}
            onClick={handleNext}
            disabled={!canProceed()}
          >
            Continue
          </button>
        ) : (
          <button
            type="button"
            className={`${styles.submitButton} ${styles.nextButton}`}
            onClick={handleSubmit}
            disabled={!canProceed() || isSubmitting}
          >
            {isSubmitting ? 'Saving...' : 'Complete Setup'}
          </button>
        )}
      </div>

      {/* Skip option */}
      <p style={{ textAlign: 'center', marginTop: '1rem' }}>
        <button
          type="button"
          onClick={handleSkip}
          style={{
            background: 'none',
            border: 'none',
            color: 'var(--ifm-color-emphasis-600)',
            cursor: 'pointer',
            textDecoration: 'underline',
          }}
        >
          Skip for now
        </button>
      </p>
    </div>
  );
}
