import { describe, expect, it } from 'vitest';
import {
  getScoreViewerLabel,
  getStaffVisibility,
  normalizeViewTab,
  resolveStaveProfile,
  shouldShowTabSwitcher,
} from './scoreViewerStaves';

describe('resolveStaveProfile', () => {
  const alphaTabModule = {
    StaveProfile: {
      Default: 7,
      Score: 11,
      ScoreTab: 22,
      Tab: 33,
    },
  };

  it('prefers the default stave profile for imported scores', () => {
    expect(resolveStaveProfile(alphaTabModule, 'guitar', 'tab')).toBe(7);
    expect(resolveStaveProfile(alphaTabModule, 'guitar', 'score')).toBe(7);
    expect(resolveStaveProfile(alphaTabModule, 'piano', 'tab')).toBe(7);
  });

  it('falls back to layout stave profiles when needed', () => {
    expect(
      resolveStaveProfile(
        {
          StaveProfile: {},
          LayoutStaveProfile: {
            Score: 5,
          },
        },
        'guitar',
        'tab',
      ),
    ).toBe(5);
  });
});

describe('score viewer mode helpers', () => {
  it('normalizes piano tabs back to score mode', () => {
    expect(normalizeViewTab('piano', 'tab')).toBe('score');
  });

  it('shows the tab switcher only for guitar', () => {
    expect(shouldShowTabSwitcher('guitar')).toBe(true);
    expect(shouldShowTabSwitcher('piano')).toBe(false);
  });

  it('returns a label that reflects the active guitar tab view', () => {
    expect(getScoreViewerLabel('guitar', 'tab')).toBe('Guitar Tab');
  });

  it('returns staff visibility flags for guitar sheet music', () => {
    expect(getStaffVisibility('guitar', 'score')).toEqual({
      showStandardNotation: true,
      showTablature: false,
    });
  });

  it('returns staff visibility flags for guitar tab view', () => {
    expect(getStaffVisibility('guitar', 'tab')).toEqual({
      showStandardNotation: false,
      showTablature: true,
    });
  });

  it('returns score-only staff visibility for piano', () => {
    expect(getStaffVisibility('piano', 'tab')).toEqual({
      showStandardNotation: true,
      showTablature: false,
    });
  });
});
