import { describe, expect, it } from 'vitest';
import { resolveStaveProfile } from './scoreViewerStaves';

describe('resolveStaveProfile', () => {
  const alphaTabModule = {
    StaveProfile: {
      Score: 11,
      Tab: 22,
    },
  };

  it('uses tab profile only for guitar tab mode', () => {
    expect(resolveStaveProfile(alphaTabModule, 'guitar', 'tab')).toBe(22);
  });

  it('uses score profile for guitar score mode', () => {
    expect(resolveStaveProfile(alphaTabModule, 'guitar', 'score')).toBe(11);
  });

  it('uses score profile for piano even if the tab toggle state is tab', () => {
    expect(resolveStaveProfile(alphaTabModule, 'piano', 'tab')).toBe(11);
  });

  it('falls back to layout stave profiles when needed', () => {
    expect(
      resolveStaveProfile(
        {
          LayoutStaveProfile: {
            Score: 5,
            Tab: 6,
          },
        },
        'guitar',
        'tab',
      ),
    ).toBe(6);
  });
});
