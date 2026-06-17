import { describe, it, expect } from 'vitest';
import {
  createGuitarBranch,
  createInitialProjectScoreState,
  createPianoBranch,
  getActiveRenderView,
  getActiveScoreBranch,
  getEventId,
  getMeasureId,
  getPianoRevisionId,
  isGuitarBranchStale,
  toHitKey,
  updateGuitarBranch,
  updatePianoBranch,
} from './types';
import type { HitKey, ScoreDocumentBundle, ProjectScoreState } from './types';

describe('toHitKey', () => {
  it('produces a canonical pipe-delimited key with all fields', () => {
    const hit: HitKey = { barIndex: 2, voiceIndex: 1, beatIndex: 0, noteIndex: 3 };
    expect(toHitKey(hit)).toBe('2|1|0|3');
  });

  it('fills missing optional fields with -1', () => {
    expect(toHitKey({ barIndex: 0 })).toBe('0|-1|-1|-1');
  });
});

describe('measure and event map helpers', () => {
  it('returns the measure id for a known bar index', () => {
    expect(getMeasureId({ '0': 'measure-a', '1': 'measure-b' }, 1)).toBe('measure-b');
  });

  it('returns null for missing measure or event maps', () => {
    expect(getMeasureId(null, 0)).toBeNull();
    expect(getEventId(null, { barIndex: 0 })).toBeNull();
  });

  it('returns the event id for a matching hit key', () => {
    expect(getEventId({ '1|0|-1|-1': 'evt-42' }, { barIndex: 1, voiceIndex: 0 })).toBe('evt-42');
  });
});

describe('getActiveRenderView', () => {
  const guitarDocument: ScoreDocumentBundle = {
    instrumentMode: 'guitar',
    views: {
      score: { xml: '<score/>', measureMap: { '0': 'score-measure' } },
      tab: { xml: '<tab/>', measureMap: { '0': 'tab-measure' } },
    },
  };

  const pianoDocument: ScoreDocumentBundle = {
    instrumentMode: 'piano',
    views: {
      score: { xml: '<piano-score/>' },
    },
  };

  it('returns the score view for piano regardless of requested tab', () => {
    expect(getActiveRenderView(pianoDocument, 'tab')?.xml).toBe('<piano-score/>');
  });

  it('returns the explicit tab view for guitar tab mode', () => {
    expect(getActiveRenderView(guitarDocument, 'tab')?.xml).toBe('<tab/>');
  });

  it('returns the score view for guitar score mode', () => {
    expect(getActiveRenderView(guitarDocument, 'score')?.xml).toBe('<score/>');
  });
});

describe('ProjectScoreState branch fields', () => {
  const base = createInitialProjectScoreState();

  it('starts on an empty piano branch with no guitar branch', () => {
    expect(base.activeBranch).toBe('piano');
    expect(base.piano.changedMeasureIds).toBeNull();
    expect(base.guitar).toBeNull();
  });

  it('accepts a draft document bundle on a specific branch', () => {
    const state: ProjectScoreState = {
      ...base,
      piano: createPianoBranch({
        draftDocument: {
          instrumentMode: 'piano',
          views: {
            score: { xml: '<score/>' },
          },
        },
      }),
    };
    expect(state.piano.draftDocument?.views.score.xml).toBe('<score/>');
  });

  it('returns null for an unconverted active guitar branch', () => {
    const state: ProjectScoreState = {
      ...base,
      activeBranch: 'guitar',
    };
    expect(getActiveScoreBranch(state)).toBeNull();
  });

  it('stores source metadata on a converted guitar branch', () => {
    const state: ProjectScoreState = {
      ...base,
      activeBranch: 'guitar',
      guitar: createGuitarBranch({
        sourcePianoRevisionId: 'piano-rev-4',
        diagnostics: { warnings: ['octave shifted'] },
        sourceMap: [{ pianoEventId: 'p1', guitarEventId: 'g1' }],
      }),
    };

    expect(getActiveScoreBranch(state)?.instrumentMode).toBe('guitar');
    expect(state.guitar?.sourcePianoRevisionId).toBe('piano-rev-4');
    expect(state.guitar?.diagnostics?.warnings).toEqual(['octave shifted']);
    expect(state.guitar?.sourceMap).toEqual([{ pianoEventId: 'p1', guitarEventId: 'g1' }]);
  });

  it('updates piano without replacing guitar and detects stale provenance', () => {
    const guitar = createGuitarBranch({
      sourcePianoRevisionId: 'score-piano@2',
    });
    const state: ProjectScoreState = {
      ...base,
      piano: createPianoBranch({
        scoreId: 'score-piano',
        revision: 2,
      }),
      guitar,
    };

    const edited = updatePianoBranch(state, (piano) => ({
      ...piano,
      revision: 3,
    }));

    expect(getPianoRevisionId(edited.piano)).toBe('score-piano@3');
    expect(edited.guitar).toBe(guitar);
    expect(isGuitarBranchStale(edited)).toBe(true);
  });

  it('updates guitar without replacing piano', () => {
    const piano = createPianoBranch({
      scoreId: 'score-piano',
      revision: 2,
    });
    const state: ProjectScoreState = {
      ...base,
      piano,
      guitar: createGuitarBranch({
        revision: 0,
        sourcePianoRevisionId: 'score-piano@2',
      }),
    };

    const edited = updateGuitarBranch(state, (guitar) => ({
      ...guitar,
      revision: 1,
    }));

    expect(edited.piano).toBe(piano);
    expect(edited.guitar?.revision).toBe(1);
    expect(isGuitarBranchStale(edited)).toBe(false);
  });
});
