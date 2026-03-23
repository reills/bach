import { describe, it, expect } from 'vitest';
import { getActiveRenderView, getEventId, getMeasureId, toHitKey } from './types';
import type { HitKey, ScoreDocumentBundle, ScoreState } from './types';

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

describe('ScoreState draft bundle fields', () => {
  const base: ScoreState = {
    scoreId: null,
    revision: null,
    document: null,
    draftId: null,
    draftDocument: null,
    draftBaseRevision: null,
    highlightMeasureId: null,
    selectedMeasureId: null,
    selectedBarIndex: null,
    lockedEventIds: null,
    changedMeasureIds: null,
    lastEventId: null,
    midi: null,
    instrumentMode: null,
  };

  it('defaults changedMeasureIds to null', () => {
    expect(base.changedMeasureIds).toBeNull();
  });

  it('accepts a draft document bundle', () => {
    const state: ScoreState = {
      ...base,
      draftDocument: {
        instrumentMode: 'guitar',
        views: {
          score: { xml: '<score/>' },
        },
      },
    };
    expect(state.draftDocument?.views.score.xml).toBe('<score/>');
  });
});
