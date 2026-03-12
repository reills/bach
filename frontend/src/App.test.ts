/**
 * Integration tests for the compose/inpaint workflow.
 *
 * Tests cover state transitions for the main flow:
 *   compose → select measure → inpaint preview → commit or discard draft
 *
 * The API layer is fully mocked; no live backend is needed.
 * State objects are built and mutated the same way App.tsx handlers do,
 * keeping the test surface focused on state transitions.
 */
import { vi, describe, it, expect, beforeEach } from 'vitest';
import {
  compose,
  inpaintPreview,
  commitDraft,
  discardDraft,
} from './api/client';
import { getMeasureId, getEventId } from './state/types';
import type { ScoreState, MeasureMap, EventHitMap } from './state/types';

vi.mock('./api/client');

const initialState: ScoreState = {
  scoreId: null,
  revision: null,
  scoreXml: null,
  measureMap: null,
  eventHitMap: null,
  draftId: null,
  draftXml: null,
  draftBaseRevision: null,
  highlightMeasureId: null,
  selectedMeasureId: null,
  selectedBarIndex: null,
  lockedEventIds: null,
  changedMeasureIds: null,
  lastEventId: null,
  midi: null,
};

// Minimal helpers that mirror the App.tsx handler state transforms.
function applyComposeResponse(
  state: ScoreState,
  response: {
    scoreId: string;
    revision: number;
    scoreXML: string;
    measureMap?: MeasureMap;
    eventHitMap?: EventHitMap;
    midi?: string;
  },
): ScoreState {
  return {
    ...state,
    scoreId: response.scoreId,
    revision: response.revision,
    scoreXml: response.scoreXML,
    measureMap: response.measureMap ?? null,
    eventHitMap: response.eventHitMap ?? null,
    midi: response.midi ?? null,
    draftId: null,
    draftXml: null,
    draftBaseRevision: null,
    highlightMeasureId: null,
    selectedMeasureId: null,
    selectedBarIndex: null,
    lockedEventIds: null,
    changedMeasureIds: null,
    lastEventId: null,
  };
}

function applyMeasureSelect(state: ScoreState, barIndex: number): ScoreState {
  const measureId = getMeasureId(state.measureMap, barIndex);
  return {
    ...state,
    selectedBarIndex: barIndex,
    selectedMeasureId: measureId,
  };
}

function applyInpaintPreviewResponse(
  state: ScoreState,
  response: {
    draftId: string;
    scoreXML: string;
    baseRevision: number;
    highlightMeasureId?: string;
    measureMap?: MeasureMap;
    eventHitMap?: EventHitMap;
    lockedEventIds?: string[];
    changedMeasureIds?: string[];
  },
): ScoreState {
  return {
    ...state,
    draftId: response.draftId,
    draftXml: response.scoreXML,
    draftBaseRevision: response.baseRevision,
    highlightMeasureId: response.highlightMeasureId ?? null,
    measureMap: response.measureMap ?? state.measureMap,
    eventHitMap: response.eventHitMap ?? state.eventHitMap,
    lockedEventIds: response.lockedEventIds ?? null,
    changedMeasureIds: response.changedMeasureIds ?? null,
  };
}

function applyCommitResponse(
  state: ScoreState,
  response: {
    scoreXML: string;
    revision: number;
    measureMap?: MeasureMap;
    eventHitMap?: EventHitMap;
  },
): ScoreState {
  return {
    ...state,
    scoreXml: response.scoreXML,
    revision: response.revision,
    measureMap: response.measureMap ?? state.measureMap,
    eventHitMap: response.eventHitMap ?? state.eventHitMap,
    draftId: null,
    draftXml: null,
    draftBaseRevision: null,
    highlightMeasureId: null,
    lockedEventIds: null,
    changedMeasureIds: null,
  };
}

function applyDiscardDraft(state: ScoreState): ScoreState {
  return {
    ...state,
    draftId: null,
    draftXml: null,
    draftBaseRevision: null,
    highlightMeasureId: null,
    lockedEventIds: null,
    changedMeasureIds: null,
  };
}

const MINIMAL_XML = '<score-partwise version="3.1"/>';
const DRAFT_XML = '<score-partwise version="3.1"><!-- draft --></score-partwise>';
const COMMITTED_XML = '<score-partwise version="3.1"><!-- committed --></score-partwise>';

describe('compose/inpaint workflow — state transitions', () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it('compose: sets scoreId, revision, scoreXml, and measureMap from the API response', async () => {
    const composeResponse = {
      scoreId: 'score-abc',
      revision: 0,
      scoreXML: MINIMAL_XML,
      measureMap: { '0': 'measure-1', '1': 'measure-2' } as MeasureMap,
    };
    vi.mocked(compose).mockResolvedValue(composeResponse);

    const response = await compose({});
    const state = applyComposeResponse(initialState, response);

    expect(state.scoreId).toBe('score-abc');
    expect(state.revision).toBe(0);
    expect(state.scoreXml).toBe(MINIMAL_XML);
    expect(state.measureMap).toEqual({ '0': 'measure-1', '1': 'measure-2' });
    // Draft fields should all be cleared
    expect(state.draftId).toBeNull();
    expect(state.draftXml).toBeNull();
    expect(state.selectedMeasureId).toBeNull();
  });

  it('compose: clears all draft and selection fields from a previous session', async () => {
    const dirtyState: ScoreState = {
      ...initialState,
      scoreId: 'old-score',
      revision: 3,
      scoreXml: '<old/>',
      draftId: 'old-draft',
      draftXml: '<old-draft/>',
      selectedMeasureId: 'old-measure',
      selectedBarIndex: 1,
      lockedEventIds: ['evt-x'],
      changedMeasureIds: ['m1'],
    };

    const composeResponse = {
      scoreId: 'score-new',
      revision: 0,
      scoreXML: MINIMAL_XML,
      measureMap: { '0': 'measure-A' } as MeasureMap,
    };
    vi.mocked(compose).mockResolvedValue(composeResponse);

    const response = await compose({});
    const state = applyComposeResponse(dirtyState, response);

    expect(state.scoreId).toBe('score-new');
    expect(state.draftId).toBeNull();
    expect(state.selectedMeasureId).toBeNull();
    expect(state.lockedEventIds).toBeNull();
    expect(state.changedMeasureIds).toBeNull();
  });

  it('select measure: resolves selectedMeasureId from measureMap via barIndex', () => {
    const state: ScoreState = {
      ...initialState,
      scoreId: 'score-abc',
      revision: 0,
      scoreXml: MINIMAL_XML,
      measureMap: { '0': 'measure-1', '1': 'measure-2' },
    };

    const after = applyMeasureSelect(state, 1);

    expect(after.selectedBarIndex).toBe(1);
    expect(after.selectedMeasureId).toBe('measure-2');
  });

  it('select measure: sets selectedMeasureId to null when measureMap is absent', () => {
    const state: ScoreState = { ...initialState };
    const after = applyMeasureSelect(state, 0);

    expect(after.selectedBarIndex).toBe(0);
    expect(after.selectedMeasureId).toBeNull();
  });

  it('inpaint preview: sets draftId, draftXml, draftBaseRevision from the API response', async () => {
    const previewResponse = {
      draftId: 'draft-xyz',
      scoreXML: DRAFT_XML,
      baseRevision: 0,
      highlightMeasureId: 'measure-1',
      lockedEventIds: ['evt-1', 'evt-2'],
      changedMeasureIds: ['measure-1'],
    };
    vi.mocked(inpaintPreview).mockResolvedValue(previewResponse);

    const scoreState: ScoreState = {
      ...initialState,
      scoreId: 'score-abc',
      revision: 0,
      scoreXml: MINIMAL_XML,
      measureMap: { '0': 'measure-1' },
      selectedMeasureId: 'measure-1',
      selectedBarIndex: 0,
    };

    const response = await inpaintPreview({
      scoreId: scoreState.scoreId!,
      measureId: scoreState.selectedMeasureId!,
      revision: scoreState.revision!,
    });
    const state = applyInpaintPreviewResponse(scoreState, response);

    expect(state.draftId).toBe('draft-xyz');
    expect(state.draftXml).toBe(DRAFT_XML);
    expect(state.draftBaseRevision).toBe(0);
    expect(state.highlightMeasureId).toBe('measure-1');
    expect(state.lockedEventIds).toEqual(['evt-1', 'evt-2']);
    expect(state.changedMeasureIds).toEqual(['measure-1']);
    // score XML unchanged until commit
    expect(state.scoreXml).toBe(MINIMAL_XML);
  });

  it('inpaint preview: retains existing measureMap when response omits one', async () => {
    const existingMeasureMap: MeasureMap = { '0': 'measure-1' };
    const previewResponse = {
      draftId: 'draft-xyz',
      scoreXML: DRAFT_XML,
      baseRevision: 0,
    };
    vi.mocked(inpaintPreview).mockResolvedValue(previewResponse);

    const scoreState: ScoreState = {
      ...initialState,
      scoreId: 'score-abc',
      revision: 0,
      scoreXml: MINIMAL_XML,
      measureMap: existingMeasureMap,
      selectedMeasureId: 'measure-1',
      selectedBarIndex: 0,
    };

    const response = await inpaintPreview({
      scoreId: scoreState.scoreId!,
      measureId: scoreState.selectedMeasureId!,
      revision: scoreState.revision!,
    });
    const state = applyInpaintPreviewResponse(scoreState, response);

    expect(state.measureMap).toEqual(existingMeasureMap);
  });

  it('commit draft: applies new scoreXml, increments revision, and clears all draft fields', async () => {
    const commitResponse = {
      scoreXML: COMMITTED_XML,
      revision: 1,
      measureMap: { '0': 'measure-1' } as MeasureMap,
    };
    vi.mocked(commitDraft).mockResolvedValue(commitResponse);

    const draftState: ScoreState = {
      ...initialState,
      scoreId: 'score-abc',
      revision: 0,
      scoreXml: MINIMAL_XML,
      measureMap: { '0': 'measure-1' },
      draftId: 'draft-xyz',
      draftXml: DRAFT_XML,
      draftBaseRevision: 0,
      highlightMeasureId: 'measure-1',
      lockedEventIds: ['evt-1'],
      changedMeasureIds: ['measure-1'],
    };

    const response = await commitDraft({
      scoreId: draftState.scoreId!,
      draftId: draftState.draftId!,
    });
    const state = applyCommitResponse(draftState, response);

    expect(state.scoreXml).toBe(COMMITTED_XML);
    expect(state.revision).toBe(1);
    expect(state.draftId).toBeNull();
    expect(state.draftXml).toBeNull();
    expect(state.draftBaseRevision).toBeNull();
    expect(state.highlightMeasureId).toBeNull();
    expect(state.lockedEventIds).toBeNull();
    expect(state.changedMeasureIds).toBeNull();
  });

  it('discard draft: clears all draft fields without touching scoreXml or revision', async () => {
    vi.mocked(discardDraft).mockResolvedValue({ ok: true });

    const draftState: ScoreState = {
      ...initialState,
      scoreId: 'score-abc',
      revision: 0,
      scoreXml: MINIMAL_XML,
      measureMap: { '0': 'measure-1' },
      draftId: 'draft-xyz',
      draftXml: DRAFT_XML,
      draftBaseRevision: 0,
      highlightMeasureId: 'measure-1',
      lockedEventIds: ['evt-1'],
      changedMeasureIds: ['measure-1'],
    };

    await discardDraft({
      scoreId: draftState.scoreId!,
      draftId: draftState.draftId!,
    });
    const state = applyDiscardDraft(draftState);

    // Score unchanged
    expect(state.scoreXml).toBe(MINIMAL_XML);
    expect(state.revision).toBe(0);
    // Draft cleared
    expect(state.draftId).toBeNull();
    expect(state.draftXml).toBeNull();
    expect(state.draftBaseRevision).toBeNull();
    expect(state.highlightMeasureId).toBeNull();
    expect(state.lockedEventIds).toBeNull();
    expect(state.changedMeasureIds).toBeNull();
  });

  it('full workflow: compose → select measure → inpaint preview → commit', async () => {
    const measureMap: MeasureMap = { '0': 'measure-1', '1': 'measure-2' };
    vi.mocked(compose).mockResolvedValue({
      scoreId: 'score-1',
      revision: 0,
      scoreXML: MINIMAL_XML,
      measureMap,
    });
    vi.mocked(inpaintPreview).mockResolvedValue({
      draftId: 'draft-1',
      scoreXML: DRAFT_XML,
      baseRevision: 0,
      highlightMeasureId: 'measure-1',
      changedMeasureIds: ['measure-1'],
    });
    vi.mocked(commitDraft).mockResolvedValue({
      scoreXML: COMMITTED_XML,
      revision: 1,
    });

    // Step 1: compose
    let state = applyComposeResponse(initialState, await compose({}));
    expect(state.scoreId).toBe('score-1');
    expect(state.measureMap).toEqual(measureMap);

    // Step 2: select measure 0
    state = applyMeasureSelect(state, 0);
    expect(state.selectedMeasureId).toBe('measure-1');

    // Step 3: inpaint preview
    state = applyInpaintPreviewResponse(
      state,
      await inpaintPreview({
        scoreId: state.scoreId!,
        measureId: state.selectedMeasureId!,
        revision: state.revision!,
      }),
    );
    expect(state.draftId).toBe('draft-1');
    expect(state.draftXml).toBe(DRAFT_XML);
    expect(state.changedMeasureIds).toEqual(['measure-1']);

    // Step 4: commit
    state = applyCommitResponse(
      state,
      await commitDraft({ scoreId: state.scoreId!, draftId: state.draftId! }),
    );
    expect(state.scoreXml).toBe(COMMITTED_XML);
    expect(state.revision).toBe(1);
    expect(state.draftId).toBeNull();
  });

  it('full workflow: compose → select measure → inpaint preview → discard', async () => {
    const measureMap: MeasureMap = { '0': 'measure-1' };
    vi.mocked(compose).mockResolvedValue({
      scoreId: 'score-2',
      revision: 0,
      scoreXML: MINIMAL_XML,
      measureMap,
    });
    vi.mocked(inpaintPreview).mockResolvedValue({
      draftId: 'draft-2',
      scoreXML: DRAFT_XML,
      baseRevision: 0,
    });
    vi.mocked(discardDraft).mockResolvedValue({ ok: true });

    // Compose
    let state = applyComposeResponse(initialState, await compose({}));

    // Select measure
    state = applyMeasureSelect(state, 0);
    expect(state.selectedMeasureId).toBe('measure-1');

    // Inpaint preview
    state = applyInpaintPreviewResponse(
      state,
      await inpaintPreview({
        scoreId: state.scoreId!,
        measureId: state.selectedMeasureId!,
        revision: state.revision!,
      }),
    );
    expect(state.draftId).toBe('draft-2');

    // Discard
    await discardDraft({ scoreId: state.scoreId!, draftId: state.draftId! });
    state = applyDiscardDraft(state);

    // Score unchanged, draft gone
    expect(state.scoreXml).toBe(MINIMAL_XML);
    expect(state.revision).toBe(0);
    expect(state.draftId).toBeNull();
    expect(state.draftXml).toBeNull();
  });

  it('status text after compose: score is loaded and measure map present', async () => {
    vi.mocked(compose).mockResolvedValue({
      scoreId: 'score-3',
      revision: 0,
      scoreXML: MINIMAL_XML,
      measureMap: { '0': 'measure-1' },
    });

    const response = await compose({});
    // App.tsx shows 'Score loaded. Click a measure to inpaint.' after compose.
    // We verify the preconditions for that message: scoreId set, draftId absent.
    const state = applyComposeResponse(initialState, response);
    expect(state.scoreId).not.toBeNull();
    expect(state.draftId).toBeNull();
  });

  it('status text after preview: draft is present and ready for review', async () => {
    vi.mocked(inpaintPreview).mockResolvedValue({
      draftId: 'draft-3',
      scoreXML: DRAFT_XML,
      baseRevision: 0,
    });

    const scoreState: ScoreState = {
      ...initialState,
      scoreId: 'score-3',
      revision: 0,
      scoreXml: MINIMAL_XML,
      measureMap: { '0': 'measure-1' },
      selectedMeasureId: 'measure-1',
      selectedBarIndex: 0,
    };
    const response = await inpaintPreview({
      scoreId: scoreState.scoreId!,
      measureId: scoreState.selectedMeasureId!,
      revision: scoreState.revision!,
    });
    // App.tsx shows 'Draft ready. Compare and commit or discard.' when draftId is set.
    const state = applyInpaintPreviewResponse(scoreState, response);
    expect(state.draftId).not.toBeNull();
  });

  it('status text after commit: draft cleared and revision updated', async () => {
    vi.mocked(commitDraft).mockResolvedValue({
      scoreXML: COMMITTED_XML,
      revision: 2,
    });

    const draftState: ScoreState = {
      ...initialState,
      scoreId: 'score-3',
      revision: 1,
      scoreXml: MINIMAL_XML,
      draftId: 'draft-3',
      draftXml: DRAFT_XML,
      draftBaseRevision: 1,
    };
    const response = await commitDraft({ scoreId: 'score-3', draftId: 'draft-3' });
    // App.tsx shows 'Draft committed.' when draftId becomes null after commit.
    const state = applyCommitResponse(draftState, response);
    expect(state.draftId).toBeNull();
    expect(state.revision).toBe(2);
  });

  it('status text after discard: draft cleared, original score intact', async () => {
    vi.mocked(discardDraft).mockResolvedValue({ ok: true });

    const draftState: ScoreState = {
      ...initialState,
      scoreId: 'score-3',
      revision: 1,
      scoreXml: MINIMAL_XML,
      draftId: 'draft-3',
      draftXml: DRAFT_XML,
    };
    await discardDraft({ scoreId: 'score-3', draftId: 'draft-3' });
    // App.tsx shows 'Draft discarded.' when draftId becomes null after discard.
    const state = applyDiscardDraft(draftState);
    expect(state.draftId).toBeNull();
    expect(state.scoreXml).toBe(MINIMAL_XML);
  });

  it('eventHitMap: getEventId resolves eventId from a hit key after compose', async () => {
    const hitMap: EventHitMap = { '0|0|-1|-1': 'evt-bar0-voice0' };
    vi.mocked(compose).mockResolvedValue({
      scoreId: 'score-4',
      revision: 0,
      scoreXML: MINIMAL_XML,
      measureMap: { '0': 'measure-1' },
      eventHitMap: hitMap,
    });

    const response = await compose({});
    const state = applyComposeResponse(initialState, response);

    const eventId = getEventId(state.eventHitMap, { barIndex: 0, voiceIndex: 0 });
    expect(eventId).toBe('evt-bar0-voice0');
  });
});
