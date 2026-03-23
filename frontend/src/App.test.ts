import { vi, describe, it, expect, beforeEach } from 'vitest';
import { compose, inpaintPreview, commitDraft, discardDraft } from './api/client';
import {
  canUseGuitarNoteActions,
  getActiveRenderView,
  getEventId,
  getMeasureId,
  inferInstrumentMode,
  type EventHitMap,
  type MeasureMap,
  type ScoreDocumentBundle,
  type ScoreState,
} from './state/types';

vi.mock('./api/client');

const MINIMAL_XML = '<score-partwise version="3.1"/>';
const DRAFT_XML = '<score-partwise version="3.1"><!-- draft --></score-partwise>';
const COMMITTED_XML = '<score-partwise version="3.1"><!-- committed --></score-partwise>';

const makeDocument = (
  instrumentMode: 'guitar' | 'piano' = 'guitar',
  overrides?: {
    scoreMeasureMap?: MeasureMap;
    scoreEventHitMap?: EventHitMap;
    tabMeasureMap?: MeasureMap;
    tabEventHitMap?: EventHitMap;
    scoreXml?: string;
    tabXml?: string;
  },
): ScoreDocumentBundle => ({
  instrumentMode,
  views: {
    score: {
      xml: overrides?.scoreXml ?? MINIMAL_XML,
      measureMap: overrides?.scoreMeasureMap,
      eventHitMap: overrides?.scoreEventHitMap,
    },
    ...(instrumentMode === 'guitar'
      ? {
          tab: {
            xml: overrides?.tabXml ?? MINIMAL_XML,
            measureMap: overrides?.tabMeasureMap,
            eventHitMap: overrides?.tabEventHitMap,
          },
        }
      : {}),
  },
});

const initialState: ScoreState = {
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

function applyComposeResponse(
  state: ScoreState,
  response: {
    scoreId: string;
    revision: number;
    document: ScoreDocumentBundle;
    midi?: string;
    instrumentMode: 'guitar' | 'piano';
  },
): ScoreState {
  return {
    ...state,
    scoreId: response.scoreId,
    revision: response.revision,
    document: response.document,
    midi: response.midi ?? null,
    instrumentMode: response.instrumentMode,
    draftId: null,
    draftDocument: null,
    draftBaseRevision: null,
    highlightMeasureId: null,
    selectedMeasureId: null,
    selectedBarIndex: null,
    lockedEventIds: null,
    changedMeasureIds: null,
    lastEventId: null,
  };
}

function applyMeasureSelect(state: ScoreState, barIndex: number, viewTab: 'score' | 'tab'): ScoreState {
  const activeDocument = state.draftDocument ?? state.document;
  const activeView = getActiveRenderView(activeDocument, viewTab);
  return {
    ...state,
    selectedBarIndex: barIndex,
    selectedMeasureId: getMeasureId(activeView?.measureMap, barIndex),
  };
}

function applyInpaintPreviewResponse(
  state: ScoreState,
  response: {
    draftId: string;
    document: ScoreDocumentBundle;
    baseRevision: number;
    highlightMeasureId?: string;
    lockedEventIds?: string[];
    changedMeasureIds?: string[];
  },
): ScoreState {
  return {
    ...state,
    draftId: response.draftId,
    draftDocument: response.document,
    draftBaseRevision: response.baseRevision,
    highlightMeasureId: response.highlightMeasureId ?? null,
    lockedEventIds: response.lockedEventIds ?? null,
    changedMeasureIds: response.changedMeasureIds ?? null,
  };
}

function applyCommitResponse(
  state: ScoreState,
  response: {
    document: ScoreDocumentBundle;
    revision: number;
  },
): ScoreState {
  return {
    ...state,
    document: response.document,
    revision: response.revision,
    instrumentMode: response.document.instrumentMode,
    draftId: null,
    draftDocument: null,
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
    draftDocument: null,
    draftBaseRevision: null,
    highlightMeasureId: null,
    lockedEventIds: null,
    changedMeasureIds: null,
  };
}

describe('compose/inpaint workflow — render bundle state', () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it('compose stores a bundled document instead of a raw score XML', async () => {
    const composeResponse = {
      scoreId: 'score-abc',
      revision: 0,
      document: makeDocument('guitar', {
        scoreMeasureMap: { '0': 'measure-1' },
        tabMeasureMap: { '0': 'measure-1-tab' },
      }),
      instrumentMode: 'guitar' as const,
    };
    vi.mocked(compose).mockResolvedValue(composeResponse);

    const state = applyComposeResponse(initialState, await compose({}));

    expect(state.document?.views.score.measureMap).toEqual({ '0': 'measure-1' });
    expect(state.document?.views.tab?.measureMap).toEqual({ '0': 'measure-1-tab' });
    expect(state.draftDocument).toBeNull();
  });

  it('measure selection resolves against the active score view', () => {
    const state: ScoreState = {
      ...initialState,
      document: makeDocument('guitar', {
        scoreMeasureMap: { '0': 'score-measure-1' },
        tabMeasureMap: { '0': 'tab-measure-1' },
      }),
    };

    expect(applyMeasureSelect(state, 0, 'score').selectedMeasureId).toBe('score-measure-1');
    expect(applyMeasureSelect(state, 0, 'tab').selectedMeasureId).toBe('tab-measure-1');
  });

  it('inpaint preview stores a draft document bundle without mutating the committed document', async () => {
    const committedDocument = makeDocument('guitar', { scoreXml: MINIMAL_XML });
    const previewResponse = {
      draftId: 'draft-xyz',
      document: makeDocument('guitar', { scoreXml: DRAFT_XML, tabXml: DRAFT_XML }),
      baseRevision: 0,
      highlightMeasureId: 'measure-1',
      lockedEventIds: ['evt-1'],
      changedMeasureIds: ['measure-1'],
    };
    vi.mocked(inpaintPreview).mockResolvedValue(previewResponse);

    const state = applyInpaintPreviewResponse(
      {
        ...initialState,
        scoreId: 'score-abc',
        revision: 0,
        document: committedDocument,
        selectedMeasureId: 'measure-1',
      },
      await inpaintPreview({ scoreId: 'score-abc', measureId: 'measure-1', revision: 0 }),
    );

    expect(state.document).toBe(committedDocument);
    expect(state.draftDocument?.views.score.xml).toBe(DRAFT_XML);
    expect(state.highlightMeasureId).toBe('measure-1');
  });

  it('commit replaces the committed document bundle and clears draft state', async () => {
    const commitResponse = {
      document: makeDocument('guitar', { scoreXml: COMMITTED_XML, tabXml: COMMITTED_XML }),
      revision: 1,
    };
    vi.mocked(commitDraft).mockResolvedValue(commitResponse);

    const state = applyCommitResponse(
      {
        ...initialState,
        scoreId: 'score-abc',
        revision: 0,
        document: makeDocument(),
        draftId: 'draft-1',
        draftDocument: makeDocument('guitar', { scoreXml: DRAFT_XML, tabXml: DRAFT_XML }),
      },
      await commitDraft({ scoreId: 'score-abc', draftId: 'draft-1' }),
    );

    expect(state.document?.views.score.xml).toBe(COMMITTED_XML);
    expect(state.draftDocument).toBeNull();
    expect(state.revision).toBe(1);
  });

  it('discard clears only the draft bundle', async () => {
    vi.mocked(discardDraft).mockResolvedValue({ ok: true });

    const state = applyDiscardDraft({
      ...initialState,
      scoreId: 'score-abc',
      revision: 0,
      document: makeDocument('guitar', { scoreXml: MINIMAL_XML }),
      draftId: 'draft-1',
      draftDocument: makeDocument('guitar', { scoreXml: DRAFT_XML }),
    });

    expect(state.document?.views.score.xml).toBe(MINIMAL_XML);
    expect(state.draftDocument).toBeNull();
  });

  it('guitar note actions stay gated by instrument mode', () => {
    expect(canUseGuitarNoteActions('guitar')).toBe(true);
    expect(canUseGuitarNoteActions('piano')).toBe(false);
  });

  it('event lookup resolves from the active tab view map', () => {
    const state: ScoreState = {
      ...initialState,
      document: makeDocument('guitar', {
        tabEventHitMap: { '0|0|-1|-1': 'evt-tab' },
      }),
    };

    const eventId = getEventId(
      getActiveRenderView(state.document, 'tab')?.eventHitMap,
      { barIndex: 0, voiceIndex: 0 },
    );
    expect(eventId).toBe('evt-tab');
  });

  it('inferInstrumentMode still detects piano and guitar from XML content', () => {
    expect(inferInstrumentMode('<score-partwise><attributes><staves>2</staves></attributes></score-partwise>')).toBe('piano');
    expect(inferInstrumentMode('<score-partwise><staff-details><staff-tuning line="1"/></staff-details></score-partwise>')).toBe('guitar');
  });
});
