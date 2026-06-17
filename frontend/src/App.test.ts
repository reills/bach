import { vi, describe, it, expect, beforeEach } from 'vitest';
import { commitDraft, compose, convertToGuitar, discardDraft, generateMeasures } from './api/client';
import {
  canUseGuitarNoteActions,
  createGuitarBranch,
  createInitialProjectScoreState,
  createPianoBranch,
  getActiveRenderView,
  getActiveScoreBranch,
  getEventId,
  getMeasureId,
  inferInstrumentMode,
  isGuitarBranchStale,
  updateGuitarBranch,
  updatePianoBranch,
  type EventHitMap,
  type MeasureMap,
  type ProjectScoreState,
  type ScoreBranch,
  type ScoreBranchKind,
  type ScoreDocumentBundle,
} from './state/types';

vi.mock('./api/client');

const MINIMAL_XML = '<score-partwise version="3.1"/>';
const PIANO_XML = '<score-partwise version="3.1"><!-- piano --></score-partwise>';
const GUITAR_XML = '<score-partwise version="3.1"><!-- guitar --></score-partwise>';
const DRAFT_XML = '<score-partwise version="3.1"><!-- draft --></score-partwise>';
const COMMITTED_XML = '<score-partwise version="3.1"><!-- committed --></score-partwise>';

const makeDocument = (
  instrumentMode: 'guitar' | 'piano' = 'piano',
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
            xml: overrides?.tabXml ?? overrides?.scoreXml ?? MINIMAL_XML,
            measureMap: overrides?.tabMeasureMap,
            eventHitMap: overrides?.tabEventHitMap,
          },
        }
      : {}),
  },
});

const initialState = createInitialProjectScoreState();

function updateBranch(
  state: ProjectScoreState,
  branchKind: ScoreBranchKind,
  update: (branch: ScoreBranch) => ScoreBranch,
): ProjectScoreState {
  if (branchKind === 'piano') {
    return {
      ...state,
      piano: update(state.piano) as ProjectScoreState['piano'],
    };
  }
  if (!state.guitar) return state;
  return {
    ...state,
    guitar: update(state.guitar) as NonNullable<ProjectScoreState['guitar']>,
  };
}

function applyComposeResponse(
  state: ProjectScoreState,
  response: {
    scoreId: string;
    revision: number;
    document: ScoreDocumentBundle;
    midi?: string;
  },
): ProjectScoreState {
  return {
    ...state,
    activeBranch: 'piano',
    piano: createPianoBranch({
      scoreId: response.scoreId,
      revision: response.revision,
      document: response.document,
      midi: response.midi ?? null,
    }),
    guitar: null,
  };
}

function applyMeasureSelect(
  state: ProjectScoreState,
  barIndex: number,
  viewTab: 'score' | 'tab',
): ProjectScoreState {
  const branch = getActiveScoreBranch(state);
  const activeDocument = branch?.draftDocument ?? branch?.document;
  const activeView = getActiveRenderView(activeDocument, viewTab);
  return updateBranch(state, state.activeBranch, (currentBranch) => ({
    ...currentBranch,
    selectedBarIndex: barIndex,
    selectedMeasureId: getMeasureId(activeView?.measureMap, barIndex),
  }));
}

function applyGenerateMeasuresResponse(
  state: ProjectScoreState,
  response: {
    document: ScoreDocumentBundle;
    revision: number;
    insertedMeasureIds: string[];
    changedMeasureIds: string[];
  },
): ProjectScoreState {
  const changedMeasureId = response.insertedMeasureIds[0] ?? response.changedMeasureIds[0] ?? null;
  return {
    ...state,
    activeBranch: 'piano',
    piano: {
      ...state.piano,
      document: response.document,
      revision: response.revision,
      draftId: null,
      draftDocument: null,
      draftBaseRevision: null,
      highlightMeasureId: changedMeasureId,
      selectedMeasureId: changedMeasureId,
      selectedBarIndex: null,
      lockedEventIds: null,
      changedMeasureIds: response.changedMeasureIds,
    },
  };
}

function applyConvertToGuitarResponse(
  state: ProjectScoreState,
  response: {
    scoreId: string;
    revision: number;
    document: ScoreDocumentBundle;
    midi?: string;
    sourcePianoRevisionId: string;
    conversionSettings: Record<string, unknown>;
    diagnostics: Record<string, unknown>;
    sourceMap: Array<Record<string, unknown>>;
  },
): ProjectScoreState {
  return {
    ...state,
    activeBranch: 'guitar',
    guitar: createGuitarBranch({
      scoreId: response.scoreId,
      revision: response.revision,
      document: response.document,
      midi: response.midi ?? null,
      sourcePianoRevisionId: response.sourcePianoRevisionId,
      conversionSettings: response.conversionSettings,
      diagnostics: response.diagnostics,
      sourceMap: response.sourceMap,
    }),
  };
}

function applyCommitResponse(
  state: ProjectScoreState,
  response: {
    document: ScoreDocumentBundle;
    revision: number;
  },
): ProjectScoreState {
  return updateBranch(state, state.activeBranch, (branch) => ({
    ...branch,
    document: response.document,
    revision: response.revision,
    draftId: null,
    draftDocument: null,
    draftBaseRevision: null,
    highlightMeasureId: null,
    lockedEventIds: null,
    changedMeasureIds: null,
  }));
}

function applyDiscardDraft(state: ProjectScoreState): ProjectScoreState {
  return updateBranch(state, state.activeBranch, (branch) => ({
    ...branch,
    draftId: null,
    draftDocument: null,
    draftBaseRevision: null,
    highlightMeasureId: null,
    lockedEventIds: null,
    changedMeasureIds: null,
  }));
}

describe('compose/generated-measure workflow — branch model', () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it('compose stores the generated score in the piano branch and clears guitar', async () => {
    const composeResponse = {
      scoreId: 'score-abc',
      revision: 0,
      document: makeDocument('piano', {
        scoreMeasureMap: { '0': 'measure-1' },
        scoreXml: PIANO_XML,
      }),
      instrumentMode: 'piano' as const,
    };
    vi.mocked(compose).mockResolvedValue(composeResponse);

    const state = applyComposeResponse(
      {
        ...initialState,
        guitar: createGuitarBranch({
          scoreId: 'old-guitar',
          document: makeDocument('guitar', { scoreXml: GUITAR_XML }),
        }),
      },
      await compose({}),
    );

    expect(state.activeBranch).toBe('piano');
    expect(state.piano.document?.views.score.measureMap).toEqual({ '0': 'measure-1' });
    expect(state.piano.document?.views.score.xml).toBe(PIANO_XML);
    expect(state.guitar).toBeNull();
  });

  it('allows switching to an empty guitar branch without losing piano', () => {
    const state: ProjectScoreState = {
      ...initialState,
      activeBranch: 'guitar',
      piano: createPianoBranch({
        scoreId: 'score-abc',
        revision: 2,
        document: makeDocument('piano', { scoreXml: PIANO_XML }),
      }),
      guitar: null,
    };

    expect(getActiveScoreBranch(state)).toBeNull();
    expect(state.piano.document?.views.score.xml).toBe(PIANO_XML);
  });

  it('can hold an independent converted guitar branch beside piano', () => {
    const state: ProjectScoreState = {
      ...initialState,
      piano: createPianoBranch({
        scoreId: 'score-piano',
        revision: 3,
        document: makeDocument('piano', { scoreXml: PIANO_XML }),
      }),
      guitar: createGuitarBranch({
        scoreId: 'score-guitar',
        revision: 0,
        document: makeDocument('guitar', { scoreXml: GUITAR_XML, tabXml: '<tab/>' }),
        sourcePianoRevisionId: '3',
        diagnostics: { droppedNotes: [] },
      }),
    };

    const editedGuitarState: ProjectScoreState = {
      ...state,
      activeBranch: 'guitar',
      guitar: state.guitar
        ? {
            ...state.guitar,
            revision: 1,
            document: makeDocument('guitar', {
              scoreXml: '<score-partwise><!-- edited guitar --></score-partwise>',
              tabXml: '<tab><!-- edited --></tab>',
            }),
          }
        : null,
    };

    expect(editedGuitarState.piano.document?.views.score.xml).toBe(PIANO_XML);
    expect(editedGuitarState.guitar?.document?.views.tab?.xml).toBe('<tab><!-- edited --></tab>');
    expect(editedGuitarState.guitar?.sourcePianoRevisionId).toBe('3');
  });

  it('stores convert-to-guitar results as an active independent branch', async () => {
    const pianoBranch = createPianoBranch({
      scoreId: 'score-piano',
      revision: 3,
      document: makeDocument('piano', { scoreXml: PIANO_XML }),
      midi: 'piano-midi',
    });
    const convertedResponse = {
      scoreId: 'score-guitar',
      revision: 0,
      branch: 'guitar' as const,
      instrumentMode: 'guitar' as const,
      document: makeDocument('guitar', { scoreXml: GUITAR_XML, tabXml: '<tab/>' }),
      scoreXML: GUITAR_XML,
      guitarMusicXml: GUITAR_XML,
      guitarTabXml: '<tab/>',
      midi: 'guitar-midi',
      sourcePianoRevisionId: 'score-piano@3',
      conversionSettings: { difficulty: 'medium' },
      diagnostics: {
        droppedNotes: [{ sourceEventId: 'p1' }],
        octaveShiftedNotes: [],
        warnings: ['dropped 1 note'],
      },
      sourceMap: [{ pianoEventId: 'p1', guitarEventId: null, dropped: true }],
    };
    vi.mocked(convertToGuitar).mockResolvedValue(convertedResponse);

    const state = applyConvertToGuitarResponse(
      {
        ...initialState,
        activeBranch: 'piano',
        piano: pianoBranch,
      },
      await convertToGuitar({
        scoreId: 'score-piano',
        revision: 3,
        sourcePianoRevisionId: 'score-piano@3',
        settings: { difficulty: 'medium' },
      }),
    );

    expect(state.activeBranch).toBe('guitar');
    expect(state.piano).toBe(pianoBranch);
    expect(state.piano.document?.views.score.xml).toBe(PIANO_XML);
    expect(state.guitar?.scoreId).toBe('score-guitar');
    expect(state.guitar?.document?.views.tab?.xml).toBe('<tab/>');
    expect(state.guitar?.midi).toBe('guitar-midi');
    expect(state.guitar?.sourcePianoRevisionId).toBe('score-piano@3');
    expect(state.guitar?.diagnostics?.warnings).toEqual(['dropped 1 note']);
    expect(state.guitar?.sourceMap).toEqual([
      { pianoEventId: 'p1', guitarEventId: null, dropped: true },
    ]);
  });

  it('measure selection updates only the active branch', () => {
    const state: ProjectScoreState = {
      ...initialState,
      piano: createPianoBranch({
        document: makeDocument('piano', {
          scoreMeasureMap: { '0': 'piano-measure-1' },
        }),
      }),
      guitar: createGuitarBranch({
        document: makeDocument('guitar', {
          scoreMeasureMap: { '0': 'guitar-score-measure-1' },
          tabMeasureMap: { '0': 'guitar-tab-measure-1' },
        }),
      }),
    };

    const pianoSelected = applyMeasureSelect(state, 0, 'score');
    const guitarSelected = applyMeasureSelect(
      { ...pianoSelected, activeBranch: 'guitar' },
      0,
      'tab',
    );

    expect(guitarSelected.piano.selectedMeasureId).toBe('piano-measure-1');
    expect(guitarSelected.guitar?.selectedMeasureId).toBe('guitar-tab-measure-1');
  });

  it('generated measure operations update piano and preserve guitar', async () => {
    const guitarBranch = createGuitarBranch({
      scoreId: 'score-guitar',
      revision: 5,
      document: makeDocument('guitar', { scoreXml: GUITAR_XML }),
    });
    const generatedResponse = {
      document: makeDocument('piano', { scoreXml: DRAFT_XML }),
      revision: 1,
      insertedMeasureIds: ['measure-1b'],
      replacedMeasureIds: ['measure-1'],
      changedMeasureIds: ['measure-1', 'measure-1b'],
    };
    vi.mocked(generateMeasures).mockResolvedValue(generatedResponse);

    const state = applyGenerateMeasuresResponse(
      {
        ...initialState,
        piano: createPianoBranch({
          scoreId: 'score-abc',
          revision: 0,
          document: makeDocument('piano', { scoreXml: MINIMAL_XML }),
          selectedMeasureId: 'measure-1',
        }),
        guitar: guitarBranch,
      },
      await generateMeasures({
        scoreId: 'score-abc',
        measureId: 'measure-1',
        revision: 0,
        operation: 'replace',
        count: 1,
      }),
    );

    expect(state.piano.document?.views.score.xml).toBe(DRAFT_XML);
    expect(state.piano.highlightMeasureId).toBe('measure-1b');
    expect(state.piano.changedMeasureIds).toEqual(['measure-1', 'measure-1b']);
    expect(state.guitar).toBe(guitarBranch);
  });

  it('piano edits leave the converted guitar branch unchanged and mark it stale', () => {
    const guitarBranch = createGuitarBranch({
      scoreId: 'score-guitar',
      revision: 2,
      document: makeDocument('guitar', { scoreXml: GUITAR_XML }),
      sourcePianoRevisionId: 'score-piano@3',
    });
    const state: ProjectScoreState = {
      ...initialState,
      piano: createPianoBranch({
        scoreId: 'score-piano',
        revision: 3,
        document: makeDocument('piano', { scoreXml: PIANO_XML }),
      }),
      guitar: guitarBranch,
    };

    const edited = updatePianoBranch(state, (piano) => ({
      ...piano,
      revision: 4,
      document: makeDocument('piano', { scoreXml: DRAFT_XML }),
    }));

    expect(edited.guitar).toBe(guitarBranch);
    expect(edited.guitar?.document?.views.score.xml).toBe(GUITAR_XML);
    expect(isGuitarBranchStale(edited)).toBe(true);
  });

  it('guitar edits leave the source piano branch unchanged', () => {
    const pianoBranch = createPianoBranch({
      scoreId: 'score-piano',
      revision: 3,
      document: makeDocument('piano', { scoreXml: PIANO_XML }),
    });
    const state: ProjectScoreState = {
      ...initialState,
      activeBranch: 'guitar',
      piano: pianoBranch,
      guitar: createGuitarBranch({
        scoreId: 'score-guitar',
        revision: 0,
        document: makeDocument('guitar', { scoreXml: GUITAR_XML }),
        sourcePianoRevisionId: 'score-piano@3',
      }),
    };

    const edited = updateGuitarBranch(state, (guitar) => ({
      ...guitar,
      revision: 1,
      document: makeDocument('guitar', {
        scoreXml: COMMITTED_XML,
        tabXml: COMMITTED_XML,
      }),
    }));

    expect(edited.piano).toBe(pianoBranch);
    expect(edited.piano.document?.views.score.xml).toBe(PIANO_XML);
    expect(edited.guitar?.document?.views.tab?.xml).toBe(COMMITTED_XML);
    expect(isGuitarBranchStale(edited)).toBe(false);
  });

  it('commit replaces only the active branch document and clears its draft', async () => {
    const commitResponse = {
      document: makeDocument('guitar', { scoreXml: COMMITTED_XML, tabXml: COMMITTED_XML }),
      revision: 1,
    };
    vi.mocked(commitDraft).mockResolvedValue(commitResponse);

    const state = applyCommitResponse(
      {
        ...initialState,
        activeBranch: 'guitar',
        piano: createPianoBranch({
          document: makeDocument('piano', { scoreXml: PIANO_XML }),
        }),
        guitar: createGuitarBranch({
          scoreId: 'score-guitar',
          revision: 0,
          document: makeDocument('guitar', { scoreXml: GUITAR_XML }),
          draftId: 'draft-1',
          draftDocument: makeDocument('guitar', { scoreXml: DRAFT_XML, tabXml: DRAFT_XML }),
        }),
      },
      await commitDraft({ scoreId: 'score-guitar', draftId: 'draft-1' }),
    );

    expect(state.piano.document?.views.score.xml).toBe(PIANO_XML);
    expect(state.guitar?.document?.views.score.xml).toBe(COMMITTED_XML);
    expect(state.guitar?.draftDocument).toBeNull();
  });

  it('discard clears only the active branch draft bundle', async () => {
    vi.mocked(discardDraft).mockResolvedValue({ ok: true });

    const state = applyDiscardDraft({
      ...initialState,
      activeBranch: 'guitar',
      piano: createPianoBranch({
        draftDocument: makeDocument('piano', { scoreXml: DRAFT_XML }),
      }),
      guitar: createGuitarBranch({
        scoreId: 'score-guitar',
        revision: 0,
        document: makeDocument('guitar', { scoreXml: GUITAR_XML }),
        draftId: 'draft-1',
        draftDocument: makeDocument('guitar', { scoreXml: DRAFT_XML }),
      }),
    });

    expect(state.piano.draftDocument?.views.score.xml).toBe(DRAFT_XML);
    expect(state.guitar?.document?.views.score.xml).toBe(GUITAR_XML);
    expect(state.guitar?.draftDocument).toBeNull();
  });

  it('guitar note actions stay gated by instrument mode', () => {
    expect(canUseGuitarNoteActions('guitar')).toBe(true);
    expect(canUseGuitarNoteActions('piano')).toBe(false);
  });

  it('event lookup resolves from the active tab view map', () => {
    const state: ProjectScoreState = {
      ...initialState,
      activeBranch: 'guitar',
      guitar: createGuitarBranch({
        document: makeDocument('guitar', {
          tabEventHitMap: { '0|0|-1|-1': 'evt-tab' },
        }),
      }),
    };

    const branch = getActiveScoreBranch(state);
    const eventId = getEventId(
      getActiveRenderView(branch?.document, 'tab')?.eventHitMap,
      { barIndex: 0, voiceIndex: 0 },
    );
    expect(eventId).toBe('evt-tab');
  });

  it('inferInstrumentMode still detects piano and guitar from XML content', () => {
    expect(inferInstrumentMode('<score-partwise><attributes><staves>2</staves></attributes></score-partwise>')).toBe('piano');
    expect(inferInstrumentMode('<score-partwise><staff-details><staff-tuning line="1"/></staff-details></score-partwise>')).toBe('guitar');
    expect(inferInstrumentMode('<score-partwise version="3.1"/>')).toBe('piano');
  });
});
