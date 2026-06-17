export type MeasureMap = Record<string, string>;
export type EventHitMap = Record<string, string>;
export type InstrumentMode = 'guitar' | 'piano';
export type ScoreViewTab = 'score' | 'tab';
export type ScoreBranchKind = 'piano' | 'guitar';

export interface RenderView {
  xml: string;
  measureMap?: MeasureMap;
  eventHitMap?: EventHitMap;
}

export interface ScoreDocumentBundle {
  instrumentMode: InstrumentMode;
  views: {
    score: RenderView;
    tab?: RenderView;
  };
}

export interface HitKey {
  barIndex: number;
  voiceIndex?: number;
  beatIndex?: number;
  noteIndex?: number;
}

export type GuitarConversionSettings = Record<string, unknown>;
export type GuitarConversionDiagnostics = Record<string, unknown>;
export type PianoToGuitarSourceMap = Array<Record<string, unknown>>;

export interface ScoreBranchBase {
  branch: ScoreBranchKind;
  scoreId: string | null;
  revision: number | null;
  document: ScoreDocumentBundle | null;
  draftId: string | null;
  draftDocument: ScoreDocumentBundle | null;
  draftBaseRevision: number | null;
  highlightMeasureId: string | null;
  selectedMeasureId: string | null;
  selectedBarIndex: number | null;
  lockedEventIds: string[] | null;
  changedMeasureIds: string[] | null;
  lastEventId: string | null;
  midi: string | null;
  instrumentMode: InstrumentMode;
}

export interface PianoScoreBranch extends ScoreBranchBase {
  branch: 'piano';
  instrumentMode: 'piano';
}

export interface GuitarScoreBranch extends ScoreBranchBase {
  branch: 'guitar';
  instrumentMode: 'guitar';
  sourcePianoRevisionId: string | null;
  conversionSettings: GuitarConversionSettings | null;
  diagnostics: GuitarConversionDiagnostics | null;
  sourceMap: PianoToGuitarSourceMap;
}

export type ScoreBranch = PianoScoreBranch | GuitarScoreBranch;

export interface ProjectScoreState {
  activeBranch: ScoreBranchKind;
  piano: PianoScoreBranch;
  guitar: GuitarScoreBranch | null;
}

export type ScoreState = ProjectScoreState;

type PianoBranchOverrides = Partial<Omit<PianoScoreBranch, 'branch' | 'instrumentMode'>>;
type GuitarBranchOverrides = Partial<Omit<GuitarScoreBranch, 'branch' | 'instrumentMode'>>;

export const createPianoBranch = (
  overrides: PianoBranchOverrides = {},
): PianoScoreBranch => ({
  branch: 'piano',
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
  instrumentMode: 'piano',
  ...overrides,
});

export const createGuitarBranch = (
  overrides: GuitarBranchOverrides = {},
): GuitarScoreBranch => ({
  branch: 'guitar',
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
  instrumentMode: 'guitar',
  sourcePianoRevisionId: null,
  conversionSettings: null,
  diagnostics: null,
  sourceMap: [],
  ...overrides,
});

export const createInitialProjectScoreState = (): ProjectScoreState => ({
  activeBranch: 'piano',
  piano: createPianoBranch(),
  guitar: null,
});

export const getActiveScoreBranch = (
  state: ProjectScoreState,
): ScoreBranch | null =>
  state.activeBranch === 'guitar' ? state.guitar : state.piano;

export const updatePianoBranch = (
  state: ProjectScoreState,
  update: (branch: PianoScoreBranch) => PianoScoreBranch,
): ProjectScoreState => ({
  ...state,
  piano: update(state.piano),
});

export const updateGuitarBranch = (
  state: ProjectScoreState,
  update: (branch: GuitarScoreBranch) => GuitarScoreBranch,
): ProjectScoreState => {
  if (!state.guitar) {
    return state;
  }
  return {
    ...state,
    guitar: update(state.guitar),
  };
};

export const getPianoRevisionId = (
  piano: PianoScoreBranch,
): string | null =>
  piano.scoreId && piano.revision !== null
    ? `${piano.scoreId}@${piano.revision}`
    : null;

export const isGuitarBranchStale = (
  state: ProjectScoreState,
): boolean => {
  if (!state.guitar?.sourcePianoRevisionId) {
    return false;
  }
  const currentPianoRevisionId = getPianoRevisionId(state.piano);
  return Boolean(
    currentPianoRevisionId &&
      currentPianoRevisionId !== state.guitar.sourcePianoRevisionId,
  );
};

export const inferInstrumentMode = (xml: string): InstrumentMode => {
  if (xml.includes('<staves>2</staves>')) {
    return 'piano';
  }

  const firstMeasure = xml.match(/<measure\b[\s\S]*?<\/measure>/i)?.[0] ?? xml;
  if (
    firstMeasure.includes('<staff-details') ||
    firstMeasure.includes('<staff-tuning')
  ) {
    return 'guitar';
  }

  return 'piano';
};

export const canUseGuitarNoteActions = (
  instrumentMode: InstrumentMode | null,
): instrumentMode is 'guitar' => instrumentMode === 'guitar';

export const getActiveRenderView = (
  document: ScoreDocumentBundle | null | undefined,
  viewTab: ScoreViewTab,
): RenderView | null => {
  if (!document) {
    return null;
  }
  if (document.instrumentMode === 'piano') {
    return document.views.score;
  }
  if (viewTab === 'tab' && document.views.tab) {
    return document.views.tab;
  }
  return document.views.score;
};

export const toHitKey = (hit: HitKey): string => {
  const voice = hit.voiceIndex ?? -1;
  const beat = hit.beatIndex ?? -1;
  const note = hit.noteIndex ?? -1;
  return `${hit.barIndex}|${voice}|${beat}|${note}`;
};

export const getMeasureId = (
  measureMap: MeasureMap | null | undefined,
  barIndex: number,
): string | null => {
  if (!measureMap) {
    return null;
  }
  return measureMap[String(barIndex)] ?? null;
};

export const getEventId = (
  eventHitMap: EventHitMap | null | undefined,
  hit: HitKey,
): string | null => {
  if (!eventHitMap) {
    return null;
  }
  return eventHitMap[toHitKey(hit)] ?? null;
};
