export type MeasureMap = Record<string, string>;
export type EventHitMap = Record<string, string>;
export type InstrumentMode = 'guitar' | 'piano';
export type ScoreViewTab = 'score' | 'tab';

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

export interface ScoreState {
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
  instrumentMode: InstrumentMode | null;
}

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

  return 'guitar';
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
