export type MeasureMap = Record<string, string>;
export type EventHitMap = Record<string, string>;
export type InstrumentMode = 'guitar' | 'piano';
export type ScoreViewTab = 'score' | 'tab';

export interface HitKey {
  barIndex: number;
  voiceIndex?: number;
  beatIndex?: number;
  noteIndex?: number;
}

export interface ScoreState {
  scoreId: string | null;
  revision: number | null;
  scoreXml: string | null;
  measureMap: MeasureMap | null;
  eventHitMap: EventHitMap | null;
  draftId: string | null;
  draftXml: string | null;
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
