import type { EventHitMap, MeasureMap } from '../state/types';

export interface ComposeRequest {
  prompt?: string;
  constraints?: Record<string, unknown>;
}

export interface ComposeResponse {
  scoreId: string;
  revision: number;
  scoreXML: string;
  measureMap?: MeasureMap;
  eventHitMap?: EventHitMap;
  midi?: string;
}

export interface InpaintPreviewRequest {
  scoreId: string;
  measureId: string;
  revision: number;
  constraints?: {
    keepHarmony?: boolean;
    keepRhythm?: boolean;
    keepSoprano?: boolean;
    fixedPitches?: string[];
    fixedOnsets?: number[];
  };
  locks?: {
    lockedEventIds?: string[];
    lockedRanges?: Array<{
      startTick: number;
      endTick: number;
      type: 'pitch' | 'onset' | 'all';
    }>;
  };
  mode?: 'window' | 'repair';
}

export interface InpaintPreviewResponse {
  draftId: string;
  scoreXML: string;
  baseRevision: number;
  highlightMeasureId?: string;
  measureMap?: MeasureMap;
  eventHitMap?: EventHitMap;
  lockedEventIds?: string[];
  changedMeasureIds?: string[];
}

export interface CommitDraftRequest {
  scoreId: string;
  draftId: string;
}

export interface CommitDraftResponse {
  scoreXML: string;
  revision: number;
  measureMap?: MeasureMap;
  eventHitMap?: EventHitMap;
}

export interface DiscardDraftRequest {
  scoreId: string;
  draftId: string;
}

export interface DiscardDraftResponse {
  ok: boolean;
}

export interface AltPositionsRequest {
  scoreId: string;
  measureId: string;
  eventHitKey?: {
    barIndex: number;
    voiceIndex?: number;
    beatIndex?: number;
    noteIndex?: number;
  };
}

export interface ApplyFingeringRequest {
  scoreId: string;
  revision: number;
  fingeringSelections: Array<{ eventId: string; stringIndex: number; fret: number }>;
}

export interface ApplyFingeringResponse {
  scoreXML: string;
  revision: number;
}
