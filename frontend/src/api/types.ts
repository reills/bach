import type { InstrumentMode, ScoreDocumentBundle } from '../state/types';

export interface ComposeConstraints {
  useGrammarMask?: boolean;
  useScg?: boolean;
  texture?: number;
  voices?: number;
  voiceCount?: number;
  temperature?: number;
  topP?: number;
  maxLength?: number;
  qualityPasses?: number;
  voiceLeading?: 'fast' | 'off' | 'balanced' | 'best';
}

export interface ComposeRequest {
  prompt?: string;
  constraints?: ComposeConstraints & Record<string, unknown>;
  render_mode?: InstrumentMode;
}

export interface ComposeResponse {
  scoreId: string;
  revision: number;
  document: ScoreDocumentBundle;
  midi?: string;
  instrumentMode: InstrumentMode;
  diagnostics?: Record<string, unknown>;
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
  document: ScoreDocumentBundle;
  baseRevision: number;
  highlightMeasureId?: string;
  lockedEventIds?: string[];
  changedMeasureIds?: string[];
}

export interface CommitDraftRequest {
  scoreId: string;
  draftId: string;
}

export interface CommitDraftResponse {
  document: ScoreDocumentBundle;
  revision: number;
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

export interface AltPositionsResponse {
  eventId: string;
  options: Array<{
    stringIndex: number;
    fret: number;
    selected: boolean;
  }>;
}

export interface ApplyFingeringRequest {
  scoreId: string;
  revision: number;
  fingeringSelections: Array<{ eventId: string; stringIndex: number; fret: number }>;
}

export interface ApplyFingeringResponse {
  document: ScoreDocumentBundle;
  revision: number;
}
