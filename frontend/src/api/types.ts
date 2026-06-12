import type { InstrumentMode, ScoreDocumentBundle } from '../state/types';

export interface ComposeConstraints {
  engine?: 'transformer' | 'emi' | 'hybrid' | 'instrumental_v6';
  useGrammarMask?: boolean;
  useScg?: boolean;
  texture?: number;
  voices?: number;
  voiceCount?: number;
  measures?: number;
  seed?: number;
  randomSeed?: number;
  temperature?: number;
  topP?: number;
  maxLength?: number;
  qualityPasses?: number;
  voiceLeading?: 'fast' | 'off' | 'balanced' | 'best';
  hybridAllowEmiFallback?: boolean;
  noveltyNgram?: number;
  sourceTokenSequences?: string[][];
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

export interface AppendMeasuresRequest {
  scoreId: string;
  revision: number;
  count: number;
}

export interface AppendMeasuresResponse {
  document: ScoreDocumentBundle;
  revision: number;
  addedMeasureIds: string[];
}

export type MeasureGenerationOperation =
  | 'prepend'
  | 'insert_before'
  | 'insert_after'
  | 'append'
  | 'replace';

export interface GenerateMeasuresRequest {
  scoreId: string;
  revision: number;
  operation: MeasureGenerationOperation;
  count: number;
  measureId?: string;
  prompt?: string;
  constraints?: ComposeConstraints & Record<string, unknown>;
  render_mode?: InstrumentMode;
}

export interface GenerateMeasuresResponse {
  document: ScoreDocumentBundle;
  revision: number;
  insertedMeasureIds: string[];
  replacedMeasureIds: string[];
  changedMeasureIds: string[];
  diagnostics?: Record<string, unknown>;
}

export interface GuitarConversionSettings {
  difficulty?: 'easy' | 'medium' | 'hard';
  maxFret?: number;
  preferredPosition?: number;
  allowOctaveShift?: boolean;
  octaveShiftPolicy?: 'none' | 'below_range' | 'outside_range';
  allowDropNotes?: boolean;
  preserveMelody?: boolean;
  preserveBass?: boolean;
  maxHandSpanFrets?: number;
  maxNotesPerOnset?: number;
  tuning?: number[];
}

export interface ConvertToGuitarRequest {
  scoreId?: string;
  revision?: number;
  pianoScore?: Record<string, unknown>;
  sourcePianoRevisionId?: string;
  settings?: GuitarConversionSettings;
}

export interface ConvertToGuitarResponse {
  scoreId: string;
  revision: number;
  branch: 'guitar';
  instrumentMode: 'guitar';
  document: ScoreDocumentBundle;
  scoreXML: string;
  guitarMusicXml: string;
  guitarTabXml?: string | null;
  midi: string;
  sourcePianoRevisionId: string;
  sourcePianoScoreId?: string | null;
  sourcePianoRevision?: number | null;
  conversionSettings: Record<string, unknown>;
  diagnostics: Record<string, unknown>;
  sourceMap: Array<Record<string, unknown>>;
}
