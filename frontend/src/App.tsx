import { useMemo, useState } from 'react';
import './App.css';
import ScoreViewer, { type ScoreViewerPlaybackAPI } from './components/ScoreViewer';
import SheetMusicViewer, { type SheetMusicPlayerAPI } from './components/SheetMusicViewer';
import FingeringPicker from './components/FingeringPicker';
import {
  altPositions,
  applyFingering,
  commitDraft,
  compose,
  discardDraft,
  generateMeasures,
} from './api/client';
import type { MeasureGenerationOperation } from './api/types';
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
  type GuitarScoreBranch,
  type HitKey,
  type InstrumentMode,
  type PianoScoreBranch,
  type ScoreBranch,
  type ScoreBranchKind,
  type ScoreViewTab,
  type ScoreState,
} from './state/types';
import {
  buildDocumentBundle,
  loadLocalData,
  type LocalManifest,
} from './mock/localData';

const DEMO_XML = `<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part-list>
    <score-part id="P1">
      <part-name>Piano</part-name>
    </score-part>
  </part-list>
  <part id="P1">
    <measure number="1" xml:id="measure-1">
      <attributes>
        <divisions>4</divisions>
        <key>
          <fifths>0</fifths>
          <mode>minor</mode>
        </key>
        <time>
          <beats>4</beats>
          <beat-type>4</beat-type>
        </time>
        <staves>2</staves>
        <clef number="1">
          <sign>G</sign>
          <line>2</line>
        </clef>
        <clef number="2">
          <sign>F</sign>
          <line>4</line>
        </clef>
      </attributes>
      <note>
        <pitch><step>E</step><octave>4</octave></pitch>
        <duration>4</duration>
        <type>quarter</type>
        <staff>1</staff>
      </note>
      <note>
        <pitch><step>D</step><octave>4</octave></pitch>
        <duration>4</duration>
        <type>quarter</type>
        <staff>1</staff>
      </note>
      <note>
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>4</duration>
        <type>quarter</type>
        <staff>1</staff>
      </note>
      <note>
        <pitch><step>B</step><octave>3</octave></pitch>
        <duration>4</duration>
        <type>quarter</type>
        <staff>1</staff>
      </note>
    </measure>
    <measure number="2" xml:id="measure-2">
      <note>
        <pitch><step>A</step><octave>3</octave></pitch>
        <duration>4</duration>
        <type>quarter</type>
        <staff>1</staff>
      </note>
      <note>
        <pitch><step>G</step><octave>3</octave></pitch>
        <duration>4</duration>
        <type>quarter</type>
        <staff>1</staff>
      </note>
      <note>
        <pitch><step>A</step><octave>3</octave></pitch>
        <duration>4</duration>
        <type>quarter</type>
        <staff>1</staff>
      </note>
      <note>
        <pitch><step>B</step><octave>3</octave></pitch>
        <duration>4</duration>
        <type>quarter</type>
        <staff>1</staff>
      </note>
    </measure>
    <measure number="3" xml:id="measure-3">
      <note>
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>8</duration>
        <type>half</type>
        <staff>1</staff>
      </note>
      <note>
        <pitch><step>G</step><octave>2</octave></pitch>
        <duration>8</duration>
        <type>half</type>
        <staff>2</staff>
      </note>
    </measure>
    <measure number="4" xml:id="measure-4">
      <note>
        <pitch><step>C</step><octave>3</octave></pitch>
        <duration>16</duration>
        <type>whole</type>
        <staff>2</staff>
      </note>
    </measure>
  </part>
</score-partwise>
`;

const initialState: ScoreState = createInitialProjectScoreState();

type StatusTone = 'idle' | 'busy' | 'success' | 'error';
type DataSource = 'api' | 'local';
type VoiceCount = 1 | 2 | 3 | 4;
type CompositionEngine = 'instrumental_v6' | 'hybrid' | 'emi' | 'transformer';

const defaultSource: DataSource =
  import.meta.env.VITE_USE_LOCAL_DATA === 'true' ? 'local' : 'api';
const DEFAULT_MEASURE_TARGET = 8;
const MAX_MEASURE_TARGET = 64;
const MAX_GENERATED_MEASURE_COUNT = 32;
const DEFAULT_GENERATION_INSTRUMENT_MODE: InstrumentMode = 'piano';

const createRandomSeed = (): number =>
  Math.floor(Math.random() * 2_147_483_646) + 1;

const clampInteger = (value: number, min: number, max: number): number => {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, Math.round(value)));
};

const findMeasureBarIndex = (
  measureMap: Record<string, string> | null | undefined,
  measureId: string | null,
): number | null => {
  if (!measureMap || !measureId) return null;
  for (const [barIndex, mappedMeasureId] of Object.entries(measureMap)) {
    if (mappedMeasureId === measureId) {
      const parsed = Number.parseInt(barIndex, 10);
      return Number.isFinite(parsed) ? parsed : null;
    }
  }
  return null;
};

const formatDiagnosticValue = (value: unknown): string => {
  if (Array.isArray(value)) return value.map(String).join(' -> ');
  if (value === null || value === undefined) return 'n/a';
  if (typeof value === 'boolean') return value ? 'yes' : 'no';
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(3);
  return String(value);
};

const updateProjectBranch = (
  state: ScoreState,
  branchKind: ScoreBranchKind,
  update: (branch: ScoreBranch) => ScoreBranch,
): ScoreState => {
  if (branchKind === 'piano') {
    return {
      ...state,
      piano: update(state.piano) as PianoScoreBranch,
    };
  }

  if (!state.guitar) {
    return state;
  }

  return {
    ...state,
    guitar: update(state.guitar) as GuitarScoreBranch,
  };
};

const clearBranchReviewState = <T extends ScoreBranch>(branch: T): T => ({
  ...branch,
  draftId: null,
  draftDocument: null,
  draftBaseRevision: null,
  highlightMeasureId: null,
  lockedEventIds: null,
  changedMeasureIds: null,
});

const App = () => {
  const [state, setState] = useState<ScoreState>(initialState);
  const [alphaTabApi, setAlphaTabApi] = useState<ScoreViewerPlaybackAPI | null>(null);
  const [playerReady, setPlayerReady] = useState(false);
  const [sheetPlayerApi, setSheetPlayerApi] = useState<SheetMusicPlayerAPI | null>(null);
  const [sheetPlayerReady, setSheetPlayerReady] = useState(false);
  const [playbackPos, setPlaybackPos] = useState<{ current: number; total: number } | null>(null);
  const [prompt, setPrompt] = useState('');
  const [dataSource, setDataSource] = useState<DataSource>(defaultSource);
  const [localManifest, setLocalManifest] = useState<LocalManifest | null>(
    null,
  );
  const instrumentMode = DEFAULT_GENERATION_INSTRUMENT_MODE;
  const [voiceCount, setVoiceCount] = useState<VoiceCount>(4);
  const [measureTarget, setMeasureTarget] = useState(DEFAULT_MEASURE_TARGET);
  const [measureActionCount, setMeasureActionCount] = useState(1);
  const [compositionEngine, setCompositionEngine] =
    useState<CompositionEngine>('instrumental_v6');
  const [viewTab, setViewTab] = useState<ScoreViewTab>('score');
  const [statusTone, setStatusTone] = useState<StatusTone>('idle');
  const [statusMessage, setStatusMessage] = useState('Ready.');
  const [lastDiagnostics, setLastDiagnostics] = useState<Record<string, unknown> | null>(null);
  const [busy, setBusy] = useState(false);
  const [fingeringPicker, setFingeringPicker] = useState<{
    eventId: string;
    options: Array<{ stringIndex: number; fret: number; selected: boolean }>;
  } | null>(null);
  const [fingeringBusy, setFingeringBusy] = useState(false);

  const activeBranch = getActiveScoreBranch(state);
  const activeDocument = activeBranch?.draftDocument ?? activeBranch?.document ?? null;
  const activeInstrumentMode =
    activeDocument?.instrumentMode ?? activeBranch?.instrumentMode ?? state.activeBranch;
  const hasGuitarBranch = Boolean(state.guitar?.draftDocument ?? state.guitar?.document);
  const effectiveViewTab =
    state.activeBranch === 'guitar' && viewTab === 'tab' && activeDocument?.views.tab
      ? 'tab'
      : 'score';
  const activeView = getActiveRenderView(activeDocument, effectiveViewTab);
  const renderXml = activeView?.xml ?? null;
  const showGuitarEmptyState = state.activeBranch === 'guitar' && !hasGuitarBranch;
  const showSheetMusic =
    !showGuitarEmptyState &&
    (activeInstrumentMode === 'piano' ||
      (activeInstrumentMode === 'guitar' && effectiveViewTab === 'score'));
  const activePlayerReady = showSheetMusic ? sheetPlayerReady : playerReady;
  const canGeneratePianoMeasures =
    state.activeBranch === 'piano' &&
    dataSource === 'api' &&
    Boolean(state.piano.scoreId) &&
    state.piano.revision !== null;

  const setStatus = (tone: StatusTone, message: string) => {
    setStatusTone(tone);
    setStatusMessage(message);
  };

  const resetLoadedScoreUi = () => {
    setViewTab('score');
    setFingeringPicker(null);
    setFingeringBusy(false);
    setPlaybackPos(null);
    setPlayerReady(false);
    setSheetPlayerReady(false);
  };

  const handleBranchChange = (branch: ScoreBranchKind) => {
    setState((prev) => ({
      ...prev,
      activeBranch: branch,
    }));
    setFingeringPicker(null);
    setPlaybackPos(null);
    setPlayerReady(false);
    setSheetPlayerReady(false);
  };

  const handleCompose = async () => {
    setBusy(true);
    setStatus('busy', 'Composing a new score...');
    try {
      const seed = createRandomSeed();
      const response = await compose({
        prompt,
        constraints: {
          engine: compositionEngine,
          texture: voiceCount,
          measures: measureTarget,
          seed,
          randomSeed: seed,
          useGrammarMask: true,
          qualityPasses: 4,
          voiceLeading: 'balanced',
        },
        render_mode: instrumentMode,
      });
      resetLoadedScoreUi();
      setState((prev) => ({
        ...prev,
        activeBranch: 'piano',
        piano: createPianoBranch({
          scoreId: response.scoreId,
          revision: response.revision,
          document: response.document,
          midi: response.midi ?? null,
        }),
        guitar: null,
      }));
      setLastDiagnostics(response.diagnostics ?? null);
      setStatus('success', 'Score loaded. Click a measure to select it.');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Compose failed.';
      setStatus('error', message);
    } finally {
      setBusy(false);
    }
  };

  const handleLoadLocal = async () => {
    setBusy(true);
    setStatus('busy', 'Loading local test-data...');
    try {
      const bundle = await loadLocalData();
      const loadedInstrumentMode = inferInstrumentMode(bundle.baseXml);
      const loadedDocument = buildDocumentBundle(bundle.baseXml, loadedInstrumentMode);
      const loadedScoreId = `local:${bundle.manifest.baseScore}`;

      resetLoadedScoreUi();
      setState((prev) => ({
        ...prev,
        activeBranch: loadedInstrumentMode,
        piano:
          loadedInstrumentMode === 'piano'
            ? createPianoBranch({
                scoreId: loadedScoreId,
                revision: 0,
                document: loadedDocument,
              })
            : createPianoBranch(),
        guitar:
          loadedInstrumentMode === 'guitar'
            ? createGuitarBranch({
                scoreId: loadedScoreId,
                revision: 0,
                document: loadedDocument,
                sourcePianoRevisionId: 'local',
              })
            : null,
      }));

      setLocalManifest(bundle.manifest);
      setStatus(
        'success',
        `Loaded ${bundle.manifest.snippets.length} local snippets.`,
      );
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : 'Failed to load local test-data.';
      setStatus('error', message);
    } finally {
      setBusy(false);
    }
  };

  const handleLoadDemo = () => {
    const demoDocument = buildDocumentBundle(DEMO_XML, inferInstrumentMode(DEMO_XML));
    resetLoadedScoreUi();
    setState((prev) => ({
      ...prev,
      activeBranch: 'piano',
      piano: createPianoBranch({
        scoreId: 'demo',
        revision: 0,
        document: demoDocument,
      }),
      guitar: null,
    }));
    setStatus('success', 'Demo score loaded locally.');
  };

  const handleMeasureClick = (barIndex: number) => {
    const measureId = getMeasureId(activeView?.measureMap, barIndex);
    if (!measureId) {
      setStatus(
        'error',
        'Measure map missing. Load a score first.',
      );
      return;
    }
    setStatus('idle', `Selected bar ${barIndex + 1}. Use Rewrite Selected or Insert Before/After.`);
    setState((prev) =>
      updateProjectBranch(prev, prev.activeBranch, (branch) => ({
        ...branch,
        selectedBarIndex: barIndex,
        selectedMeasureId: measureId,
      })),
    );
  };

  const handleNoteClick = async (hit: HitKey) => {
    const eventId = getEventId(activeView?.eventHitMap, hit);
    setState((prev) =>
      updateProjectBranch(prev, prev.activeBranch, (branch) => ({
        ...branch,
        lastEventId: eventId,
      })),
    );

    if (!canUseGuitarNoteActions(activeInstrumentMode)) {
      setFingeringPicker(null);
      return;
    }

    if (!eventId || !activeBranch?.scoreId) return;
    const measureId = getMeasureId(activeView?.measureMap, hit.barIndex);
    if (!measureId) return;

    setFingeringBusy(true);
    try {
      const resp = await altPositions({ scoreId: activeBranch.scoreId, measureId, eventHitKey: hit });
      setFingeringPicker({ eventId: resp.eventId, options: resp.options });
    } catch {
      // Note has no alternate positions data — silently ignore
    } finally {
      setFingeringBusy(false);
    }
  };

  const handleApplyFingering = async (stringIndex: number, fret: number) => {
    const guitarBranch = state.activeBranch === 'guitar' ? state.guitar : null;
    if (!fingeringPicker || !guitarBranch?.scoreId || guitarBranch.revision === null) return;
    setFingeringBusy(true);
    try {
      const resp = await applyFingering({
        scoreId: guitarBranch.scoreId,
        revision: guitarBranch.revision,
        fingeringSelections: [{ eventId: fingeringPicker.eventId, stringIndex, fret }],
      });
      setState((prev) =>
        updateProjectBranch(prev, 'guitar', (branch) => ({
          ...branch,
          document: resp.document,
          revision: resp.revision,
        })),
      );
      setFingeringPicker(null);
      setStatus('success', 'Fingering applied.');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Apply fingering failed.';
      setStatus('error', message);
    } finally {
      setFingeringBusy(false);
    }
  };

  const handleCommitDraft = async () => {
    const branchKind = state.activeBranch;
    const branch = getActiveScoreBranch(state);

    if (dataSource === 'local') {
      if (!branch?.draftDocument) {
        setStatus('error', 'No draft to commit.');
        return;
      }
      setState((prev) =>
        updateProjectBranch(prev, branchKind, (currentBranch) =>
          clearBranchReviewState({
            ...currentBranch,
            document: currentBranch.draftDocument,
            revision: (currentBranch.revision ?? 0) + 1,
          }),
        ),
      );
      setStatus('success', 'Local draft committed.');
      return;
    }

    if (!branch?.scoreId || !branch.draftId) {
      setStatus('error', 'No draft to commit.');
      return;
    }

    setBusy(true);
    setStatus('busy', 'Committing draft...');
    try {
      const response = await commitDraft({
        scoreId: branch.scoreId,
        draftId: branch.draftId,
      });
      setState((prev) =>
        updateProjectBranch(prev, branchKind, (currentBranch) =>
          clearBranchReviewState({
            ...currentBranch,
            document: response.document,
            revision: response.revision,
          }),
        ),
      );
      setStatus('success', 'Draft committed.');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Commit failed.';
      setStatus('error', message);
    } finally {
      setBusy(false);
    }
  };

  const handleDiscardDraft = async () => {
    const branchKind = state.activeBranch;
    const branch = getActiveScoreBranch(state);

    if (dataSource === 'local') {
      if (!branch?.draftDocument) {
        setStatus('error', 'No draft to discard.');
        return;
      }
      setState((prev) =>
        updateProjectBranch(prev, branchKind, (currentBranch) =>
          clearBranchReviewState(currentBranch),
        ),
      );
      setStatus('success', 'Local draft discarded.');
      return;
    }

    if (!branch?.scoreId || !branch.draftId) {
      setStatus('error', 'No draft to discard.');
      return;
    }

    setBusy(true);
    setStatus('busy', 'Discarding draft...');
    try {
      await discardDraft({
        scoreId: branch.scoreId,
        draftId: branch.draftId,
      });
      setState((prev) =>
        updateProjectBranch(prev, branchKind, (currentBranch) =>
          clearBranchReviewState(currentBranch),
        ),
      );
      setStatus('success', 'Draft discarded.');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Discard failed.';
      setStatus('error', message);
    } finally {
      setBusy(false);
    }
  };

  const handleGenerateMeasures = async (operation: MeasureGenerationOperation) => {
    const pianoBranch = state.piano;

    if (state.activeBranch !== 'piano') {
      setStatus('error', 'Switch to Piano to generate or rewrite measures.');
      return;
    }
    if (dataSource !== 'api') {
      setStatus('error', 'Generating measures is available for backend scores only.');
      return;
    }
    if (!pianoBranch.scoreId || pianoBranch.revision === null) {
      setStatus('error', 'No backend score loaded. Compose first.');
      return;
    }
    const needsSelection =
      operation === 'insert_before' ||
      operation === 'insert_after' ||
      operation === 'replace';
    if (needsSelection && !pianoBranch.selectedMeasureId) {
      setStatus('error', 'Select a measure first.');
      return;
    }

    const count = clampInteger(measureActionCount, 1, MAX_GENERATED_MEASURE_COUNT);
    setMeasureActionCount(count);
    setBusy(true);
    setStatus('busy', `Generating ${count} measure${count === 1 ? '' : 's'}...`);
    try {
      const seed = createRandomSeed();
      const response = await generateMeasures({
        scoreId: pianoBranch.scoreId,
        revision: pianoBranch.revision,
        operation,
        count,
        measureId: needsSelection ? pianoBranch.selectedMeasureId ?? undefined : undefined,
        prompt,
        constraints: {
          engine: compositionEngine,
          texture: voiceCount,
          measures: count,
          seed,
          randomSeed: seed,
          useGrammarMask: true,
          qualityPasses: 4,
          voiceLeading: 'balanced',
        },
        render_mode: instrumentMode,
      });
      const changedMeasureId =
        response.insertedMeasureIds[0] ?? response.changedMeasureIds[0] ?? null;
      const selectedBarIndex = findMeasureBarIndex(
        response.document.views.score.measureMap,
        changedMeasureId,
      );
      resetLoadedScoreUi();
      setState((prev) => ({
        ...prev,
        activeBranch: 'piano',
        piano: {
          ...prev.piano,
          document: response.document,
          revision: response.revision,
          draftId: null,
          draftDocument: null,
          draftBaseRevision: null,
          highlightMeasureId: changedMeasureId,
          selectedMeasureId: changedMeasureId,
          selectedBarIndex,
          lockedEventIds: null,
          changedMeasureIds: response.changedMeasureIds,
        },
      }));
      setLastDiagnostics(response.diagnostics ?? null);
      setStatus('success', 'Generated measure operation applied.');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Generate measures failed.';
      setStatus('error', message);
    } finally {
      setBusy(false);
    }
  };

  const handleExportXml = () => {
    const xml = renderXml;
    if (!xml) {
      setStatus('error', 'No MusicXML to export.');
      return;
    }
    const blob = new Blob([xml], { type: 'application/xml' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `score-${activeBranch?.scoreId ?? state.activeBranch}.musicxml`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleExportMidi = () => {
    if (!activeBranch?.midi) {
      setStatus('error', 'No MIDI available from the backend.');
      return;
    }
    const href = activeBranch.midi.startsWith('data:')
      ? activeBranch.midi
      : `data:audio/midi;base64,${activeBranch.midi}`;
    const link = document.createElement('a');
    link.href = href;
    link.download = `score-${activeBranch.scoreId ?? state.activeBranch}.mid`;
    link.click();
  };

  const handlePlay = () => {
    if (showSheetMusic) {
      sheetPlayerApi?.play();
    } else {
      alphaTabApi?.play();
    }
  };

  const handlePause = () => {
    if (showSheetMusic) {
      sheetPlayerApi?.pause();
    } else {
      alphaTabApi?.pause();
    }
  };

  const handleStop = () => {
    if (showSheetMusic) {
      sheetPlayerApi?.stop();
    } else {
      alphaTabApi?.stop();
    }
    setPlaybackPos(null);
  };

  const formatTime = (ms: number) => {
    const s = Math.floor(ms / 1000);
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
  };

  const statusClass = useMemo(
    () => `status-pill status-pill--${statusTone}`,
    [statusTone],
  );
  const hybridDiagnostics =
    lastDiagnostics?.hybrid && typeof lastDiagnostics.hybrid === 'object'
      ? (lastDiagnostics.hybrid as Record<string, unknown>)
      : null;
  const noveltyDiagnostics =
    lastDiagnostics?.novelty && typeof lastDiagnostics.novelty === 'object'
      ? (lastDiagnostics.novelty as Record<string, unknown>)
      : null;

  return (
    <div className="app">
      <header className="app__header">
        <div className="brand">
          <span className="brand__icon">🎼</span>
          <div className="brand__text">
            <span className="brand__title">Bach Gen</span>
            <span className="brand__tag">Instrumental counterpoint workspace</span>
          </div>
        </div>
        <div className="status">
          <span className={statusClass}>
            <span className="status-pill__icon" />
            {statusMessage}
          </span>
          <span className="status__meta">
            {dataSource === 'local' ? 'Local mode' : 'API mode'}
          </span>
        </div>
      </header>

      <main className="app__main">
        <section className="panel panel--controls">
          {/* Source Selection */}
          <div className="panel__section">
            <div className="panel__title">
              <span className="panel__title-icon">⚡</span>
              Get Started
            </div>
            <div className="control-card">
              <label className="field">
                <span>Data source</span>
                <select
                  value={dataSource}
                  onChange={(event) =>
                    setDataSource(event.target.value as DataSource)
                  }
                >
                  <option value="api">Backend API</option>
                  <option value="local">Local test-data</option>
                </select>
              </label>
              {dataSource === 'api' ? (
                <>
                  <label className="field">
                    <span>Engine</span>
                    <select
                      value={compositionEngine}
                      onChange={(event) =>
                        setCompositionEngine(event.target.value as CompositionEngine)
                      }
                    >
                      <option value="instrumental_v6">Instrumental v6</option>
                      <option value="hybrid">Hybrid</option>
                      <option value="emi">EMI symbolic</option>
                      <option value="transformer">Transformer</option>
                    </select>
                  </label>
                  <div className="field">
                    <span>Voices</span>
                    <div className="segmented-control" role="group" aria-label="Voices">
                      {([1, 2, 3, 4] as VoiceCount[]).map((count) => (
                        <button
                          key={count}
                          type="button"
                          className={`segmented-control__item ${
                            voiceCount === count ? 'segmented-control__item--active' : ''
                          }`}
                          onClick={() => setVoiceCount(count)}
                        >
                          {count}
                        </button>
                      ))}
                    </div>
                  </div>
                  <label className="field">
                    <span>Measures</span>
                    <input
                      type="number"
                      min={1}
                      max={MAX_MEASURE_TARGET}
                      value={measureTarget}
                      onChange={(event) =>
                        setMeasureTarget(
                          clampInteger(
                            Number.parseInt(event.target.value, 10),
                            1,
                            MAX_MEASURE_TARGET,
                          ),
                        )
                      }
                    />
                  </label>
                  <label className="field">
                    <span>Prompt (optional)</span>
                    <textarea
                      value={prompt}
                      onChange={(event) => setPrompt(event.target.value)}
                      placeholder="e.g., Baroque style, minor key, moderate tempo..."
                      rows={2}
                    />
                  </label>
                </>
              ) : (
                <div className="helper-text">
                  Uses local MusicXML files for testing without a backend.
                </div>
              )}
              {dataSource === 'local' && localManifest ? (
                <div className="info-grid">
                  <div className="info-item">
                    <span className="info-label">Base file</span>
                    <span className="info-value">{localManifest.baseScore}</span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">Snippets</span>
                    <span className="info-value">{localManifest.snippets.length}</span>
                  </div>
                </div>
              ) : null}
              {dataSource === 'api' && lastDiagnostics ? (
                <div className="info-grid">
                  <div className="info-item">
                    <span className="info-label">Engine</span>
                    <span className="info-value">{formatDiagnosticValue(lastDiagnostics.engine)}</span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">Retrieved</span>
                    <span className="info-value">
                      {formatDiagnosticValue(hybridDiagnostics?.retrievedFragmentCount)}
                    </span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">Role plan</span>
                    <span className="info-value">
                      {formatDiagnosticValue(hybridDiagnostics?.rolePlan)}
                    </span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">Novelty</span>
                    <span className="info-value">
                      {formatDiagnosticValue(noveltyDiagnostics?.high_copy_risk ? 'risk' : 'pass')}
                    </span>
                  </div>
                  {lastDiagnostics.hybridFallbackReason ? (
                    <div className="info-item">
                      <span className="info-label">Fallback</span>
                      <span className="info-value">
                        {formatDiagnosticValue(lastDiagnostics.hybridFallbackReason)}
                      </span>
                    </div>
                  ) : null}
                </div>
              ) : null}
              <div className="button-row">
                <button
                  className="btn btn--primary"
                  onClick={dataSource === 'local' ? handleLoadLocal : handleCompose}
                  disabled={busy}
                >
                  <span className="btn__icon">{dataSource === 'local' ? '📂' : '✨'}</span>
                  {dataSource === 'local' ? 'Load Test Data' : 'Generate Score'}
                </button>
                <button className="btn btn--ghost" onClick={handleLoadDemo}>
                  <span className="btn__icon">🎵</span>
                  Demo
                </button>
              </div>
            </div>
          </div>
        </section>

          {/* Measure Actions */}
        <aside className="panel panel--actions">
          <div className="panel__section panel__section--measure-actions">
            <div className="panel__title">
              <span className="panel__title-icon">🎨</span>
              Measure Actions
            </div>
            <div className="control-card">
              {activeBranch?.selectedMeasureId ? (
                <div className="selection-badge">
                  <span>📍</span>
                  Bar {activeBranch.selectedBarIndex !== null ? activeBranch.selectedBarIndex + 1 : '?'}
                  {' · '}
                  <span style={{ opacity: 0.7 }}>{activeBranch.selectedMeasureId}</span>
                </div>
              ) : (
                <div className="helper-text">
                  {state.activeBranch === 'piano'
                    ? 'Click a rendered measure to select it.'
                    : 'Switch to Piano to generate or rewrite measures.'}
                </div>
              )}

              <div className="inline-action">
                <label className="field field--inline">
                  <span>Measures</span>
                  <input
                    type="number"
                    min={1}
                    max={MAX_GENERATED_MEASURE_COUNT}
                    value={measureActionCount}
                    onChange={(event) =>
                      setMeasureActionCount(
                        clampInteger(
                          Number.parseInt(event.target.value, 10),
                          1,
                          MAX_GENERATED_MEASURE_COUNT,
                        ),
                      )
                    }
                  />
                </label>
              </div>

              <div className="button-row">
                <button
                  className="btn btn--ghost"
                  onClick={() => handleGenerateMeasures('prepend')}
                  disabled={busy || !canGeneratePianoMeasures}
                >
                  Generate Start
                </button>
                <button
                  className="btn btn--ghost"
                  onClick={() => handleGenerateMeasures('insert_before')}
                  disabled={
                    busy ||
                    !canGeneratePianoMeasures ||
                    !state.piano.selectedMeasureId
                  }
                >
                  Insert Before
                </button>
                <button
                  className="btn btn--ghost"
                  onClick={() => handleGenerateMeasures('replace')}
                  disabled={
                    busy ||
                    !canGeneratePianoMeasures ||
                    !state.piano.selectedMeasureId
                  }
                >
                  Rewrite Selected
                </button>
                <button
                  className="btn btn--ghost"
                  onClick={() => handleGenerateMeasures('insert_after')}
                  disabled={
                    busy ||
                    !canGeneratePianoMeasures ||
                    !state.piano.selectedMeasureId
                  }
                >
                  Insert After
                </button>
                <button
                  className="btn btn--ghost"
                  onClick={() => handleGenerateMeasures('append')}
                  disabled={busy || !canGeneratePianoMeasures}
                >
                  Generate End
                </button>
              </div>

              <div className="helper-text">
                Generated operations splice new model output against the selected score context. Note dragging still needs a dedicated notation-edit API.
              </div>

              {activeBranch?.draftId && (
                <div className="draft-indicator">
                  <span className="draft-indicator__icon">📝</span>
                  <div className="draft-indicator__body">
                    <span className="draft-indicator__text">Draft ready for review</span>
                    {(activeBranch.changedMeasureIds || activeBranch.lockedEventIds) && (
                      <span className="draft-indicator__meta">
                        {[
                          activeBranch.changedMeasureIds &&
                            `${activeBranch.changedMeasureIds.length} measure${activeBranch.changedMeasureIds.length !== 1 ? 's' : ''} changed`,
                          activeBranch.lockedEventIds &&
                            `${activeBranch.lockedEventIds.length} event${activeBranch.lockedEventIds.length !== 1 ? 's' : ''} locked`,
                        ]
                          .filter(Boolean)
                          .join(' · ')}
                      </span>
                    )}
                  </div>
                  <div className="draft-indicator__actions">
                    <button
                      className="btn btn--small btn--primary"
                      onClick={handleCommitDraft}
                      disabled={busy}
                    >
                      ✓ Keep
                    </button>
                    <button
                      className="btn btn--small btn--ghost"
                      onClick={handleDiscardDraft}
                      disabled={busy}
                    >
                      ✕ Discard
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Playback & Export */}
          <div className="panel__section panel__section--playback">
            <div className="panel__title">
              <span className="panel__title-icon">🎧</span>
              Playback & Export
            </div>
            <div className="control-card">
              <div className="playback-bar">
                <div className="playback-bar__controls">
                  <button
                    className="btn btn--icon-only"
                    onClick={handlePlay}
                    disabled={!renderXml || !activePlayerReady}
                    title={activePlayerReady ? 'Play' : 'Loading…'}
                  >
                    ▶
                  </button>
                  <button
                    className="btn btn--icon-only"
                    onClick={handlePause}
                    disabled={!renderXml || !activePlayerReady}
                    title="Pause"
                  >
                    ⏸
                  </button>
                  <button
                    className="btn btn--icon-only"
                    onClick={handleStop}
                    disabled={!renderXml || !activePlayerReady}
                    title="Stop"
                  >
                    ⏹
                  </button>
                </div>
                {playbackPos && (
                  <span className="playback-bar__time">
                    {formatTime(playbackPos.current)} / {formatTime(playbackPos.total)}
                  </span>
                )}
                <div className="playback-bar__divider" />
                <div className="playback-bar__exports">
                  <button
                    className="btn btn--ghost"
                    onClick={handleExportXml}
                    disabled={!renderXml}
                  >
                    📄 XML
                  </button>
                  <button
                    className="btn btn--ghost"
                    onClick={handleExportMidi}
                    disabled={!activeBranch?.midi}
                  >
                    🎹 MIDI
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Fingering Picker */}
          {fingeringPicker && canUseGuitarNoteActions(activeInstrumentMode) && (
            <div className="panel__section panel__section--fingering">
              <div className="panel__title">
                <span className="panel__title-icon">🎸</span>
                Fingering
              </div>
              <div className="control-card">
                <FingeringPicker
                  options={fingeringPicker.options}
                  onSelect={handleApplyFingering}
                  onClose={() => setFingeringPicker(null)}
                  busy={fingeringBusy}
                />
              </div>
            </div>
          )}

        </aside>

        <section className="panel panel--viewer">
          <div
            className="branch-tabs"
            role="tablist"
            aria-label="Score branches"
          >
            <button
              type="button"
              role="tab"
              aria-selected={state.activeBranch === 'piano'}
              className={`branch-tabs__item ${
                state.activeBranch === 'piano' ? 'branch-tabs__item--active' : ''
              }`}
              onClick={() => handleBranchChange('piano')}
            >
              <span className="branch-tabs__label">Piano</span>
              <span className="branch-tabs__meta">
                {state.piano.document ? 'Source score' : 'No score'}
              </span>
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={state.activeBranch === 'guitar'}
              className={`branch-tabs__item ${
                state.activeBranch === 'guitar' ? 'branch-tabs__item--active' : ''
              }`}
              onClick={() => handleBranchChange('guitar')}
            >
              <span className="branch-tabs__label">Guitar Arrangement</span>
              <span className="branch-tabs__meta">
                {hasGuitarBranch ? 'Editable branch' : 'Not converted'}
              </span>
            </button>
          </div>

          {showGuitarEmptyState ? (
            <div className="score-viewer">
              <div className="score-viewer__header">
                <span className="score-viewer__label">
                  <span className="score-viewer__label-icon">🎸</span>
                  Guitar Arrangement
                </span>
              </div>
              <div className="score-viewer__empty">
                <span className="score-viewer__empty-icon">🎸</span>
                <span className="score-viewer__empty-title">
                  Convert the piano score to guitar first.
                </span>
                <span className="score-viewer__empty-subtitle">
                  The guitar branch will become an independent editable arrangement after conversion.
                </span>
              </div>
            </div>
          ) : showSheetMusic ? (
            <SheetMusicViewer
              scoreXml={renderXml}
              highlightMeasureId={activeBranch?.highlightMeasureId ?? null}
              selectedBarIndex={activeBranch?.selectedBarIndex ?? null}
              instrumentMode={activeInstrumentMode}
              viewTab={effectiveViewTab}
              onViewTabChange={setViewTab}
              onMeasureClick={handleMeasureClick}
              onApiReady={setSheetPlayerApi}
              onPlayerReady={() => setSheetPlayerReady(true)}
              onPositionChanged={(current, total) => setPlaybackPos({ current, total })}
            />
          ) : (
            <ScoreViewer
              scoreXml={renderXml}
              highlightMeasureId={activeBranch?.highlightMeasureId ?? null}
              selectedBarIndex={activeBranch?.selectedBarIndex ?? null}
              instrumentMode={activeInstrumentMode}
              viewTab={effectiveViewTab}
              onViewTabChange={setViewTab}
              onMeasureClick={handleMeasureClick}
              onNoteClick={handleNoteClick}
              onApiReady={setAlphaTabApi}
              onPlayerReady={() => setPlayerReady(true)}
              onPositionChanged={(current, total) => setPlaybackPos({ current, total })}
            />
          )}
        </section>

      </main>
    </div>
  );
};

export default App;
