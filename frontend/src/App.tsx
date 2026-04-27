import { useMemo, useState } from 'react';
import './App.css';
import ScoreViewer, { type ScoreViewerPlaybackAPI } from './components/ScoreViewer';
import SheetMusicViewer, { type SheetMusicPlayerAPI } from './components/SheetMusicViewer';
import FingeringPicker from './components/FingeringPicker';
import { compose, inpaintPreview, commitDraft, discardDraft, altPositions, applyFingering } from './api/client';
import {
  canUseGuitarNoteActions,
  getActiveRenderView,
  getEventId,
  getMeasureId,
  inferInstrumentMode,
  type HitKey,
  type InstrumentMode,
  type ScoreViewTab,
  type ScoreState,
} from './state/types';
import {
  buildDocumentBundle,
  loadLocalData,
  pickRandomSnippet,
  replaceMeasureAtIndex,
  type LocalManifest,
} from './mock/localData';

const DEMO_XML = `<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part-list>
    <score-part id="P1">
      <part-name>Classical Guitar</part-name>
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
        <clef>
          <sign>G</sign>
          <line>2</line>
          <clef-octave-change>-1</clef-octave-change>
        </clef>
        <staff-details>
          <staff-lines>6</staff-lines>
          <staff-tuning line="1">
            <tuning-step>E</tuning-step>
            <tuning-octave>4</tuning-octave>
          </staff-tuning>
          <staff-tuning line="2">
            <tuning-step>B</tuning-step>
            <tuning-octave>3</tuning-octave>
          </staff-tuning>
          <staff-tuning line="3">
            <tuning-step>G</tuning-step>
            <tuning-octave>3</tuning-octave>
          </staff-tuning>
          <staff-tuning line="4">
            <tuning-step>D</tuning-step>
            <tuning-octave>3</tuning-octave>
          </staff-tuning>
          <staff-tuning line="5">
            <tuning-step>A</tuning-step>
            <tuning-octave>2</tuning-octave>
          </staff-tuning>
          <staff-tuning line="6">
            <tuning-step>E</tuning-step>
            <tuning-octave>2</tuning-octave>
          </staff-tuning>
        </staff-details>
      </attributes>
      <note>
        <pitch><step>E</step><octave>4</octave></pitch>
        <duration>4</duration>
        <type>quarter</type>
        <notations><technical><string>1</string><fret>0</fret></technical></notations>
      </note>
      <note>
        <pitch><step>D</step><octave>4</octave></pitch>
        <duration>4</duration>
        <type>quarter</type>
        <notations><technical><string>2</string><fret>3</fret></technical></notations>
      </note>
      <note>
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>4</duration>
        <type>quarter</type>
        <notations><technical><string>2</string><fret>1</fret></technical></notations>
      </note>
      <note>
        <pitch><step>B</step><octave>3</octave></pitch>
        <duration>4</duration>
        <type>quarter</type>
        <notations><technical><string>2</string><fret>0</fret></technical></notations>
      </note>
    </measure>
    <measure number="2" xml:id="measure-2">
      <note>
        <pitch><step>A</step><octave>3</octave></pitch>
        <duration>4</duration>
        <type>quarter</type>
        <notations><technical><string>3</string><fret>2</fret></technical></notations>
      </note>
      <note>
        <pitch><step>G</step><octave>3</octave></pitch>
        <duration>4</duration>
        <type>quarter</type>
        <notations><technical><string>3</string><fret>0</fret></technical></notations>
      </note>
      <note>
        <pitch><step>A</step><octave>3</octave></pitch>
        <duration>4</duration>
        <type>quarter</type>
        <notations><technical><string>3</string><fret>2</fret></technical></notations>
      </note>
      <note>
        <pitch><step>B</step><octave>3</octave></pitch>
        <duration>4</duration>
        <type>quarter</type>
        <notations><technical><string>2</string><fret>0</fret></technical></notations>
      </note>
    </measure>
    <measure number="3" xml:id="measure-3">
      <note>
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>8</duration>
        <type>half</type>
        <notations><technical><string>2</string><fret>1</fret></technical></notations>
      </note>
      <note>
        <pitch><step>B</step><octave>3</octave></pitch>
        <duration>8</duration>
        <type>half</type>
        <notations><technical><string>2</string><fret>0</fret></technical></notations>
      </note>
    </measure>
    <measure number="4" xml:id="measure-4">
      <note>
        <pitch><step>E</step><octave>3</octave></pitch>
        <duration>16</duration>
        <type>whole</type>
        <notations><technical><string>4</string><fret>2</fret></technical></notations>
      </note>
    </measure>
  </part>
</score-partwise>
`;

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

type StatusTone = 'idle' | 'busy' | 'success' | 'error';
type DataSource = 'api' | 'local';

const defaultSource: DataSource =
  import.meta.env.VITE_USE_LOCAL_DATA === 'true' ? 'local' : 'api';

const App = () => {
  const [state, setState] = useState<ScoreState>(initialState);
  const [alphaTabApi, setAlphaTabApi] = useState<ScoreViewerPlaybackAPI | null>(null);
  const [playerReady, setPlayerReady] = useState(false);
  const [sheetPlayerApi, setSheetPlayerApi] = useState<SheetMusicPlayerAPI | null>(null);
  const [sheetPlayerReady, setSheetPlayerReady] = useState(false);
  const [playbackPos, setPlaybackPos] = useState<{ current: number; total: number } | null>(null);
  const [prompt, setPrompt] = useState('');
  const [dataSource, setDataSource] = useState<DataSource>(defaultSource);
  const [localSnippets, setLocalSnippets] = useState<string[]>([]);
  const [localManifest, setLocalManifest] = useState<LocalManifest | null>(
    null,
  );
  const [instrumentMode, setInstrumentMode] = useState<InstrumentMode>('guitar');
  const [viewTab, setViewTab] = useState<ScoreViewTab>('score');
  const [mode, setMode] = useState<'window' | 'repair'>('window');
  const [constraints, setConstraints] = useState({
    keepHarmony: false,
    keepRhythm: false,
    keepSoprano: false,
  });
  const [statusTone, setStatusTone] = useState<StatusTone>('idle');
  const [statusMessage, setStatusMessage] = useState('Ready.');
  const [busy, setBusy] = useState(false);
  const [fingeringPicker, setFingeringPicker] = useState<{
    eventId: string;
    options: Array<{ stringIndex: number; fret: number; selected: boolean }>;
  } | null>(null);
  const [fingeringBusy, setFingeringBusy] = useState(false);

  const activeDocument = state.draftDocument ?? state.document;
  const effectiveViewTab =
    viewTab === 'tab' && activeDocument?.views.tab ? 'tab' : 'score';
  const activeView = getActiveRenderView(activeDocument, effectiveViewTab);
  const renderXml = activeView?.xml ?? null;
  const showSheetMusic =
    activeDocument?.instrumentMode === 'piano' ||
    (activeDocument?.instrumentMode === 'guitar' && effectiveViewTab === 'score');
  const measureCount = activeView?.measureMap ? Object.keys(activeView.measureMap).length : 0;
  const eventMapCount = activeView?.eventHitMap
    ? Object.keys(activeView.eventHitMap).length
    : 0;

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

  const handleCompose = async () => {
    setBusy(true);
    setStatus('busy', 'Composing a new score...');
    try {
      const response = await compose({
        prompt,
        constraints: { useGrammarMask: true },
        render_mode: instrumentMode,
      });
      resetLoadedScoreUi();
      setState((prev) => ({
        ...prev,
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
      }));
      setStatus('success', 'Score loaded. Click a measure to inpaint.');
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

      resetLoadedScoreUi();
      setState((prev) => ({
        ...prev,
        scoreId: `local:${bundle.manifest.baseScore}`,
        revision: 0,
        document: buildDocumentBundle(bundle.baseXml, loadedInstrumentMode),
        midi: null,
        instrumentMode: loadedInstrumentMode,
        draftId: null,
        draftDocument: null,
        draftBaseRevision: null,
        highlightMeasureId: null,
        selectedMeasureId: null,
        selectedBarIndex: null,
        lockedEventIds: null,
        changedMeasureIds: null,
        lastEventId: null,
      }));

      setLocalSnippets(bundle.snippetXmls);
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
    resetLoadedScoreUi();
    setState((prev) => ({
      ...prev,
      scoreId: 'demo',
      revision: 0,
      document: buildDocumentBundle(DEMO_XML, inferInstrumentMode(DEMO_XML)),
      instrumentMode: inferInstrumentMode(DEMO_XML),
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
    }
    setState((prev) => ({
      ...prev,
      selectedBarIndex: barIndex,
      selectedMeasureId: measureId,
    }));
  };

  const handleNoteClick = async (hit: HitKey) => {
    const eventId = getEventId(activeView?.eventHitMap, hit);
    setState((prev) => ({
      ...prev,
      lastEventId: eventId,
    }));

    if (!canUseGuitarNoteActions(state.instrumentMode)) {
      setFingeringPicker(null);
      return;
    }

    if (!eventId || !state.scoreId) return;
    const measureId = getMeasureId(activeView?.measureMap, hit.barIndex);
    if (!measureId) return;

    setFingeringBusy(true);
    try {
      const resp = await altPositions({ scoreId: state.scoreId, measureId, eventHitKey: hit });
      setFingeringPicker({ eventId: resp.eventId, options: resp.options });
    } catch {
      // Note has no alternate positions data — silently ignore
    } finally {
      setFingeringBusy(false);
    }
  };

  const handleApplyFingering = async (stringIndex: number, fret: number) => {
    if (!fingeringPicker || !state.scoreId || state.revision === null) return;
    setFingeringBusy(true);
    try {
      const resp = await applyFingering({
        scoreId: state.scoreId,
        revision: state.revision,
        fingeringSelections: [{ eventId: fingeringPicker.eventId, stringIndex, fret }],
      });
      setState((prev) => ({
        ...prev,
        document: resp.document,
        instrumentMode: resp.document.instrumentMode,
        revision: resp.revision,
      }));
      setFingeringPicker(null);
      setStatus('success', 'Fingering applied.');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Apply fingering failed.';
      setStatus('error', message);
    } finally {
      setFingeringBusy(false);
    }
  };

  const handleInpaintPreview = async () => {
    if (dataSource === 'local') {
      if (!state.document?.views.score.xml) {
        setStatus('error', 'Load local test-data first.');
        return;
      }
      if (state.selectedBarIndex === null) {
        setStatus('error', 'Select a measure to inpaint.');
        return;
      }
      if (!localSnippets.length) {
        setStatus('error', 'No snippet measures loaded.');
        return;
      }

      setBusy(true);
      setStatus('busy', 'Splicing a random local measure...');
      try {
        const snippet = pickRandomSnippet(localSnippets);
        if (!snippet) {
          throw new Error('No snippet measures available.');
        }
        const result = replaceMeasureAtIndex(
          state.document.views.tab?.xml ?? state.document.views.score.xml,
          state.selectedBarIndex,
          snippet,
        );

        setState((prev) => ({
          ...prev,
          draftId: `local-${Date.now()}`,
          draftDocument: buildDocumentBundle(
            result.xml,
            prev.instrumentMode ?? 'guitar',
          ),
          draftBaseRevision: prev.revision,
          highlightMeasureId:
            result.measureId ?? prev.selectedMeasureId ?? null,
          lockedEventIds: null,
          changedMeasureIds: result.measureId
            ? [result.measureId]
            : prev.selectedMeasureId
            ? [prev.selectedMeasureId]
            : null,
        }));
        setStatus('success', 'Local draft ready. Commit or discard.');
      } catch (error) {
        const message =
          error instanceof Error ? error.message : 'Local inpaint failed.';
        setStatus('error', message);
      } finally {
        setBusy(false);
      }
      return;
    }

    if (!state.scoreId || state.revision === null) {
      setStatus('error', 'No score loaded. Compose first.');
      return;
    }
    if (!state.selectedMeasureId) {
      setStatus('error', 'Select a measure to inpaint.');
      return;
    }

    setBusy(true);
    setStatus('busy', 'Requesting inpaint preview...');
    try {
      const response = await inpaintPreview({
        scoreId: state.scoreId,
        measureId: state.selectedMeasureId,
        revision: state.revision,
        constraints,
        mode,
      });
      setState((prev) => ({
        ...prev,
        draftId: response.draftId,
        draftDocument: response.document,
        draftBaseRevision: response.baseRevision,
        highlightMeasureId: response.highlightMeasureId ?? null,
        lockedEventIds: response.lockedEventIds ?? null,
        changedMeasureIds: response.changedMeasureIds ?? null,
      }));
      setStatus('success', 'Draft ready. Compare and commit or discard.');
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Inpaint preview failed.';
      setStatus('error', message);
    } finally {
      setBusy(false);
    }
  };

  const handleCommitDraft = async () => {
    if (dataSource === 'local') {
      if (!state.draftDocument) {
        setStatus('error', 'No draft to commit.');
        return;
      }
      setState((prev) => ({
        ...prev,
        document: prev.draftDocument,
        revision: (prev.revision ?? 0) + 1,
        draftId: null,
        draftDocument: null,
        draftBaseRevision: null,
        highlightMeasureId: null,
        lockedEventIds: null,
        changedMeasureIds: null,
      }));
      setStatus('success', 'Local draft committed.');
      return;
    }

    if (!state.scoreId || !state.draftId) {
      setStatus('error', 'No draft to commit.');
      return;
    }

    setBusy(true);
    setStatus('busy', 'Committing draft...');
    try {
      const response = await commitDraft({
        scoreId: state.scoreId,
        draftId: state.draftId,
      });
      setState((prev) => ({
        ...prev,
        document: response.document,
        instrumentMode: response.document.instrumentMode,
        revision: response.revision,
        draftId: null,
        draftDocument: null,
        draftBaseRevision: null,
        highlightMeasureId: null,
        lockedEventIds: null,
        changedMeasureIds: null,
      }));
      setStatus('success', 'Draft committed.');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Commit failed.';
      setStatus('error', message);
    } finally {
      setBusy(false);
    }
  };

  const handleDiscardDraft = async () => {
    if (dataSource === 'local') {
      if (!state.draftDocument) {
        setStatus('error', 'No draft to discard.');
        return;
      }
      setState((prev) => ({
        ...prev,
        draftId: null,
        draftDocument: null,
        draftBaseRevision: null,
        highlightMeasureId: null,
        lockedEventIds: null,
        changedMeasureIds: null,
      }));
      setStatus('success', 'Local draft discarded.');
      return;
    }

    if (!state.scoreId || !state.draftId) {
      setStatus('error', 'No draft to discard.');
      return;
    }

    setBusy(true);
    setStatus('busy', 'Discarding draft...');
    try {
      await discardDraft({
        scoreId: state.scoreId,
        draftId: state.draftId,
      });
      setState((prev) => ({
        ...prev,
        draftId: null,
        draftDocument: null,
        draftBaseRevision: null,
        highlightMeasureId: null,
        lockedEventIds: null,
        changedMeasureIds: null,
      }));
      setStatus('success', 'Draft discarded.');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Discard failed.';
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
    link.download = `score-${state.scoreId ?? 'draft'}.musicxml`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleExportMidi = () => {
    if (!state.midi) {
      setStatus('error', 'No MIDI available from the backend.');
      return;
    }
    const href = state.midi.startsWith('data:')
      ? state.midi
      : `data:audio/midi;base64,${state.midi}`;
    const link = document.createElement('a');
    link.href = href;
    link.download = `score-${state.scoreId ?? 'draft'}.mid`;
    link.click();
  };

  const activePlayerReady = showSheetMusic ? sheetPlayerReady : playerReady;

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

  const workflowStep = useMemo(() => {
    if (!state.scoreId) return 0;
    if (!state.selectedMeasureId) return 1;
    if (!state.draftId) return 2;
    return 3;
  }, [state.scoreId, state.selectedMeasureId, state.draftId]);

  return (
    <div className="app">
      <header className="app__header">
        <div className="brand">
          <span className="brand__icon">🎸</span>
          <div className="brand__text">
            <span className="brand__title">Bach Gen</span>
            <span className="brand__tag">AI-powered guitar counterpoint</span>
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
                    <span>Instrument</span>
                    <select
                      value={instrumentMode}
                      onChange={(event) =>
                        setInstrumentMode(event.target.value as InstrumentMode)
                      }
                    >
                      <option value="guitar">Guitar</option>
                      <option value="piano">Piano</option>
                    </select>
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

          {/* Inpaint Section */}
          <div className="panel__section">
            <div className="panel__title">
              <span className="panel__title-icon">🎨</span>
              Inpaint Measure
            </div>
            <div className="control-card">
              {state.selectedMeasureId ? (
                <div className="selection-badge">
                  <span>📍</span>
                  Bar {state.selectedBarIndex !== null ? state.selectedBarIndex + 1 : '?'}
                  {' · '}
                  <span style={{ opacity: 0.7 }}>{state.selectedMeasureId}</span>
                </div>
              ) : (
                <div className="helper-text">
                  👆 Click a measure in the score to select it for inpainting.
                </div>
              )}

              <div className="toggle-grid">
                <label className={`toggle-chip ${constraints.keepHarmony ? 'toggle-chip--active' : ''}`}>
                  <span className="toggle-chip__check">{constraints.keepHarmony ? '✓' : ''}</span>
                  <input
                    type="checkbox"
                    checked={constraints.keepHarmony}
                    onChange={(event) =>
                      setConstraints((prev) => ({
                        ...prev,
                        keepHarmony: event.target.checked,
                      }))
                    }
                  />
                  Harmony
                </label>
                <label className={`toggle-chip ${constraints.keepRhythm ? 'toggle-chip--active' : ''}`}>
                  <span className="toggle-chip__check">{constraints.keepRhythm ? '✓' : ''}</span>
                  <input
                    type="checkbox"
                    checked={constraints.keepRhythm}
                    onChange={(event) =>
                      setConstraints((prev) => ({
                        ...prev,
                        keepRhythm: event.target.checked,
                      }))
                    }
                  />
                  Rhythm
                </label>
                <label className={`toggle-chip ${constraints.keepSoprano ? 'toggle-chip--active' : ''}`}>
                  <span className="toggle-chip__check">{constraints.keepSoprano ? '✓' : ''}</span>
                  <input
                    type="checkbox"
                    checked={constraints.keepSoprano}
                    onChange={(event) =>
                      setConstraints((prev) => ({
                        ...prev,
                        keepSoprano: event.target.checked,
                      }))
                    }
                  />
                  Soprano
                </label>
              </div>

              <label className="field">
                <span>Mode</span>
                <select
                  value={mode}
                  onChange={(event) =>
                    setMode(event.target.value as 'window' | 'repair')
                  }
                >
                  <option value="window">Window (regenerate)</option>
                  <option value="repair">Repair (fix issues)</option>
                </select>
              </label>

              <button
                className="btn btn--primary"
                onClick={handleInpaintPreview}
                disabled={busy || !state.scoreId || !state.selectedMeasureId}
                style={{ width: '100%' }}
              >
                <span className="btn__icon">🔄</span>
                Generate Preview
              </button>

              {state.draftId && (
                <div className="draft-indicator">
                  <span className="draft-indicator__icon">📝</span>
                  <div className="draft-indicator__body">
                    <span className="draft-indicator__text">Draft ready for review</span>
                    {(state.changedMeasureIds || state.lockedEventIds) && (
                      <span className="draft-indicator__meta">
                        {[
                          state.changedMeasureIds &&
                            `${state.changedMeasureIds.length} measure${state.changedMeasureIds.length !== 1 ? 's' : ''} changed`,
                          state.lockedEventIds &&
                            `${state.lockedEventIds.length} event${state.lockedEventIds.length !== 1 ? 's' : ''} locked`,
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

          {/* Fingering Picker */}
          {fingeringPicker && canUseGuitarNoteActions(state.instrumentMode) && (
            <div className="panel__section">
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

          {/* Playback & Export */}
          <div className="panel__section">
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
                    disabled={!state.midi}
                  >
                    🎹 MIDI
                  </button>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="panel panel--viewer">
          {showSheetMusic ? (
            <SheetMusicViewer
              scoreXml={renderXml}
              highlightMeasureId={state.highlightMeasureId}
              instrumentMode={activeDocument?.instrumentMode ?? state.instrumentMode}
              viewTab={effectiveViewTab}
              onViewTabChange={setViewTab}
              onApiReady={setSheetPlayerApi}
              onPlayerReady={() => setSheetPlayerReady(true)}
              onPositionChanged={(current, total) => setPlaybackPos({ current, total })}
            />
          ) : (
            <ScoreViewer
              scoreXml={renderXml}
              highlightMeasureId={state.highlightMeasureId}
              instrumentMode={activeDocument?.instrumentMode ?? state.instrumentMode}
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

        <aside className="panel panel--meta">
          {/* Workflow Stepper */}
          <div className="panel__section">
            <div className="panel__title">
              <span className="panel__title-icon">📋</span>
              Workflow
            </div>
            <div className="workflow-stepper">
              <div className={`workflow-step ${workflowStep > 0 ? 'workflow-step--complete' : ''} ${workflowStep === 0 ? 'workflow-step--active' : ''}`}>
                <div className="workflow-step__number">{workflowStep > 0 ? '✓' : '1'}</div>
                <div className="workflow-step__content">
                  <div className="workflow-step__label">Load or compose a score</div>
                </div>
              </div>
              <div className={`workflow-step ${workflowStep > 1 ? 'workflow-step--complete' : ''} ${workflowStep === 1 ? 'workflow-step--active' : ''}`}>
                <div className="workflow-step__number">{workflowStep > 1 ? '✓' : '2'}</div>
                <div className="workflow-step__content">
                  <div className="workflow-step__label">Click a measure to select</div>
                </div>
              </div>
              <div className={`workflow-step ${workflowStep > 2 ? 'workflow-step--complete' : ''} ${workflowStep === 2 ? 'workflow-step--active' : ''}`}>
                <div className="workflow-step__number">{workflowStep > 2 ? '✓' : '3'}</div>
                <div className="workflow-step__content">
                  <div className="workflow-step__label">Generate an inpaint preview</div>
                </div>
              </div>
              <div className={`workflow-step ${workflowStep === 3 ? 'workflow-step--active' : ''}`}>
                <div className="workflow-step__number">4</div>
                <div className="workflow-step__content">
                  <div className="workflow-step__label">Keep or discard changes</div>
                </div>
              </div>
            </div>
          </div>

          {/* Session Info */}
          <div className="panel__section">
            <div className="panel__title">
              <span className="panel__title-icon">💾</span>
              Session
            </div>
            <div className="meta-card">
              <div className="info-grid">
                <div className="info-item">
                  <span className="info-label">Score</span>
                  <span className={`info-value ${state.scoreId ? '' : 'info-value--muted'}`}>
                    {state.scoreId ? (state.scoreId.length > 12 ? state.scoreId.slice(0, 12) + '…' : state.scoreId) : 'None'}
                  </span>
                </div>
                <div className="info-item">
                  <span className="info-label">Revision</span>
                  <span className={`info-value ${state.revision !== null ? 'info-value--highlight' : 'info-value--muted'}`}>
                    {state.revision !== null ? `v${state.revision}` : '—'}
                  </span>
                </div>
                <div className="info-item">
                  <span className="info-label">Measures</span>
                  <span className="info-value">
                    {measureCount || '—'}
                  </span>
                </div>
                <div className="info-item">
                  <span className="info-label">Events</span>
                  <span className="info-value">
                    {eventMapCount || '—'}
                  </span>
                </div>
              </div>
              {state.draftId && (
                <>
                  <div className="divider" />
                  <div className="info-item">
                    <span className="info-label">Active Draft</span>
                    <span className="info-value info-value--highlight">
                      {state.draftId.length > 16 ? state.draftId.slice(0, 16) + '…' : state.draftId}
                    </span>
                  </div>
                </>
              )}
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
};

export default App;
