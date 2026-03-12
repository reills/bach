import { useMemo, useState } from 'react';
import './App.css';
import ScoreViewer from './components/ScoreViewer';
import FingeringPicker from './components/FingeringPicker';
import { compose, inpaintPreview, commitDraft, discardDraft, altPositions, applyFingering } from './api/client';
import {
  getEventId,
  getMeasureId,
  type HitKey,
  type ScoreState,
} from './state/types';
import {
  buildMeasureMap,
  loadLocalData,
  parseScoreXml,
  pickRandomSnippet,
  replaceMeasureAtIndex,
  type LocalManifest,
} from './mock/localData';

const DEMO_XML = `<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part-list>
    <score-part id="P1">
      <part-name>Guitar</part-name>
    </score-part>
  </part-list>
  <part id="P1">
    <measure number="1" xml:id="measure-1">
      <attributes>
        <divisions>24</divisions>
        <key>
          <fifths>0</fifths>
        </key>
        <time>
          <beats>4</beats>
          <beat-type>4</beat-type>
        </time>
        <clef>
          <sign>TAB</sign>
          <line>5</line>
        </clef>
      </attributes>
      <note>
        <pitch>
          <step>E</step>
          <octave>4</octave>
        </pitch>
        <duration>24</duration>
        <type>quarter</type>
        <notations>
          <technical>
            <string>1</string>
            <fret>0</fret>
          </technical>
        </notations>
      </note>
      <note>
        <pitch>
          <step>G</step>
          <octave>4</octave>
        </pitch>
        <duration>24</duration>
        <type>quarter</type>
        <notations>
          <technical>
            <string>1</string>
            <fret>3</fret>
          </technical>
        </notations>
      </note>
      <note>
        <pitch>
          <step>A</step>
          <octave>4</octave>
        </pitch>
        <duration>24</duration>
        <type>quarter</type>
        <notations>
          <technical>
            <string>1</string>
            <fret>5</fret>
          </technical>
        </notations>
      </note>
      <note>
        <pitch>
          <step>G</step>
          <octave>4</octave>
        </pitch>
        <duration>24</duration>
        <type>quarter</type>
        <notations>
          <technical>
            <string>1</string>
            <fret>3</fret>
          </technical>
        </notations>
      </note>
    </measure>
    <measure number="2" xml:id="measure-2">
      <note>
        <pitch>
          <step>F</step>
          <octave>4</octave>
        </pitch>
        <duration>24</duration>
        <type>quarter</type>
        <notations>
          <technical>
            <string>1</string>
            <fret>1</fret>
          </technical>
        </notations>
      </note>
      <note>
        <pitch>
          <step>E</step>
          <octave>4</octave>
        </pitch>
        <duration>24</duration>
        <type>quarter</type>
        <notations>
          <technical>
            <string>1</string>
            <fret>0</fret>
          </technical>
        </notations>
      </note>
      <note>
        <pitch>
          <step>D</step>
          <octave>4</octave>
        </pitch>
        <duration>24</duration>
        <type>quarter</type>
        <notations>
          <technical>
            <string>2</string>
            <fret>3</fret>
          </technical>
        </notations>
      </note>
      <note>
        <pitch>
          <step>E</step>
          <octave>4</octave>
        </pitch>
        <duration>24</duration>
        <type>quarter</type>
        <notations>
          <technical>
            <string>1</string>
            <fret>0</fret>
          </technical>
        </notations>
      </note>
    </measure>
  </part>
</score-partwise>
`;

const initialState: ScoreState = {
  scoreId: null,
  revision: null,
  scoreXml: null,
  measureMap: null,
  eventHitMap: null,
  draftId: null,
  draftXml: null,
  draftBaseRevision: null,
  highlightMeasureId: null,
  selectedMeasureId: null,
  selectedBarIndex: null,
  lockedEventIds: null,
  changedMeasureIds: null,
  lastEventId: null,
  midi: null,
};

type StatusTone = 'idle' | 'busy' | 'success' | 'error';
type DataSource = 'api' | 'local';

const defaultSource: DataSource =
  import.meta.env.VITE_USE_LOCAL_DATA === 'true' ? 'local' : 'api';

const App = () => {
  const [state, setState] = useState<ScoreState>(initialState);
  const [alphaTabApi, setAlphaTabApi] = useState<any>(null);
  const [prompt, setPrompt] = useState('');
  const [dataSource, setDataSource] = useState<DataSource>(defaultSource);
  const [localSnippets, setLocalSnippets] = useState<string[]>([]);
  const [localManifest, setLocalManifest] = useState<LocalManifest | null>(
    null,
  );
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

  const renderXml = state.draftXml ?? state.scoreXml;
  const measureCount = state.measureMap ? Object.keys(state.measureMap).length : 0;
  const eventMapCount = state.eventHitMap
    ? Object.keys(state.eventHitMap).length
    : 0;

  const setStatus = (tone: StatusTone, message: string) => {
    setStatusTone(tone);
    setStatusMessage(message);
  };

  const handleCompose = async () => {
    setBusy(true);
    setStatus('busy', 'Composing a new score...');
    try {
      const response = await compose({ prompt });
      setState((prev) => ({
        ...prev,
        scoreId: response.scoreId,
        revision: response.revision,
        scoreXml: response.scoreXML,
        measureMap: response.measureMap ?? null,
        eventHitMap: response.eventHitMap ?? null,
        midi: response.midi ?? null,
        draftId: null,
        draftXml: null,
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
      const baseDoc = parseScoreXml(bundle.baseXml);
      const measureMap = buildMeasureMap(baseDoc);

      setState((prev) => ({
        ...prev,
        scoreId: `local:${bundle.manifest.baseScore}`,
        revision: 0,
        scoreXml: bundle.baseXml,
        measureMap,
        eventHitMap: null,
        midi: null,
        draftId: null,
        draftXml: null,
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
    setState((prev) => ({
      ...prev,
      scoreId: 'demo',
      revision: 0,
      scoreXml: DEMO_XML,
      measureMap: {
        '0': 'demo-measure-1',
        '1': 'demo-measure-2',
      },
      eventHitMap: null,
      draftId: null,
      draftXml: null,
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
    const measureId = getMeasureId(state.measureMap, barIndex);
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
    const eventId = getEventId(state.eventHitMap, hit);
    setState((prev) => ({
      ...prev,
      lastEventId: eventId,
    }));

    if (!eventId || !state.scoreId) return;
    const measureId = getMeasureId(state.measureMap, hit.barIndex);
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
        scoreXml: resp.scoreXML,
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
      if (!state.scoreXml) {
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
          state.scoreXml,
          state.selectedBarIndex,
          snippet,
        );

        setState((prev) => ({
          ...prev,
          draftId: `local-${Date.now()}`,
          draftXml: result.xml,
          draftBaseRevision: prev.revision,
          highlightMeasureId:
            result.measureId ?? prev.selectedMeasureId ?? null,
          measureMap: result.measureMap,
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
        draftXml: response.scoreXML,
        draftBaseRevision: response.baseRevision,
        highlightMeasureId: response.highlightMeasureId ?? null,
        measureMap: response.measureMap ?? prev.measureMap,
        eventHitMap: response.eventHitMap ?? prev.eventHitMap,
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
      if (!state.draftXml) {
        setStatus('error', 'No draft to commit.');
        return;
      }
      setState((prev) => ({
        ...prev,
        scoreXml: prev.draftXml,
        revision: (prev.revision ?? 0) + 1,
        draftId: null,
        draftXml: null,
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
        scoreXml: response.scoreXML,
        revision: response.revision,
        measureMap: response.measureMap ?? prev.measureMap,
        eventHitMap: response.eventHitMap ?? prev.eventHitMap,
        draftId: null,
        draftXml: null,
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
      if (!state.draftXml) {
        setStatus('error', 'No draft to discard.');
        return;
      }
      setState((prev) => ({
        ...prev,
        draftId: null,
        draftXml: null,
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
        draftXml: null,
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

  const handlePlay = () => {
    if (!alphaTabApi) {
      setStatus('error', 'AlphaTab not ready yet.');
      return;
    }
    const player = alphaTabApi.player ?? alphaTabApi.Player;
    if (player?.play) {
      player.play();
    } else if (alphaTabApi.play) {
      alphaTabApi.play();
    }
  };

  const handlePause = () => {
    const player = alphaTabApi?.player ?? alphaTabApi?.Player;
    if (player?.pause) {
      player.pause();
    } else if (alphaTabApi?.pause) {
      alphaTabApi.pause();
    }
  };

  const handleStop = () => {
    const player = alphaTabApi?.player ?? alphaTabApi?.Player;
    if (player?.stop) {
      player.stop();
    } else if (alphaTabApi?.stop) {
      alphaTabApi.stop();
    }
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
                <label className="field">
                  <span>Prompt (optional)</span>
                  <textarea
                    value={prompt}
                    onChange={(event) => setPrompt(event.target.value)}
                    placeholder="e.g., Baroque style, minor key, moderate tempo..."
                    rows={2}
                  />
                </label>
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
          {fingeringPicker && (
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
                    disabled={!renderXml}
                    title="Play"
                  >
                    ▶
                  </button>
                  <button
                    className="btn btn--icon-only"
                    onClick={handlePause}
                    disabled={!renderXml}
                    title="Pause"
                  >
                    ⏸
                  </button>
                  <button
                    className="btn btn--icon-only"
                    onClick={handleStop}
                    disabled={!renderXml}
                    title="Stop"
                  >
                    ⏹
                  </button>
                </div>
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
          <ScoreViewer
            scoreXml={renderXml}
            highlightMeasureId={state.highlightMeasureId}
            onMeasureClick={handleMeasureClick}
            onNoteClick={handleNoteClick}
            onApiReady={setAlphaTabApi}
          />
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
