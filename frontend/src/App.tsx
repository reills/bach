import { useMemo, useState } from 'react';
import './App.css';
import ScoreViewer from './components/ScoreViewer';
import { compose, inpaintPreview, commitDraft, discardDraft } from './api/client';
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

  const handleNoteClick = (hit: HitKey) => {
    const eventId = getEventId(state.eventHitMap, hit);
    setState((prev) => ({
      ...prev,
      lastEventId: eventId,
    }));
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

  return (
    <div className="app">
      <header className="app__header">
        <div className="brand">
          <span className="brand__title">Bach Gen</span>
          <span className="brand__tag">Inpainted guitar counterpoint</span>
        </div>
        <div className="status">
          <span className={statusClass}>{statusMessage}</span>
          <span className="status__meta">
            Source: {dataSource} · API:{' '}
            {import.meta.env.VITE_API_BASE_URL ?? 'same-origin'}
          </span>
        </div>
      </header>

      <main className="app__main">
        <section className="panel panel--controls">
          <div className="panel__title">Compose</div>
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
              <span>Prompt</span>
              <textarea
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                placeholder="Describe the mood, tempo, or constraints."
                rows={3}
              />
            </label>
            ) : (
              <div className="helper-text">
                Load `/public/test-data/manifest.json` to mock inpaint drafts
                without the backend.
              </div>
            )}
            {dataSource === 'local' ? (
              <div className="info-grid">
                <div>
                  <span className="info-label">Base file</span>
                  <span className="info-value">
                    {localManifest?.baseScore ?? 'manifest missing'}
                  </span>
                </div>
                <div>
                  <span className="info-label">Snippets</span>
                  <span className="info-value">
                    {localManifest?.snippets.length ?? 0}
                  </span>
                </div>
              </div>
            ) : null}
            <div className="button-row">
              <button
                className="btn btn--primary"
                onClick={dataSource === 'local' ? handleLoadLocal : handleCompose}
                disabled={busy}
              >
                {dataSource === 'local' ? 'Load test-data' : 'Compose'}
              </button>
              <button className="btn btn--ghost" onClick={handleLoadDemo}>
                Load demo
              </button>
            </div>
          </div>

          <div className="panel__title">Inpaint Draft</div>
          <div className="control-card">
            <div className="info-grid">
              <div>
                <span className="info-label">Selected measure</span>
                <span className="info-value">
                  {state.selectedMeasureId ?? 'none'}
                </span>
              </div>
              <div>
                <span className="info-label">Bar index</span>
                <span className="info-value">
                  {state.selectedBarIndex ?? '--'}
                </span>
              </div>
            </div>

            <div className="toggle-grid">
              <label>
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
                Keep harmony
              </label>
              <label>
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
                Keep rhythm
              </label>
              <label>
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
                Keep soprano
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
                <option value="window">Window</option>
                <option value="repair">Repair</option>
              </select>
            </label>

            <div className="button-row">
              <button
                className="btn btn--primary"
                onClick={handleInpaintPreview}
                disabled={busy || !state.scoreId}
              >
                Inpaint preview
              </button>
              <button
                className="btn btn--ghost"
                onClick={handleCommitDraft}
                disabled={busy || !state.draftId}
              >
                Commit
              </button>
              <button
                className="btn btn--ghost"
                onClick={handleDiscardDraft}
                disabled={busy || !state.draftId}
              >
                Discard
              </button>
            </div>
          </div>

          <div className="panel__title">Playback + Export</div>
          <div className="control-card">
            <div className="button-row">
              <button className="btn" onClick={handlePlay}>
                Play
              </button>
              <button className="btn" onClick={handlePause}>
                Pause
              </button>
              <button className="btn" onClick={handleStop}>
                Stop
              </button>
            </div>
            <div className="button-row">
              <button className="btn btn--ghost" onClick={handleExportXml}>
                Export MusicXML
              </button>
              <button
                className="btn btn--ghost"
                onClick={handleExportMidi}
                disabled={!state.midi}
              >
                Export MIDI
              </button>
            </div>
            <div className="helper-text">
              AlphaTab player is browser-only. Load a score before playback.
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
          <div className="panel__title">Session</div>
          <div className="meta-card">
            <div>
              <span className="info-label">Score ID</span>
              <span className="info-value">{state.scoreId ?? 'none'}</span>
            </div>
            <div>
              <span className="info-label">Revision</span>
              <span className="info-value">
                {state.revision ?? 'unassigned'}
              </span>
            </div>
            <div>
              <span className="info-label">Draft ID</span>
              <span className="info-value">{state.draftId ?? 'none'}</span>
            </div>
            <div>
              <span className="info-label">Measure map</span>
              <span className="info-value">
                {measureCount ? `${measureCount} measures` : 'missing'}
              </span>
            </div>
            <div>
              <span className="info-label">Event map</span>
              <span className="info-value">
                {eventMapCount ? `${eventMapCount} hits` : 'missing'}
              </span>
            </div>
            <div>
              <span className="info-label">Last event</span>
              <span className="info-value">{state.lastEventId ?? '--'}</span>
            </div>
          </div>

          <div className="panel__title">Draft locks</div>
          <div className="meta-card">
            <div>
              <span className="info-label">Locked events</span>
              <span className="info-value">
                {state.lockedEventIds?.length ?? 0}
              </span>
            </div>
            <div className="helper-text">
              Carry-in notes should appear here once the API responds.
            </div>
          </div>

          <div className="panel__title">Workflow</div>
          <div className="meta-card">
            <ol>
              <li>Compose or load a score.</li>
              <li>Click a measure in the viewer.</li>
              <li>Preview an inpaint draft.</li>
              <li>Commit or discard.</li>
            </ol>
          </div>
        </aside>
      </main>
    </div>
  );
};

export default App;
