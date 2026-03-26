import { useEffect, useRef, type MutableRefObject } from 'react';
import * as alphaTab from '@coderline/alphatab';
import type { HitKey, InstrumentMode, ScoreViewTab } from '../state/types';
import { VerovioPlayer } from '../playback/VerovioPlayer';
import { createAlphaTabExternalMediaHandler } from '../playback/alphaTabExternalMedia';
import { getVerovioToolkit } from '../playback/verovioToolkit';
import {
  getScoreViewerLabel,
  getStaffVisibility,
  resolveStaveProfile,
  shouldShowTabSwitcher,
} from './scoreViewerStaves';

export interface ScoreViewerPlaybackAPI {
  pause: () => void;
  play: () => void;
  playPause: () => void;
  stop: () => void;
}

interface ScoreViewerPlaybackControllerOptions {
  alphaTabApiRef: MutableRefObject<any>;
  midiPlayerRef: MutableRefObject<VerovioPlayer | null>;
  updatePosition: (timeMs: number) => void;
}

export const createScoreViewerPlaybackController = ({
  alphaTabApiRef,
  midiPlayerRef,
  updatePosition,
}: ScoreViewerPlaybackControllerOptions): ScoreViewerPlaybackAPI => ({
  play: () => {
    if (alphaTabApiRef.current?.play) {
      void alphaTabApiRef.current.play();
      return;
    }
    void midiPlayerRef.current?.play();
  },
  pause: () => {
    midiPlayerRef.current?.pause();
    alphaTabApiRef.current?.pause?.();
  },
  playPause: () => {
    if (alphaTabApiRef.current?.playPause) {
      alphaTabApiRef.current.playPause();
      return;
    }
    void midiPlayerRef.current?.play();
  },
  stop: () => {
    midiPlayerRef.current?.stop();
    alphaTabApiRef.current?.stop?.();
    updatePosition(0);
  },
});

interface ScoreViewerProps {
  scoreXml: string | null;
  highlightMeasureId: string | null;
  instrumentMode: InstrumentMode | null;
  viewTab: ScoreViewTab;
  onViewTabChange?: (viewTab: ScoreViewTab) => void;
  onMeasureClick?: (barIndex: number) => void;
  onNoteClick?: (hit: HitKey) => void;
  onApiReady?: (api: ScoreViewerPlaybackAPI | null) => void;
  onPlayerReady?: () => void;
  onPositionChanged?: (currentTime: number, endTime: number) => void;
}

const loadScoreXml = (api: any, scoreXml: string) => {
  if (!scoreXml || typeof scoreXml !== 'string' || scoreXml.trim().length === 0) {
    console.warn('loadScoreXml: invalid or empty scoreXml');
    return;
  }
  try {
    const encoder = new TextEncoder();
    const data = encoder.encode(scoreXml);
    if (typeof api?.load === 'function') {
      api.load(data);
      return;
    }
    if (typeof api?.render === 'function') {
      api.render(data);
      return;
    }
    console.error('AlphaTab API has no recognized load method');
  } catch (error) {
    console.error('Failed to load score into AlphaTab:', error);
  }
};

const resolveHitKey = (args: any): HitKey | null => {
  const note = args?.note;
  const beat = note?.beat;
  const voice = beat?.voice;
  const bar = voice?.bar;
  const barIndex = bar?.index;
  if (typeof barIndex !== 'number') {
    return null;
  }
  return {
    barIndex,
    voiceIndex: typeof voice?.index === 'number' ? voice.index : undefined,
    beatIndex: typeof beat?.index === 'number' ? beat.index : undefined,
    noteIndex: typeof note?.index === 'number' ? note.index : undefined,
  };
};

const ScoreViewer = ({
  scoreXml,
  highlightMeasureId,
  instrumentMode,
  viewTab,
  onViewTabChange,
  onMeasureClick,
  onNoteClick,
  onApiReady,
  onPlayerReady,
  onPositionChanged,
}: ScoreViewerProps) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const apiRef = useRef<any>(null);
  const midiPlayerRef = useRef<VerovioPlayer | null>(null);
  const alphaTabReadyRef = useRef(false);
  const midiReadyRef = useRef(false);
  const onMeasureClickRef = useRef<ScoreViewerProps['onMeasureClick']>();
  const onNoteClickRef = useRef<ScoreViewerProps['onNoteClick']>();
  const onApiReadyRef = useRef<ScoreViewerProps['onApiReady']>();
  const onPlayerReadyRef = useRef<ScoreViewerProps['onPlayerReady']>();
  const onPositionChangedRef = useRef<ScoreViewerProps['onPositionChanged']>();
  const instrumentModeRef = useRef(instrumentMode);
  const viewTabRef = useRef(viewTab);

  useEffect(() => { onMeasureClickRef.current = onMeasureClick; }, [onMeasureClick]);
  useEffect(() => { onNoteClickRef.current = onNoteClick; }, [onNoteClick]);
  useEffect(() => { onApiReadyRef.current = onApiReady; }, [onApiReady]);
  useEffect(() => { onPlayerReadyRef.current = onPlayerReady; }, [onPlayerReady]);
  useEffect(() => { onPositionChangedRef.current = onPositionChanged; }, [onPositionChanged]);
  useEffect(() => { instrumentModeRef.current = instrumentMode; }, [instrumentMode]);
  useEffect(() => { viewTabRef.current = viewTab; }, [viewTab]);

  const bindExternalPlayer = () => {
    const api = apiRef.current;
    const midiPlayer = midiPlayerRef.current;
    const output = api?.player?.output;

    if (!api || !midiPlayer || !output) {
      return;
    }

    output.handler = createAlphaTabExternalMediaHandler(midiPlayer);
    output.updatePosition?.(0);
  };

  const updateAlphaTabPlaybackPosition = (timeMs: number) => {
    apiRef.current?.player?.output?.updatePosition?.(timeMs);
  };

  const maybeNotifyPlayerReady = () => {
    if (alphaTabReadyRef.current && midiReadyRef.current) {
      onPlayerReadyRef.current?.();
    }
  };

  const applyStaffVisibility = (score: any) => {
    const visibility = getStaffVisibility(
      instrumentModeRef.current ?? null,
      viewTabRef.current,
    );

    for (const track of score?.tracks ?? []) {
      for (const staff of track?.staves ?? []) {
        staff.showStandardNotation = visibility.showStandardNotation;
        staff.showTablature = visibility.showTablature;
      }
    }
  };

  // Mount: create single AlphaTab instance
  useEffect(() => {
    if (!containerRef.current || apiRef.current) {
      return;
    }

    const api = new (alphaTab as any).AlphaTabApi(containerRef.current, {
      core: {
        includeNoteBounds: true,
        fontDirectory: '/font/',
      },
      display: {
        layoutMode:
          (alphaTab as any).LayoutMode?.Parchment ??
          (alphaTab as any).LayoutMode?.Page ??
          0,
        barsPerRow: 4,
        stretchForce: 0.6,
        justifyLastSystem: false,
        padding: [36, 8, 64, 24],
        staveProfile: resolveStaveProfile(alphaTab, instrumentMode, viewTab),
        resources: {
          copyrightFont: '11px Arial',
        },
      },
      player: {
        enablePlayer: true,
        playerMode:
          (alphaTab as any).PlayerMode?.EnabledExternalMedia ??
          (alphaTab as any).PlayerMode?.EnabledSynthesizer,
        enableCursor: true,
        enableElementHighlighting: true,
      },
    });
    apiRef.current = api;
    onApiReadyRef.current?.(
      createScoreViewerPlaybackController({
        alphaTabApiRef: apiRef,
        midiPlayerRef,
        updatePosition: updateAlphaTabPlaybackPosition,
      }),
    );

    if (api?.scoreLoaded?.on) {
      api.scoreLoaded.on((score: any) => {
        applyStaffVisibility(score);
      });
    }

    if (api?.playerReady?.on) {
      api.playerReady.on(() => {
        alphaTabReadyRef.current = true;
        bindExternalPlayer();
        maybeNotifyPlayerReady();
      });
    }

    if (api?.noteMouseDown?.on) {
      api.noteMouseDown.on((args: unknown) => {
        const hit = resolveHitKey(args);
        if (!hit) {
          return;
        }
        onMeasureClickRef.current?.(hit.barIndex);
        onNoteClickRef.current?.(hit);
      });
    }

    if (api?.barMouseDown?.on) {
      api.barMouseDown.on((args: any) => {
        const barIndex = args?.bar?.index;
        if (typeof barIndex === 'number') {
          onMeasureClickRef.current?.(barIndex);
        }
      });
    }

    return () => {
      onApiReadyRef.current?.(null);
      midiPlayerRef.current?.dispose();
      midiPlayerRef.current = null;
      apiRef.current?.destroy?.();
      apiRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!scoreXml || !apiRef.current) {
      midiReadyRef.current = false;
      midiPlayerRef.current?.stop();
      return;
    }

    midiReadyRef.current = false;
    loadScoreXml(apiRef.current, scoreXml);

    let cancelled = false;

    const loadPlaybackMidi = async () => {
      try {
        const toolkit = await getVerovioToolkit();
        if (cancelled) {
          return;
        }

        toolkit.loadData(scoreXml);
        const midiBase64 = toolkit.renderToMIDI();
        if (!midiBase64 || cancelled) {
          return;
        }

        if (!midiPlayerRef.current) {
          midiPlayerRef.current = new VerovioPlayer({
            onPositionChanged: (current, total) => {
              onPositionChangedRef.current?.(current, total);
              updateAlphaTabPlaybackPosition(current);
            },
            onStopped: () => updateAlphaTabPlaybackPosition(0),
          });
        } else {
          midiPlayerRef.current.stop();
        }

        midiPlayerRef.current.load(midiBase64);
        bindExternalPlayer();
        midiReadyRef.current = true;
        maybeNotifyPlayerReady();
      } catch (error) {
        console.error('Failed to create tab playback MIDI:', error);
      }
    };

    void loadPlaybackMidi();

    return () => {
      cancelled = true;
    };
  }, [scoreXml]);
  const label = getScoreViewerLabel(instrumentMode, viewTab);
  const showSwitcher = shouldShowTabSwitcher(instrumentMode);

  return (
    <div className="score-viewer">
      <div className="score-viewer__header">
        <span className="score-viewer__label">
          <span className="score-viewer__label-icon">🎼</span>
          {label}
        </span>
          {showSwitcher ? (
            <div
              className="score-viewer__switcher"
              role="tablist"
              aria-label="Score view"
            >
              <button
                type="button"
                role="tab"
                aria-selected={viewTab === 'score'}
                className={`score-viewer__switch ${viewTab === 'score' ? 'score-viewer__switch--active' : ''}`}
                onClick={() => onViewTabChange?.('score')}
              >
                Sheet Music
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={viewTab === 'tab'}
                className={`score-viewer__switch ${viewTab === 'tab' ? 'score-viewer__switch--active' : ''}`}
                onClick={() => onViewTabChange?.('tab')}
              >
                Guitar Tab
              </button>
            </div>
          ) : null}
        {highlightMeasureId ? (
          <span className="score-viewer__badge">
            <span>✏️</span>
            Draft: {highlightMeasureId}
          </span>
        ) : null}
      </div>
      {!scoreXml ? (
        <div className="score-viewer__empty">
          <span className="score-viewer__empty-icon">🎸</span>
          <span className="score-viewer__empty-title">No score loaded</span>
          <span className="score-viewer__empty-subtitle">
            Generate a new composition or load test data to get started.
          </span>
        </div>
      ) : null}
      <div className="score-viewer__canvas" ref={containerRef} />
    </div>
  );
};

export default ScoreViewer;
