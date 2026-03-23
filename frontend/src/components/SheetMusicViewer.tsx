import { useEffect, useRef } from 'react';
import createVerovioModule from 'verovio/wasm';
import { VerovioToolkit } from 'verovio/esm';
import type { ScoreViewTab } from '../state/types';
import { VerovioPlayer } from '../playback/VerovioPlayer';

export interface SheetMusicPlayerAPI {
  playPause: () => void;
  stop: () => void;
}

interface SheetMusicViewerProps {
  scoreXml: string | null;
  highlightMeasureId: string | null;
  instrumentMode: 'guitar' | 'piano' | null;
  viewTab: ScoreViewTab;
  onViewTabChange?: (tab: ScoreViewTab) => void;
  onPlayerReady?: () => void;
  onPositionChanged?: (currentMs: number, totalMs: number) => void;
  onApiReady?: (api: SheetMusicPlayerAPI) => void;
}

type VerovioToolkitLike = Pick<
  VerovioToolkit,
  'getPageCount' | 'loadData' | 'redoLayout' | 'renderToSVG' | 'setOptions' | 'renderToMIDI'
>;

// Verovio pageWidth is in units of 1/10 mm. Default A4 = 2100 (210 mm).
// To fill the container at scale=34 on a 96dpi screen we multiply pixels by:
//   25.4 / (96 * 0.1 * 0.34) ≈ 7.782
const VEROVIO_SCALE = 34;
const PIXELS_TO_VU = 25.4 / (96 * 0.1 * (VEROVIO_SCALE / 100));
const MIN_PAGE_WIDTH = 2100; // A4 minimum so narrow containers still look reasonable
let verovioToolkitPromise: Promise<VerovioToolkitLike> | null = null;

export const getScoreTitle = (scoreXml: string | null): string | null => {
  if (!scoreXml) {
    return null;
  }

  const match = scoreXml.match(/<part-name>([\s\S]*?)<\/part-name>/i);
  return match?.[1]?.trim() || null;
};

export const prepareSheetMusicXml = (scoreXml: string): string =>
  scoreXml
    .replace(/<part-name>[\s\S]*?<\/part-name>/gi, '<part-name></part-name>')
    .replace(
      /<part-abbreviation>[\s\S]*?<\/part-abbreviation>/gi,
      '<part-abbreviation></part-abbreviation>',
    );

const createToolkit = async (): Promise<VerovioToolkitLike> => {
  const verovioModule = await createVerovioModule();
  return new VerovioToolkit(verovioModule);
};

export const getVerovioToolkit = (): Promise<VerovioToolkitLike> => {
  if (!verovioToolkitPromise) {
    verovioToolkitPromise = createToolkit();
  }
  return verovioToolkitPromise;
};

export const resolveVerovioPageWidth = (containerWidth: number): number =>
  Math.max(Math.floor(containerWidth * PIXELS_TO_VU), MIN_PAGE_WIDTH);

export const renderVerovioScore = (
  toolkit: VerovioToolkitLike,
  scoreXml: string,
  pageWidth: number,
): string => {
  toolkit.setOptions({
    adjustPageHeight: true,
    footer: 'none',
    header: 'none',
    pageWidth,
    scale: 34,
  });
  toolkit.loadData(prepareSheetMusicXml(scoreXml));
  toolkit.redoLayout();

  const pageCount = toolkit.getPageCount();
  const pages: string[] = [];
  for (let pageNumber = 1; pageNumber <= pageCount; pageNumber += 1) {
    pages.push(toolkit.renderToSVG(pageNumber));
  }
  return pages.join('');
};

export const __testing__ = {
  resetVerovioToolkitCache() {
    verovioToolkitPromise = null;
  },
};

const SheetMusicViewer = ({
  scoreXml,
  highlightMeasureId,
  instrumentMode,
  viewTab,
  onViewTabChange,
  onPlayerReady,
  onPositionChanged,
  onApiReady,
}: SheetMusicViewerProps) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const playerRef = useRef<VerovioPlayer | null>(null);
  const onPlayerReadyRef = useRef(onPlayerReady);
  const onPositionChangedRef = useRef(onPositionChanged);
  useEffect(() => { onPlayerReadyRef.current = onPlayerReady; }, [onPlayerReady]);
  useEffect(() => { onPositionChangedRef.current = onPositionChanged; }, [onPositionChanged]);

  // Expose stable API to parent on mount
  useEffect(() => {
    if (!onApiReady) return;
    onApiReady({
      playPause: () => playerRef.current?.playPause(),
      stop: () => playerRef.current?.stop(),
    });
    return () => {
      playerRef.current?.dispose();
      playerRef.current = null;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }

    if (!scoreXml) {
      container.innerHTML = '';
      playerRef.current?.stop();
      return;
    }

    let cancelled = false;
    let animationFrameId: number | null = null;

    const renderScore = async () => {
      try {
        const toolkit = await getVerovioToolkit();
        if (cancelled || !containerRef.current) {
          return;
        }

        const svg = renderVerovioScore(
          toolkit,
          scoreXml,
          resolveVerovioPageWidth(containerRef.current.clientWidth),
        );

        if (!cancelled) {
          containerRef.current.innerHTML = svg;
        }

        // Load MIDI from the same toolkit state (loadData was called inside renderVerovioScore)
        const midiBase64 = toolkit.renderToMIDI();
        if (!cancelled && midiBase64) {
          if (!playerRef.current) {
            playerRef.current = new VerovioPlayer({
              onPositionChanged: (cur, tot) => onPositionChangedRef.current?.(cur, tot),
            });
          } else {
            playerRef.current.stop();
          }
          playerRef.current.load(midiBase64);
          onPlayerReadyRef.current?.();
        }
      } catch (err: unknown) {
        console.error('Verovio failed to render score:', err);
      }
    };

    void renderScore();

    const resizeObserver =
      typeof ResizeObserver === 'undefined'
        ? null
        : new ResizeObserver(() => {
            if (animationFrameId !== null) {
              cancelAnimationFrame(animationFrameId);
            }
            animationFrameId = requestAnimationFrame(() => {
              void renderScore();
            });
          });

    resizeObserver?.observe(container);

    return () => {
      cancelled = true;
      if (animationFrameId !== null) {
        cancelAnimationFrame(animationFrameId);
      }
      resizeObserver?.disconnect();
    };
  }, [scoreXml]);

  const label =
    instrumentMode === 'guitar'
      ? 'Guitar Sheet Music'
      : instrumentMode === 'piano'
        ? 'Piano Sheet Music'
        : 'Sheet Music';
  const scoreTitle = getScoreTitle(scoreXml) ?? label;

  const showSwitcher = instrumentMode === 'guitar';

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
      {scoreXml ? <div className="score-viewer__score-title">{scoreTitle}</div> : null}
      <div className="score-viewer__canvas score-viewer__canvas--sheet" ref={containerRef} />
    </div>
  );
};

export default SheetMusicViewer;
