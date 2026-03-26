import { useEffect, useRef } from 'react';
import type { ScoreViewTab } from '../state/types';
import { VerovioPlayer } from '../playback/VerovioPlayer';
import { getVerovioCursorIds, type VerovioElementsAtTime } from '../playback/verovioCursor';
import {
  getVerovioToolkit,
  resetVerovioToolkitCache,
  type VerovioToolkitLike,
} from '../playback/verovioToolkit';

export { getVerovioToolkit };

export interface SheetMusicPlayerAPI {
  play: () => void;
  pause: () => void;
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
  onApiReady?: (api: SheetMusicPlayerAPI | null) => void;
}

// Verovio pageWidth is in units of 1/10 mm. Default A4 = 2100 (210 mm).
// To fill the container at scale=34 on a 96dpi screen we multiply pixels by:
//   25.4 / (96 * 0.1 * 0.34) ≈ 7.782
const VEROVIO_SCALE = 34;
const PIXELS_TO_VU = 25.4 / (96 * 0.1 * (VEROVIO_SCALE / 100));
const MIN_PAGE_WIDTH = 2100; // A4 minimum so narrow containers still look reasonable
const MAX_PAGE_WIDTH = 2600;
const MIN_MEASURES_PER_SYSTEM = 4;
const DEFAULT_MEASURES_PER_SYSTEM = 5;
const MAX_MEASURES_PER_SYSTEM = 6;

export const getScoreTitle = (scoreXml: string | null): string | null => {
  if (!scoreXml) {
    return null;
  }

  const match = scoreXml.match(/<part-name>([\s\S]*?)<\/part-name>/i);
  return match?.[1]?.trim() || null;
};

const addSystemBreakToMeasure = (measureXml: string): string => {
  if (/<print\b[^>]*new-system\s*=\s*['"]yes['"][^>]*\/?>/i.test(measureXml)) {
    return measureXml;
  }

  if (/<print\b/i.test(measureXml)) {
    return measureXml.replace(/<print\b([^>]*?)(\/?)>/i, (_match, rawAttrs = '', selfClosing = '') => {
      const attrs = String(rawAttrs);
      const nextAttrs = /new-system\s*=/i.test(attrs)
        ? attrs.replace(/new-system\s*=\s*(['"])[^'"]*\1/i, 'new-system="yes"')
        : `${attrs.trimEnd()} new-system="yes"`;
      const normalizedAttrs = nextAttrs.trim().length > 0 ? ` ${nextAttrs.trim()}` : '';
      return `<print${normalizedAttrs}${selfClosing}>`;
    });
  }

  return measureXml.replace(
    /(<measure(?=[\s>])[^>]*>)(\s*(?:<attributes\b[\s\S]*?<\/attributes>\s*)?)/i,
    (_match, openTag: string, prefix: string) => `${openTag}${prefix}<print new-system="yes"/>`,
  );
};

export const insertSystemBreaks = (scoreXml: string, measuresPerSystem: number): string => {
  if (!Number.isFinite(measuresPerSystem) || measuresPerSystem < 2) {
    return scoreXml;
  }

  return scoreXml.replace(/<part(?=[\s>])[^>]*>[\s\S]*?<\/part>/gi, (partXml) => {
    let measureIndex = 0;

    return partXml.replace(/<measure(?=[\s>])[^>]*>[\s\S]*?<\/measure>/gi, (measureXml) => {
      measureIndex += 1;
      if (measureIndex === 1 || (measureIndex - 1) % measuresPerSystem !== 0) {
        return measureXml;
      }
      return addSystemBreakToMeasure(measureXml);
    });
  });
};

export const prepareSheetMusicXml = (
  scoreXml: string,
  measuresPerSystem = DEFAULT_MEASURES_PER_SYSTEM,
): string =>
  insertSystemBreaks(
    scoreXml
      .replace(/<part-name>[\s\S]*?<\/part-name>/gi, '<part-name></part-name>')
      .replace(
        /<part-abbreviation>[\s\S]*?<\/part-abbreviation>/gi,
        '<part-abbreviation></part-abbreviation>',
      ),
    measuresPerSystem,
  );

export const resolveVerovioPageWidth = (containerWidth: number): number =>
  Math.min(MAX_PAGE_WIDTH, Math.max(Math.floor(containerWidth * PIXELS_TO_VU), MIN_PAGE_WIDTH));

export const resolveMeasuresPerSystem = (containerWidth: number): number => {
  if (containerWidth < 720) {
    return MIN_MEASURES_PER_SYSTEM;
  }
  if (containerWidth < 1080) {
    return DEFAULT_MEASURES_PER_SYSTEM;
  }
  return MAX_MEASURES_PER_SYSTEM;
};

export const renderVerovioScore = (
  toolkit: VerovioToolkitLike,
  scoreXml: string,
  pageWidth: number,
  measuresPerSystem = DEFAULT_MEASURES_PER_SYSTEM,
): string => {
  toolkit.setOptions({
    adjustPageHeight: true,
    breaks: 'encoded',
    footer: 'none',
    header: 'none',
    pageWidth,
    scale: 34,
  });
  toolkit.loadData(prepareSheetMusicXml(scoreXml, measuresPerSystem));
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
    resetVerovioToolkitCache();
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
  const toolkitRef = useRef<VerovioToolkitLike | null>(null);
  const activeCursorIdsRef = useRef<string[]>([]);
  const onPlayerReadyRef = useRef(onPlayerReady);
  const onPositionChangedRef = useRef(onPositionChanged);
  useEffect(() => { onPlayerReadyRef.current = onPlayerReady; }, [onPlayerReady]);
  useEffect(() => { onPositionChangedRef.current = onPositionChanged; }, [onPositionChanged]);

  const clearPlaybackCursor = () => {
    const container = containerRef.current;
    if (!container) {
      activeCursorIdsRef.current = [];
      return;
    }

    for (const id of activeCursorIdsRef.current) {
      container
        .querySelectorAll<SVGElement>(`#${CSS.escape(id)}`)
        .forEach((element) => element.classList.remove('sheet-music-viewer__playback-cursor'));
    }
    activeCursorIdsRef.current = [];
  };

  const updatePlaybackCursor = (elementsAtTime: VerovioElementsAtTime | null | undefined) => {
    const container = containerRef.current;
    if (!container) {
      activeCursorIdsRef.current = [];
      return;
    }

    const nextCursorIds = getVerovioCursorIds(elementsAtTime);
    const previousIds = activeCursorIdsRef.current;

    for (const id of previousIds) {
      if (nextCursorIds.includes(id)) {
        continue;
      }
      container
        .querySelectorAll<SVGElement>(`#${CSS.escape(id)}`)
        .forEach((element) => element.classList.remove('sheet-music-viewer__playback-cursor'));
    }

    for (const id of nextCursorIds) {
      if (previousIds.includes(id)) {
        continue;
      }
      container
        .querySelectorAll<SVGElement>(`#${CSS.escape(id)}`)
        .forEach((element) => element.classList.add('sheet-music-viewer__playback-cursor'));
    }

    activeCursorIdsRef.current = nextCursorIds;
  };

  // Expose stable API to parent on mount
  useEffect(() => {
    onApiReady?.({
      play: () => void playerRef.current?.play(),
      pause: () => playerRef.current?.pause(),
      playPause: () => playerRef.current?.playPause(),
      stop: () => playerRef.current?.stop(),
    });
    return () => {
      onApiReady?.(null);
      clearPlaybackCursor();
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
      clearPlaybackCursor();
      toolkitRef.current = null;
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
        toolkitRef.current = toolkit;

        const svg = renderVerovioScore(
          toolkit,
          scoreXml,
          resolveVerovioPageWidth(containerRef.current.clientWidth),
          resolveMeasuresPerSystem(containerRef.current.clientWidth),
        );

        if (!cancelled) {
          containerRef.current.innerHTML = svg;
        }

        // Load MIDI from the same toolkit state (loadData was called inside renderVerovioScore)
        const midiBase64 = toolkit.renderToMIDI();
        if (!cancelled && midiBase64) {
          if (!playerRef.current) {
            playerRef.current = new VerovioPlayer({
              onPositionChanged: (cur, tot) => {
                onPositionChangedRef.current?.(cur, tot);
                updatePlaybackCursor(toolkitRef.current?.getElementsAtTime(cur));
              },
              onStopped: clearPlaybackCursor,
            });
          } else {
            playerRef.current.stop();
            clearPlaybackCursor();
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
      clearPlaybackCursor();
      toolkitRef.current = null;
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
