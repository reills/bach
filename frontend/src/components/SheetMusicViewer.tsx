import { useEffect, useRef } from 'react';
import createVerovioModule from 'verovio/wasm';
import { VerovioToolkit } from 'verovio/esm';
import type { ScoreViewTab } from '../state/types';

interface SheetMusicViewerProps {
  scoreXml: string | null;
  highlightMeasureId: string | null;
  instrumentMode: 'guitar' | 'piano' | null;
  viewTab: ScoreViewTab;
  onViewTabChange?: (tab: ScoreViewTab) => void;
}

type VerovioToolkitLike = Pick<
  VerovioToolkit,
  'getPageCount' | 'loadData' | 'redoLayout' | 'renderToSVG' | 'setOptions'
>;

const MIN_PAGE_WIDTH = 320;
let verovioToolkitPromise: Promise<VerovioToolkitLike> | null = null;

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
  Math.max(Math.floor(containerWidth), MIN_PAGE_WIDTH);

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
    scale: 42,
  });
  toolkit.loadData(scoreXml);
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
}: SheetMusicViewerProps) => {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }

    if (!scoreXml) {
      container.innerHTML = '';
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
      <div className="score-viewer__canvas" ref={containerRef} />
    </div>
  );
};

export default SheetMusicViewer;
