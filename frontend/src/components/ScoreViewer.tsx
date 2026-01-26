import { useEffect, useRef } from 'react';
import * as alphaTab from '@coderline/alphatab';
import type { HitKey } from '../state/types';

interface ScoreViewerProps {
  scoreXml: string | null;
  highlightMeasureId: string | null;
  onMeasureClick?: (barIndex: number) => void;
  onNoteClick?: (hit: HitKey) => void;
  onApiReady?: (api: unknown) => void;
}

const loadScoreXml = (api: any, scoreXml: string) => {
  if (!scoreXml || typeof scoreXml !== 'string' || scoreXml.trim().length === 0) {
    console.warn('loadScoreXml: invalid or empty scoreXml');
    return;
  }
  try {
    // Convert string to Uint8Array for AlphaTab
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
  onMeasureClick,
  onNoteClick,
  onApiReady,
}: ScoreViewerProps) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const apiRef = useRef<any>(null);
  const onMeasureClickRef = useRef<ScoreViewerProps['onMeasureClick']>();
  const onNoteClickRef = useRef<ScoreViewerProps['onNoteClick']>();
  const onApiReadyRef = useRef<ScoreViewerProps['onApiReady']>();

  useEffect(() => {
    onMeasureClickRef.current = onMeasureClick;
  }, [onMeasureClick]);

  useEffect(() => {
    onNoteClickRef.current = onNoteClick;
  }, [onNoteClick]);

  useEffect(() => {
    onApiReadyRef.current = onApiReady;
  }, [onApiReady]);

  useEffect(() => {
    if (!containerRef.current || apiRef.current) {
      return;
    }

    const settings = {
      layout: {
        staveProfile:
          (alphaTab as any).StaveProfile?.Score ??
          (alphaTab as any).LayoutStaveProfile?.Score ??
          0,
      },
      core: {
        includeNoteBounds: true,
        fontDirectory: '/font/',
      },
      player: {
        enablePlayer: true,
      },
      display: {
        resources: {
          copyrightFont: '11px Arial',
        },
      },
    };

    const api = new (alphaTab as any).AlphaTabApi(containerRef.current, settings);
    apiRef.current = api;
    onApiReadyRef.current?.(api);

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
      apiRef.current?.destroy?.();
      apiRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!apiRef.current || !scoreXml) {
      return;
    }
    loadScoreXml(apiRef.current, scoreXml);
  }, [scoreXml]);

  return (
    <div className="score-viewer">
      <div className="score-viewer__header">
        <span className="score-viewer__label">Score + Tab</span>
        {highlightMeasureId ? (
          <span className="score-viewer__highlight">
            Draft focus: {highlightMeasureId}
          </span>
        ) : null}
      </div>
      {!scoreXml ? (
        <div className="score-viewer__empty">
          Load a score to start rendering.
        </div>
      ) : null}
      <div className="score-viewer__canvas" ref={containerRef} />
    </div>
  );
};

export default ScoreViewer;
