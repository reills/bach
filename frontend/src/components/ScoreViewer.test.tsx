import { describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import ScoreViewer, { createScoreViewerPlaybackController } from './ScoreViewer';

const renderTab = (viewTab: 'score' | 'tab' = 'tab') =>
  renderToStaticMarkup(
    <ScoreViewer
      scoreXml="<score-partwise version='3.1'/>"
      highlightMeasureId={null}
      instrumentMode="guitar"
      viewTab={viewTab}
      onViewTabChange={() => {}}
    />,
  );

describe('ScoreViewer (guitar tab renderer)', () => {
  it('renders the Guitar Tab label', () => {
    expect(renderTab('tab')).toContain('Guitar Tab');
  });

  it('always shows the Sheet Music / Guitar Tab switcher', () => {
    const markup = renderTab('tab');
    expect(markup).toContain('Sheet Music');
    expect(markup).toContain('Guitar Tab');
  });

  it('marks Guitar Tab as active when viewTab is tab', () => {
    const markup = renderTab('tab');
    // The Guitar Tab button should have aria-selected="true"
    expect(markup).toMatch(/Guitar Tab[\s\S]*?aria-selected="true"|aria-selected="true"[\s\S]*?Guitar Tab/);
  });

  it('marks Sheet Music as active when viewTab is score', () => {
    const markup = renderTab('score');
    // The Sheet Music button should have aria-selected="true"
    expect(markup).toMatch(/Sheet Music[\s\S]*?aria-selected="true"|aria-selected="true"[\s\S]*?Sheet Music/);
  });

  it('pauses and stops the shared MIDI player directly while keeping AlphaTab in sync', () => {
    const alphaTabApiRef = {
      current: {
        pause: vi.fn(),
        play: vi.fn(async () => undefined),
        playPause: vi.fn(),
        stop: vi.fn(),
      },
    };
    const midiPlayerRef = {
      current: {
        pause: vi.fn(),
        play: vi.fn(async () => undefined),
        stop: vi.fn(),
      },
    };
    const updatePosition = vi.fn();

    const controller = createScoreViewerPlaybackController({
      alphaTabApiRef,
      midiPlayerRef: midiPlayerRef as any,
      updatePosition,
    });

    controller.pause();
    controller.stop();

    expect(midiPlayerRef.current.pause).toHaveBeenCalledTimes(1);
    expect(alphaTabApiRef.current.pause).toHaveBeenCalledTimes(1);
    expect(midiPlayerRef.current.stop).toHaveBeenCalledTimes(1);
    expect(alphaTabApiRef.current.stop).toHaveBeenCalledTimes(1);
    expect(updatePosition).toHaveBeenCalledWith(0);
  });

  it('falls back to the shared MIDI player when AlphaTab transport is unavailable', async () => {
    const midiPlayerRef = {
      current: {
        pause: vi.fn(),
        play: vi.fn(async () => undefined),
        stop: vi.fn(),
      },
    };

    const controller = createScoreViewerPlaybackController({
      alphaTabApiRef: { current: null } as any,
      midiPlayerRef: midiPlayerRef as any,
      updatePosition: vi.fn(),
    });

    controller.play();

    expect(midiPlayerRef.current.play).toHaveBeenCalledTimes(1);
  });
});
