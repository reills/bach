import { describe, expect, it, vi } from 'vitest';
import { createAlphaTabExternalMediaHandler } from './alphaTabExternalMedia';

describe('createAlphaTabExternalMediaHandler', () => {
  it('delegates transport controls and volume to the MIDI controller', async () => {
    const controller = {
      totalDurationMs: 3210,
      pause: vi.fn(),
      play: vi.fn(async () => undefined),
      seekTo: vi.fn(),
      setVolume: vi.fn(),
    };

    const handler = createAlphaTabExternalMediaHandler(controller);

    expect(handler.backingTrackDuration).toBe(3210);

    handler.masterVolume = 0.42;
    handler.playbackRate = 0.85;
    handler.seekTo(900);
    handler.play();
    handler.pause();

    expect(handler.masterVolume).toBe(0.42);
    expect(handler.playbackRate).toBe(0.85);
    expect(controller.setVolume).toHaveBeenCalledWith(0.42);
    expect(controller.seekTo).toHaveBeenCalledWith(900);
    expect(controller.play).toHaveBeenCalledTimes(1);
    expect(controller.pause).toHaveBeenCalledTimes(1);
  });
});
