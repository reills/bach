import { describe, expect, it, vi } from 'vitest';
import { VerovioPlayer, resolvePlaybackProgram } from './VerovioPlayer';

describe('VerovioPlayer', () => {
  it('maps guitar playback to the General MIDI nylon guitar program', () => {
    expect(resolvePlaybackProgram('piano')).toBe(0);
    expect(resolvePlaybackProgram('guitar')).toBe(24);
  });

  it('keeps the requested playback instrument until explicitly changed', () => {
    const player = new VerovioPlayer({ playbackInstrument: 'guitar' });

    expect((player as any).playbackInstrument).toBe('guitar');
    player.setPlaybackInstrument('piano');
    expect((player as any).playbackInstrument).toBe('piano');
  });

  it('does not resume playback when seeking while already paused', () => {
    const player = new VerovioPlayer();
    const playSpy = vi.spyOn(player, 'play').mockResolvedValue();

    (player as any).midi = {};
    (player as any).totalMs = 5000;
    (player as any).state = 'paused';

    player.seekTo(1200);

    expect(playSpy).not.toHaveBeenCalled();
    expect((player as any).pauseOffsetMs).toBe(1200);
  });

  it('restarts playback after a seek when already playing', () => {
    const player = new VerovioPlayer();
    const playSpy = vi.spyOn(player, 'play').mockResolvedValue();

    (player as any).midi = {};
    (player as any).totalMs = 5000;
    (player as any).state = 'playing';

    player.seekTo(2400);

    expect(playSpy).toHaveBeenCalledTimes(1);
    expect((player as any).pauseOffsetMs).toBe(2400);
  });
});
