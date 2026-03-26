import { describe, expect, it, vi } from 'vitest';
import { VerovioPlayer } from './VerovioPlayer';

describe('VerovioPlayer', () => {
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
