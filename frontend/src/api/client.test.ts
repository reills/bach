import { afterEach, describe, expect, it, vi } from 'vitest';
import { compose } from './client';

describe('compose client', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('includes render_mode in compose requests', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        scoreId: 'score-1',
        revision: 0,
        document: {
          instrumentMode: 'guitar',
          views: {
            score: {
              xml: '<score-partwise version="3.1"/>',
            },
          },
        },
        instrumentMode: 'guitar',
      }),
    });
    vi.stubGlobal('fetch', fetchMock);

    await compose({ prompt: 'test', render_mode: 'piano' });

    expect(fetchMock).toHaveBeenCalledWith(
      '/compose',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ prompt: 'test', render_mode: 'piano' }),
      }),
    );
  });
});
