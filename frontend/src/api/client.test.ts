import { afterEach, describe, expect, it, vi } from 'vitest';
import { compose, generateMeasures } from './client';

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

  it('posts generated measure operations to the backend route', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        document: {
          instrumentMode: 'piano',
          views: {
            score: {
              xml: '<score-partwise version="3.1"/>',
            },
          },
        },
        revision: 2,
        insertedMeasureIds: ['m2'],
        replacedMeasureIds: [],
        changedMeasureIds: ['m2'],
      }),
    });
    vi.stubGlobal('fetch', fetchMock);

    await generateMeasures({
      scoreId: 'score-1',
      revision: 1,
      operation: 'append',
      count: 1,
    });

    expect(fetchMock).toHaveBeenCalledWith(
      '/generate_measures',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          scoreId: 'score-1',
          revision: 1,
          operation: 'append',
          count: 1,
        }),
      }),
    );
  });
});
