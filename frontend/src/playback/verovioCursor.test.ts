import { describe, expect, it } from 'vitest';
import { getVerovioCursorIds } from './verovioCursor';

describe('getVerovioCursorIds', () => {
  it('collects unique note, chord, and rest ids in playback order', () => {
    expect(
      getVerovioCursorIds({
        chords: ['chord-1', 'shared'],
        notes: ['note-1', 'shared'],
        rests: ['rest-1'],
      }),
    ).toEqual(['note-1', 'shared', 'chord-1', 'rest-1']);
  });

  it('returns an empty list when no timed elements are available', () => {
    expect(getVerovioCursorIds(undefined)).toEqual([]);
    expect(getVerovioCursorIds({})).toEqual([]);
  });
});
