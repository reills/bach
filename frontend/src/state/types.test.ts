import { describe, it, expect } from 'vitest';
import { toHitKey, getMeasureId, getEventId } from './types';
import type { HitKey, MeasureMap, EventHitMap, ScoreState } from './types';

describe('toHitKey', () => {
  it('produces a canonical pipe-delimited key with all fields', () => {
    const hit: HitKey = { barIndex: 2, voiceIndex: 1, beatIndex: 0, noteIndex: 3 };
    expect(toHitKey(hit)).toBe('2|1|0|3');
  });

  it('fills missing optional fields with -1', () => {
    const hit: HitKey = { barIndex: 0 };
    expect(toHitKey(hit)).toBe('0|-1|-1|-1');
  });

  it('fills partial optional fields with -1', () => {
    const hit: HitKey = { barIndex: 5, voiceIndex: 2 };
    expect(toHitKey(hit)).toBe('5|2|-1|-1');
  });
});

describe('getMeasureId', () => {
  it('returns the measure id for a known bar index', () => {
    const measureMap: MeasureMap = { '0': 'measure-a', '1': 'measure-b' };
    expect(getMeasureId(measureMap, 1)).toBe('measure-b');
  });

  it('returns null for an unknown bar index', () => {
    const measureMap: MeasureMap = { '0': 'measure-a' };
    expect(getMeasureId(measureMap, 5)).toBeNull();
  });

  it('returns null when measureMap is null', () => {
    expect(getMeasureId(null, 0)).toBeNull();
  });

  it('returns null when measureMap is undefined', () => {
    expect(getMeasureId(undefined, 0)).toBeNull();
  });
});

describe('getEventId', () => {
  it('returns the event id for a matching hit key', () => {
    const hitMap: EventHitMap = { '1|0|-1|-1': 'evt-42' };
    const hit: HitKey = { barIndex: 1, voiceIndex: 0 };
    expect(getEventId(hitMap, hit)).toBe('evt-42');
  });

  it('returns null when no key matches', () => {
    const hitMap: EventHitMap = {};
    const hit: HitKey = { barIndex: 0 };
    expect(getEventId(hitMap, hit)).toBeNull();
  });

  it('returns null when eventHitMap is null', () => {
    const hit: HitKey = { barIndex: 0 };
    expect(getEventId(null, hit)).toBeNull();
  });

  it('returns null when eventHitMap is undefined', () => {
    const hit: HitKey = { barIndex: 0 };
    expect(getEventId(undefined, hit)).toBeNull();
  });

  it('uses full hit key with all fields for lookup', () => {
    const hitMap: EventHitMap = { '3|2|1|0': 'evt-99' };
    const hit: HitKey = { barIndex: 3, voiceIndex: 2, beatIndex: 1, noteIndex: 0 };
    expect(getEventId(hitMap, hit)).toBe('evt-99');
  });
});

describe('ScoreState changedMeasureIds', () => {
  const base: ScoreState = {
    scoreId: null,
    revision: null,
    scoreXml: null,
    measureMap: null,
    eventHitMap: null,
    draftId: null,
    draftXml: null,
    draftBaseRevision: null,
    highlightMeasureId: null,
    selectedMeasureId: null,
    selectedBarIndex: null,
    lockedEventIds: null,
    changedMeasureIds: null,
    lastEventId: null,
    midi: null,
  };

  it('defaults to null', () => {
    expect(base.changedMeasureIds).toBeNull();
  });

  it('accepts an array of measure id strings', () => {
    const s: ScoreState = { ...base, changedMeasureIds: ['m1', 'm2'] };
    expect(s.changedMeasureIds).toEqual(['m1', 'm2']);
  });

  it('accepts a single-element array', () => {
    const s: ScoreState = { ...base, changedMeasureIds: ['measure-7'] };
    expect(s.changedMeasureIds).toHaveLength(1);
    expect(s.changedMeasureIds![0]).toBe('measure-7');
  });
});
