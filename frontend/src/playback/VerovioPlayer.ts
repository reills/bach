import { Midi } from '@tonejs/midi';

// ---------------------------------------------------------------------------
// WebAudioFont type declarations
// The player script is served from /webaudiofont/WebAudioFontPlayer.js and
// attaches WebAudioFontPlayer to the global scope as a plain var.
// ---------------------------------------------------------------------------

interface WafLoader {
  startLoad(ctx: AudioContext, url: string, varName: string): void;
  waitLoad(cb: () => void): void;
  findInstrument(program: number): number;
  instrumentInfo(index: number): WafPresetInfo;
}

interface WafEnvelope {
  cancel(): void;
}

interface WafPlayerInstance {
  loader: WafLoader;
  queueWaveTable(
    ctx: AudioContext,
    target: AudioNode,
    preset: object,
    when: number,
    pitch: number,
    duration: number,
    volume: number,
  ): WafEnvelope;
  cancelQueue(ctx: AudioContext): void;
}

declare global {
  // eslint-disable-next-line no-var
  var WebAudioFontPlayer: new () => WafPlayerInstance;
}

// ---------------------------------------------------------------------------
// Preset configuration.
// ---------------------------------------------------------------------------

export type VerovioPlaybackInstrument = 'guitar' | 'piano';

interface WafPresetInfo {
  variable: string;
  url: string;
  title: string;
  pitch: number;
}

const PLAYBACK_PROGRAMS: Record<VerovioPlaybackInstrument, number> = {
  piano: 0,
  guitar: 24,
};

export const resolvePlaybackProgram = (instrument: VerovioPlaybackInstrument): number =>
  PLAYBACK_PROGRAMS[instrument];

// ---------------------------------------------------------------------------
// Script loading — deduplicated by URL
// ---------------------------------------------------------------------------

const loadedScriptUrls = new Set<string>();

function loadScript(src: string): Promise<void> {
  if (loadedScriptUrls.has(src)) return Promise.resolve();
  loadedScriptUrls.add(src);
  return new Promise<void>((resolve, reject) => {
    const el = document.createElement('script');
    el.type = 'text/javascript';
    el.src = src;
    el.onload = () => resolve();
    el.onerror = () => {
      loadedScriptUrls.delete(src);
      reject(new Error(`VerovioPlayer: failed to load script ${src}`));
    };
    document.head.appendChild(el);
  });
}

// ---------------------------------------------------------------------------
// Shared AudioContext + WebAudioFont player
// Both are singletons so we don't re-decode the preset on every score load.
// ---------------------------------------------------------------------------

let sharedCtx: AudioContext | null = null;
let sharedGain: GainNode | null = null;
let sharedWafPlayer: WafPlayerInstance | null = null;
const presetLoadPromises = new Map<VerovioPlaybackInstrument, Promise<void>>();

function getOrCreateAudio(): { ctx: AudioContext; gain: GainNode } {
  if (!sharedCtx) {
    const Ctor =
      window.AudioContext ??
      (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    sharedCtx = new Ctor!();
    sharedGain = sharedCtx.createGain();
    sharedGain.gain.value = 0.7;
    sharedGain.connect(sharedCtx.destination);
  }
  return { ctx: sharedCtx, gain: sharedGain! };
}

function getPresetInfo(instrument: VerovioPlaybackInstrument): WafPresetInfo {
  if (!sharedWafPlayer) {
    sharedWafPlayer = new WebAudioFontPlayer();
  }
  const program = resolvePlaybackProgram(instrument);
  return sharedWafPlayer.loader.instrumentInfo(sharedWafPlayer.loader.findInstrument(program));
}

/** Loads the WebAudioFont player script + decodes the selected preset once. */
function ensurePresetReady(instrument: VerovioPlaybackInstrument): Promise<void> {
  const existing = presetLoadPromises.get(instrument);
  if (existing) return existing;

  const loadPromise = (async () => {
    // 1. Load the player library from our own /public dir
    await loadScript('/webaudiofont/WebAudioFontPlayer.js');

    // 2. Create shared WAF player instance
    const { ctx } = getOrCreateAudio();
    if (!sharedWafPlayer) {
      sharedWafPlayer = new WebAudioFontPlayer();
    }

    // 3. Kick off download + decode of the selected General MIDI preset
    const presetInfo = getPresetInfo(instrument);
    sharedWafPlayer.loader.startLoad(ctx, presetInfo.url, presetInfo.variable);

    // 4. Wait until all zones are decoded (buffers populated)
    await new Promise<void>((resolve) => {
      sharedWafPlayer!.loader.waitLoad(resolve);
    });
  })();

  presetLoadPromises.set(instrument, loadPromise);
  return loadPromise;
}

// ---------------------------------------------------------------------------
// VerovioPlayer — public API mirrors the previous oscillator implementation
// ---------------------------------------------------------------------------

type PositionCallback = (currentMs: number, totalMs: number) => void;

interface VerovioPlayerOptions {
  playbackInstrument?: VerovioPlaybackInstrument;
  onPositionChanged?: PositionCallback;
  onStopped?: () => void;
}

type PlayerState = 'idle' | 'playing' | 'paused';

/**
 * Sample-based audio player for Verovio-rendered scores.
 *
 * Parses the MIDI produced by toolkit.renderToMIDI() and plays it back using
 * WebAudioFont instrument presets via the Web Audio API.
 *
 * Public API: play(), pause(), stop(), playPause(), dispose(),
 *             onPositionChanged(currentMs, totalMs)
 */
export class VerovioPlayer {
  private midi: Midi | null = null;
  private totalMs = 0;
  private tickerId: ReturnType<typeof setInterval> | null = null;
  /** AudioContext.currentTime that maps to piece t=0, accounting for seek. */
  private originTime = 0;
  private pauseOffsetMs = 0;
  private state: PlayerState = 'idle';
  private playbackInstrument: VerovioPlaybackInstrument;
  private readonly onPositionChanged?: PositionCallback;
  private readonly onStopped?: () => void;

  constructor(opts: VerovioPlayerOptions = {}) {
    this.playbackInstrument = opts.playbackInstrument ?? 'piano';
    this.onPositionChanged = opts.onPositionChanged;
    this.onStopped = opts.onStopped;
    // Start loading the preset eagerly so it's decoded by the time the user
    // clicks Play. Errors are swallowed here — play() will surface them.
    void ensurePresetReady(this.playbackInstrument).catch(() => undefined);
  }

  setPlaybackInstrument(instrument: VerovioPlaybackInstrument): void {
    if (this.playbackInstrument === instrument) {
      return;
    }
    this.stop();
    this.playbackInstrument = instrument;
    void ensurePresetReady(this.playbackInstrument).catch(() => undefined);
  }

  /** Load a base64-encoded MIDI string returned by toolkit.renderToMIDI(). */
  load(midiBase64: string): void {
    this.stop();
    if (!midiBase64) return;

    const binary = atob(midiBase64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    this.midi = new Midi(bytes.buffer);
    this.totalMs = this.midi.duration * 1000;
    this.pauseOffsetMs = 0;
  }

  async play(): Promise<void> {
    if (!this.midi || this.state === 'playing') return;

    try {
      await ensurePresetReady(this.playbackInstrument);
    } catch (err) {
      console.error('VerovioPlayer: could not load instrument preset:', err);
      return;
    }

    const { ctx, gain } = getOrCreateAudio();
    if (ctx.state === 'suspended') {
      await ctx.resume();
    }

    const presetInfo = getPresetInfo(this.playbackInstrument);
    const preset = (window as unknown as Record<string, object>)[presetInfo.variable];
    if (!preset) {
      console.error('VerovioPlayer: preset not available after loading');
      return;
    }

    const startOffsetSec = this.pauseOffsetMs / 1000;
    const nowSec = ctx.currentTime;
    this.originTime = nowSec - startOffsetSec;
    this.state = 'playing';

    for (const track of this.midi.tracks) {
      for (const note of track.notes) {
        // Skip notes whose attack+release both end before our seek point
        if (note.time + note.duration < startOffsetSec - 0.01) continue;

        const when = this.originTime + note.time;
        // Don't schedule notes already more than 50 ms in the past
        if (when < nowSec - 0.05) continue;

        sharedWafPlayer!.queueWaveTable(
          ctx,
          gain,
          preset,
          when,
          note.midi,
          note.duration,
          note.velocity,
        );
      }
    }

    this.startTicker(ctx);
  }

  pause(): void {
    if (this.state !== 'playing') return;
    const ctx = sharedCtx;
    if (ctx) {
      this.pauseOffsetMs = (ctx.currentTime - this.originTime) * 1000;
      sharedWafPlayer?.cancelQueue(ctx);
    }
    this.state = 'paused';
    this.stopTicker();
    this.onPositionChanged?.(this.pauseOffsetMs, this.totalMs);
  }

  stop(): void {
    if (this.state === 'playing' && sharedCtx) {
      sharedWafPlayer?.cancelQueue(sharedCtx);
    }
    this.state = 'idle';
    this.pauseOffsetMs = 0;
    this.stopTicker();
    this.onStopped?.();
    this.onPositionChanged?.(0, this.totalMs);
  }

  playPause(): void {
    if (this.state === 'playing') {
      this.pause();
    } else {
      void this.play();
    }
  }

  get isLoaded(): boolean {
    return this.midi !== null;
  }

  get totalDurationMs(): number {
    return this.totalMs;
  }

  setVolume(volume: number): void {
    const { gain } = getOrCreateAudio();
    gain.gain.value = Math.max(0, volume);
  }

  seekTo(timeMs: number): void {
    const nextOffsetMs = Math.max(0, Math.min(timeMs, this.totalMs));
    const wasPlaying = this.state === 'playing';

    if (wasPlaying) {
      if (sharedCtx) {
        sharedWafPlayer?.cancelQueue(sharedCtx);
      }
      this.stopTicker();
      this.state = 'paused';
    }

    this.pauseOffsetMs = nextOffsetMs;
    this.onPositionChanged?.(this.pauseOffsetMs, this.totalMs);

    if (wasPlaying && this.midi) {
      void this.play();
    }
  }

  /** Release references. Does not close the shared AudioContext. */
  dispose(): void {
    this.stop();
    this.midi = null;
  }

  private startTicker(ctx: AudioContext): void {
    this.tickerId = setInterval(() => {
      if (this.state !== 'playing') return;
      const elapsed = (ctx.currentTime - this.originTime) * 1000;
      this.onPositionChanged?.(elapsed, this.totalMs);
      if (elapsed >= this.totalMs) {
        this.stop();
      }
    }, 100);
  }

  private stopTicker(): void {
    if (this.tickerId !== null) {
      clearInterval(this.tickerId);
      this.tickerId = null;
    }
  }
}
