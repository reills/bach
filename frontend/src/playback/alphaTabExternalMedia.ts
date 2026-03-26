export interface ExternalMidiPlaybackController {
  readonly totalDurationMs: number;
  play: () => Promise<void> | void;
  pause: () => void;
  seekTo: (timeMs: number) => void;
  setVolume: (volume: number) => void;
}

export interface AlphaTabExternalMediaHandler {
  readonly backingTrackDuration: number;
  playbackRate: number;
  masterVolume: number;
  seekTo: (time: number) => void;
  play: () => void;
  pause: () => void;
}

export const createAlphaTabExternalMediaHandler = (
  controller: ExternalMidiPlaybackController,
): AlphaTabExternalMediaHandler => {
  let masterVolume = 1;
  let playbackRate = 1;

  return {
    get backingTrackDuration() {
      return controller.totalDurationMs;
    },
    get playbackRate() {
      return playbackRate;
    },
    set playbackRate(value: number) {
      playbackRate = value;
    },
    get masterVolume() {
      return masterVolume;
    },
    set masterVolume(value: number) {
      masterVolume = value;
      controller.setVolume(value);
    },
    seekTo(time: number) {
      controller.seekTo(time);
    },
    play() {
      void controller.play();
    },
    pause() {
      controller.pause();
    },
  };
};
