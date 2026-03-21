export type ScoreViewerInstrumentMode = 'guitar' | 'piano' | null;
export type ScoreViewerTab = 'score' | 'tab';

export const resolveStaveProfile = (
  alphaTabModule: any,
  instrumentMode: ScoreViewerInstrumentMode,
  viewTab: ScoreViewerTab,
) =>
  instrumentMode === 'guitar' && viewTab === 'tab'
    ? alphaTabModule.StaveProfile?.Tab ?? alphaTabModule.LayoutStaveProfile?.Tab ?? 2
    : alphaTabModule.StaveProfile?.Score ?? alphaTabModule.LayoutStaveProfile?.Score ?? 1;
