import type { InstrumentMode, ScoreViewTab } from '../state/types';

export type ScoreViewerInstrumentMode = InstrumentMode | null;
export type ScoreViewerTab = ScoreViewTab;
export interface ScoreViewerStaffVisibility {
  showStandardNotation: boolean;
  showTablature: boolean;
}

export const normalizeViewTab = (
  instrumentMode: ScoreViewerInstrumentMode,
  viewTab: ScoreViewerTab,
): ScoreViewerTab => (instrumentMode === 'guitar' ? viewTab : 'score');

export const shouldShowTabSwitcher = (
  instrumentMode: ScoreViewerInstrumentMode,
): instrumentMode is 'guitar' => instrumentMode === 'guitar';

export const getScoreViewerLabel = (
  instrumentMode: ScoreViewerInstrumentMode,
  viewTab: ScoreViewerTab,
): string => {
  const activeViewTab = normalizeViewTab(instrumentMode, viewTab);
  if (instrumentMode === 'guitar') {
    return activeViewTab === 'tab' ? 'Guitar Tab' : 'Guitar Sheet Music';
  }
  if (instrumentMode === 'piano') {
    return 'Piano Sheet Music';
  }
  return 'Sheet Music';
};

export const getStaffVisibility = (
  instrumentMode: ScoreViewerInstrumentMode,
  viewTab: ScoreViewerTab,
): ScoreViewerStaffVisibility => {
  const activeViewTab = normalizeViewTab(instrumentMode, viewTab);
  if (instrumentMode === 'guitar') {
    return activeViewTab === 'tab'
      ? {
          showStandardNotation: false,
          showTablature: true,
        }
      : {
          showStandardNotation: true,
          showTablature: false,
        };
  }

  return {
    showStandardNotation: true,
    showTablature: false,
  };
};

export const resolveStaveProfile = (
  alphaTabModule: any,
  _instrumentMode: ScoreViewerInstrumentMode,
  _viewTab: ScoreViewerTab,
) =>
  alphaTabModule.StaveProfile?.Default ??
  alphaTabModule.StaveProfile?.ScoreTab ??
  alphaTabModule.StaveProfile?.Score ??
  alphaTabModule.LayoutStaveProfile?.Score ??
  0;
