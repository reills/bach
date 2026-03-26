import { beforeEach, describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';

const {
  createVerovioModuleMock,
  toolkitMock,
  VerovioToolkitMock,
} = vi.hoisted(() => {
  const toolkit = {
    getPageCount: vi.fn(() => 2),
    loadData: vi.fn(() => true),
    redoLayout: vi.fn(),
    renderToMIDI: vi.fn(() => ''),
    renderToSVG: vi.fn((pageNumber: number) => `<svg data-page="${pageNumber}"></svg>`),
    setOptions: vi.fn(),
  };

  return {
    createVerovioModuleMock: vi.fn(async () => ({ module: 'verovio' })),
    toolkitMock: toolkit,
    VerovioToolkitMock: vi.fn(function VerovioToolkitMockImpl() {
      return toolkit;
    }),
  };
});

vi.mock('verovio/wasm', () => ({
  default: createVerovioModuleMock,
}));

vi.mock('verovio/esm', () => ({
  VerovioToolkit: VerovioToolkitMock,
}));

import SheetMusicViewer, {
  __testing__,
  getScoreTitle,
  getVerovioToolkit,
  insertSystemBreaks,
  prepareSheetMusicXml,
  renderVerovioScore,
  resolveMeasuresPerSystem,
  resolveVerovioPageWidth,
} from './SheetMusicViewer';

const renderViewer = (
  instrumentMode: 'guitar' | 'piano' | null,
  viewTab: 'score' | 'tab' = 'score',
) =>
  renderToStaticMarkup(
    <SheetMusicViewer
      scoreXml="<score-partwise version='3.1'/>"
      highlightMeasureId={null}
      instrumentMode={instrumentMode}
      viewTab={viewTab}
      onViewTabChange={() => {}}
    />,
  );

beforeEach(() => {
  __testing__.resetVerovioToolkitCache();
  createVerovioModuleMock.mockClear();
  VerovioToolkitMock.mockClear();
  toolkitMock.getPageCount.mockReset();
  toolkitMock.getPageCount.mockReturnValue(2);
  toolkitMock.loadData.mockClear();
  toolkitMock.redoLayout.mockClear();
  toolkitMock.renderToMIDI.mockClear();
  toolkitMock.renderToSVG.mockReset();
  toolkitMock.renderToSVG.mockImplementation(
    (pageNumber: number) => `<svg data-page="${pageNumber}"></svg>`,
  );
  toolkitMock.setOptions.mockClear();
});

describe('SheetMusicViewer', () => {
  it('renders Guitar Sheet Music label for guitar in score view', () => {
    expect(renderViewer('guitar', 'score')).toContain('Guitar Sheet Music');
  });

  it('renders Piano Sheet Music label for piano', () => {
    expect(renderViewer('piano')).toContain('Piano Sheet Music');
  });

  it('renders generic Sheet Music label when instrumentMode is null', () => {
    expect(renderViewer(null)).toContain('Sheet Music');
  });

  it('shows the Sheet Music / Guitar Tab switcher for guitar mode', () => {
    const markup = renderViewer('guitar', 'score');
    expect(markup).toContain('Sheet Music');
    expect(markup).toContain('Guitar Tab');
  });

  it('hides the tab switcher for piano mode', () => {
    const markup = renderViewer('piano');
    expect(markup).not.toContain('Guitar Tab');
  });

  it('marks Sheet Music as active when viewTab is score', () => {
    const markup = renderViewer('guitar', 'score');
    expect(markup).toContain('aria-selected="true"');
  });

  it('marks Guitar Tab as active when viewTab is tab', () => {
    const markup = renderViewer('guitar', 'tab');
    expect(markup).toMatch(/Guitar Tab[\s\S]*?aria-selected="true"|aria-selected="true"[\s\S]*?Guitar Tab/);
  });

  it('renders the empty state when scoreXml is null', () => {
    const markup = renderToStaticMarkup(
      <SheetMusicViewer
        scoreXml={null}
        highlightMeasureId={null}
        instrumentMode="piano"
        viewTab="score"
      />,
    );
    expect(markup).toContain('No score loaded');
  });

  it('shows the draft badge when highlightMeasureId is set', () => {
    const markup = renderToStaticMarkup(
      <SheetMusicViewer
        scoreXml="<score-partwise/>"
        highlightMeasureId="measure-3"
        instrumentMode="piano"
        viewTab="score"
      />,
    );
    expect(markup).toContain('measure-3');
  });

  it('caches the Verovio toolkit initialization', async () => {
    const first = await getVerovioToolkit();
    const second = await getVerovioToolkit();

    expect(createVerovioModuleMock).toHaveBeenCalledTimes(1);
    expect(VerovioToolkitMock).toHaveBeenCalledTimes(1);
    expect(second).toBe(first);
  });

  it('renders all Verovio pages into one SVG payload', async () => {
    const toolkit = await getVerovioToolkit();

    const svg = renderVerovioScore(toolkit, '<score-partwise/>', 840);

    expect(toolkitMock.setOptions).toHaveBeenCalledWith(
      expect.objectContaining({
        adjustPageHeight: true,
        breaks: 'encoded',
        footer: 'none',
        header: 'none',
        pageWidth: 840,
        scale: 34,
      }),
    );
    expect(toolkitMock.loadData).toHaveBeenCalledWith('<score-partwise/>');
    expect(toolkitMock.redoLayout).toHaveBeenCalledTimes(1);
    expect(toolkitMock.getPageCount).toHaveBeenCalledTimes(1);
    expect(toolkitMock.renderToSVG).toHaveBeenNthCalledWith(1, 1);
    expect(toolkitMock.renderToSVG).toHaveBeenNthCalledWith(2, 2);
    expect(svg).toBe('<svg data-page="1"></svg><svg data-page="2"></svg>');
  });

  it('keeps a minimum Verovio page width for narrow containers', () => {
    // 250.2px * 7.782 ≈ 1947 VU, which is less than A4 minimum (2100), so clamp to 2100
    expect(resolveVerovioPageWidth(250.2)).toBe(2100);
    // Wide containers should not expand the virtual page indefinitely.
    expect(resolveVerovioPageWidth(912.9)).toBe(2600);
  });

  it('chooses 4 to 6 measures per system from the available width', () => {
    expect(resolveMeasuresPerSystem(640)).toBe(4);
    expect(resolveMeasuresPerSystem(900)).toBe(5);
    expect(resolveMeasuresPerSystem(1280)).toBe(6);
  });

  it('extracts the rendered score title from part-name', () => {
    expect(
      getScoreTitle(
        '<score-partwise><part-list><score-part><part-name>Classical Guitar</part-name></score-part></part-list></score-partwise>',
      ),
    ).toBe('Classical Guitar');
  });

  it('removes inline part labels from the rendered sheet-music xml', () => {
    expect(
      prepareSheetMusicXml(
        '<score-partwise><part-list><score-part><part-name>Classical Guitar</part-name><part-abbreviation>Gtr.</part-abbreviation></score-part></part-list></score-partwise>',
      ),
    ).not.toContain('Classical Guitar');
  });

  it('inserts new-system breaks every few measures for generated scores', () => {
    const withBreaks = insertSystemBreaks(
      [
        '<score-partwise>',
        '  <part id="P1">',
        '    <measure number="1"><note/></measure>',
        '    <measure number="2"><note/></measure>',
        '    <measure number="3"><note/></measure>',
        '    <measure number="4"><note/></measure>',
        '    <measure number="5"><note/></measure>',
        '    <measure number="6"><note/></measure>',
        '    <measure number="7"><note/></measure>',
        '    <measure number="8"><note/></measure>',
        '    <measure number="9"><note/></measure>',
        '  </part>',
        '</score-partwise>',
      ].join(''),
      4,
    );

    expect(withBreaks).toContain('<measure number="5"><print new-system="yes"/><note/></measure>');
    expect(withBreaks).toContain('<measure number="9"><print new-system="yes"/><note/></measure>');
  });

  it('upgrades an existing print element instead of duplicating it', () => {
    const withBreaks = insertSystemBreaks(
      [
        '<score-partwise>',
        '  <part id="P1">',
        '    <measure number="1"><note/></measure>',
        '    <measure number="2"><note/></measure>',
        '    <measure number="3"><note/></measure>',
        '    <measure number="4"><note/></measure>',
        '    <measure number="5"><print><system-layout/></print><note/></measure>',
        '  </part>',
        '</score-partwise>',
      ].join(''),
      4,
    );

    expect(withBreaks).toContain('<measure number="5"><print new-system="yes"><system-layout/></print><note/></measure>');
    expect(withBreaks.match(/new-system="yes"/g)).toHaveLength(1);
  });
});
