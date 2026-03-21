import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import ScoreViewer from './ScoreViewer';

const renderTab = (viewTab: 'score' | 'tab' = 'tab') =>
  renderToStaticMarkup(
    <ScoreViewer
      scoreXml="<score-partwise version='3.1'/>"
      highlightMeasureId={null}
      viewTab={viewTab}
      onViewTabChange={() => {}}
    />,
  );

describe('ScoreViewer (guitar tab renderer)', () => {
  it('renders the Guitar Tab label', () => {
    expect(renderTab('tab')).toContain('Guitar Tab');
  });

  it('always shows the Sheet Music / Guitar Tab switcher', () => {
    const markup = renderTab('tab');
    expect(markup).toContain('Sheet Music');
    expect(markup).toContain('Guitar Tab');
  });

  it('marks Guitar Tab as active when viewTab is tab', () => {
    const markup = renderTab('tab');
    // The Guitar Tab button should have aria-selected="true"
    expect(markup).toMatch(/Guitar Tab[\s\S]*?aria-selected="true"|aria-selected="true"[\s\S]*?Guitar Tab/);
  });

  it('marks Sheet Music as active when viewTab is score', () => {
    const markup = renderTab('score');
    // The Sheet Music button should have aria-selected="true"
    expect(markup).toMatch(/Sheet Music[\s\S]*?aria-selected="true"|aria-selected="true"[\s\S]*?Sheet Music/);
  });
});
