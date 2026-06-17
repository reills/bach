import { describe, expect, it } from 'vitest';
import { createStandardNotationXml } from './localData';

const GUITAR_XML = `<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part id="P1">
    <measure number="1" xml:id="measure-1">
      <attributes>
        <divisions>4</divisions>
        <staff-details>
          <staff-lines>6</staff-lines>
          <staff-tuning line="1">
            <tuning-step>E</tuning-step>
            <tuning-octave>4</tuning-octave>
          </staff-tuning>
        </staff-details>
      </attributes>
      <note>
        <pitch><step>E</step><octave>4</octave></pitch>
        <duration>4</duration>
        <type>quarter</type>
        <notations>
          <technical>
            <string>1</string>
            <fret>0</fret>
          </technical>
        </notations>
      </note>
    </measure>
  </part>
</score-partwise>`;

describe('createStandardNotationXml', () => {
  it('removes guitar tab staff metadata from the score view xml', () => {
    const result = createStandardNotationXml(GUITAR_XML);

    expect(result).not.toContain('<staff-details>');
    expect(result).not.toContain('<staff-lines>6</staff-lines>');
    expect(result).not.toContain('<staff-tuning');
  });

  it('removes string and fret technical notation from the score view xml', () => {
    const result = createStandardNotationXml(GUITAR_XML);

    expect(result).not.toContain('<string>1</string>');
    expect(result).not.toContain('<fret>0</fret>');
    expect(result).not.toContain('<technical>');
    expect(result).not.toContain('<notations>');
  });
});
