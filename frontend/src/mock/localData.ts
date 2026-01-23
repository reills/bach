import type { MeasureMap } from '../state/types';

export interface LocalManifest {
  baseScore: string;
  snippets: string[];
}

export interface LocalDataBundle {
  baseXml: string;
  snippetXmls: string[];
  manifest: LocalManifest;
}

const parseXml = (xml: string): Document => {
  const parser = new DOMParser();
  const doc = parser.parseFromString(xml, 'application/xml');
  const error = doc.querySelector('parsererror');
  if (error) {
    throw new Error('Invalid XML content.');
  }
  return doc;
};

const ensureScoreDocument = (xml: string): Document => {
  const hasScore = xml.includes('<score-partwise');
  if (hasScore) {
    return parseXml(xml);
  }

  const wrapped = `<?xml version="1.0" encoding="UTF-8"?>\n<score-partwise version="3.1">\n  <part id="P1">\n${xml}\n  </part>\n</score-partwise>`;
  return parseXml(wrapped);
};

export const parseScoreXml = (xml: string): Document => ensureScoreDocument(xml);

const getFirstPart = (doc: Document): Element | null =>
  doc.querySelector('part');

export const buildMeasureMap = (doc: Document): MeasureMap => {
  const part = getFirstPart(doc);
  const map: MeasureMap = {};
  if (!part) {
    return map;
  }

  const measures = Array.from(part.querySelectorAll('measure'));
  measures.forEach((measure, index) => {
    const id = measure.getAttribute('xml:id') ?? `measure-${index + 1}`;
    map[String(index)] = id;
  });

  return map;
};

export const replaceMeasureAtIndex = (
  baseXml: string,
  barIndex: number,
  snippetXml: string,
): {
  xml: string;
  measureMap: MeasureMap;
  measureId: string | null;
} => {
  const baseDoc = ensureScoreDocument(baseXml);
  const snippetDoc = ensureScoreDocument(snippetXml);

  const basePart = getFirstPart(baseDoc);
  const snippetMeasure = snippetDoc.querySelector('measure');
  if (!basePart || !snippetMeasure) {
    throw new Error('Missing measure data for replacement.');
  }

  const baseMeasures = Array.from(basePart.querySelectorAll('measure'));
  const baseMeasure = baseMeasures[barIndex];
  if (!baseMeasure) {
    throw new Error(`No measure found at index ${barIndex}.`);
  }

  const newMeasure = baseDoc.importNode(snippetMeasure, true) as Element;
  const baseNumber = baseMeasure.getAttribute('number');
  const baseId = baseMeasure.getAttribute('xml:id');
  if (baseNumber) {
    newMeasure.setAttribute('number', baseNumber);
  }
  if (baseId) {
    newMeasure.setAttribute('xml:id', baseId);
  }

  if (!newMeasure.querySelector('attributes')) {
    const baseAttributes = baseMeasure.querySelector('attributes');
    if (baseAttributes) {
      newMeasure.insertBefore(
        baseDoc.importNode(baseAttributes, true),
        newMeasure.firstChild,
      );
    }
  }

  baseMeasure.replaceWith(newMeasure);

  return {
    xml: new XMLSerializer().serializeToString(baseDoc),
    measureMap: buildMeasureMap(baseDoc),
    measureId: baseId ?? null,
  };
};

const fetchText = async (path: string): Promise<string> => {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load ${path}`);
  }
  return response.text();
};

export const loadLocalData = async (): Promise<LocalDataBundle> => {
  let manifest: LocalManifest;
  try {
    const res = await fetch('/test-data/manifest.json');
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    }
    manifest = await res.json();
  } catch (error) {
    console.error('Failed to load manifest:', error);
    throw new Error(`Cannot load /test-data/manifest.json: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }

  if (!manifest.baseScore || !manifest.snippets?.length) {
    throw new Error('Manifest must include baseScore and snippets.');
  }

  console.log('Loading base score:', manifest.baseScore);
  const baseXml = await fetchText(`/test-data/${manifest.baseScore}`);
  console.log('Base XML loaded, length:', baseXml.length);

  console.log('Loading snippets:', manifest.snippets);
  const snippetXmls = await Promise.all(
    manifest.snippets.map((snippet) => fetchText(`/test-data/${snippet}`)),
  );
  console.log('All snippets loaded:', snippetXmls.length);

  return {
    baseXml,
    snippetXmls,
    manifest,
  };
};

export const pickRandomSnippet = (snippets: string[]): string | null => {
  if (!snippets.length) {
    return null;
  }
  const index = Math.floor(Math.random() * snippets.length);
  return snippets[index];
};
