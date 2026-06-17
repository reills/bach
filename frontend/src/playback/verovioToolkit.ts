import createVerovioModule from 'verovio/wasm';
import { VerovioToolkit } from 'verovio/esm';

export type VerovioToolkitLike = Pick<
  VerovioToolkit,
  'getElementsAtTime' | 'getPageCount' | 'loadData' | 'redoLayout' | 'renderToMIDI' | 'renderToSVG' | 'setOptions'
>;

let verovioToolkitPromise: Promise<VerovioToolkitLike> | null = null;

const createToolkit = async (): Promise<VerovioToolkitLike> => {
  const verovioModule = await createVerovioModule();
  return new VerovioToolkit(verovioModule);
};

export const getVerovioToolkit = (): Promise<VerovioToolkitLike> => {
  if (!verovioToolkitPromise) {
    verovioToolkitPromise = createToolkit();
  }
  return verovioToolkitPromise;
};

export const resetVerovioToolkitCache = (): void => {
  verovioToolkitPromise = null;
};
