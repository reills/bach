declare module 'verovio/wasm' {
  export default function createVerovioModule(): Promise<unknown>;
}

declare module 'verovio/esm' {
  export class VerovioToolkit {
    constructor(module: unknown);
    getPageCount(): number;
    loadData(data: string): boolean;
    redoLayout(options?: Record<string, unknown>): void;
    renderToSVG(pageNo?: number, xmlDeclaration?: boolean): string;
    setOptions(options: Record<string, unknown>): void;
  }
}
