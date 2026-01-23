import type {
  AltPositionsRequest,
  ApplyFingeringRequest,
  ApplyFingeringResponse,
  CommitDraftRequest,
  CommitDraftResponse,
  ComposeRequest,
  ComposeResponse,
  DiscardDraftRequest,
  DiscardDraftResponse,
  InpaintPreviewRequest,
  InpaintPreviewResponse,
} from './types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '';

const apiPost = async <T>(path: string, body: unknown): Promise<T> => {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body ?? {}),
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(
      message || `POST ${path} failed with status ${response.status}`,
    );
  }

  return (await response.json()) as T;
};

export const compose = (body: ComposeRequest = {}): Promise<ComposeResponse> =>
  apiPost<ComposeResponse>('/compose', body);

export const inpaintPreview = (
  body: InpaintPreviewRequest,
): Promise<InpaintPreviewResponse> =>
  apiPost<InpaintPreviewResponse>('/inpaint_preview', body);

export const commitDraft = (
  body: CommitDraftRequest,
): Promise<CommitDraftResponse> =>
  apiPost<CommitDraftResponse>('/commit_draft', body);

export const discardDraft = (
  body: DiscardDraftRequest,
): Promise<DiscardDraftResponse> =>
  apiPost<DiscardDraftResponse>('/discard_draft', body);

export const altPositions = (
  body: AltPositionsRequest,
): Promise<Record<string, unknown>> =>
  apiPost<Record<string, unknown>>('/alt_positions', body);

export const applyFingering = (
  body: ApplyFingeringRequest,
): Promise<ApplyFingeringResponse> =>
  apiPost<ApplyFingeringResponse>('/apply_fingering', body);
