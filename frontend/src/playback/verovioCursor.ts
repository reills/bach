export interface VerovioElementsAtTime {
  chords?: string[];
  measure?: string;
  notes?: string[];
  page?: number;
  rests?: string[];
}

export const getVerovioCursorIds = (
  elementsAtTime: VerovioElementsAtTime | null | undefined,
): string[] => {
  if (!elementsAtTime) {
    return [];
  }

  return Array.from(
    new Set([
      ...(elementsAtTime.notes ?? []),
      ...(elementsAtTime.chords ?? []),
      ...(elementsAtTime.rests ?? []),
    ].filter((id): id is string => typeof id === 'string' && id.length > 0)),
  );
};
