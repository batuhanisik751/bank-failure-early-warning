/** Vocabulary shared by the query modules (CONTRACT 15). */
export const MODELS = ["gbdt_mono", "hazard"] as const;
export type Model = (typeof MODELS)[number];

export const BANDS = ["high", "elevated", "low"] as const;
export type Band = (typeof BANDS)[number];

export function isBand(value: unknown): value is Band {
  return typeof value === "string" && (BANDS as readonly string[]).includes(value);
}

export function isModel(value: unknown): value is Model {
  return typeof value === "string" && (MODELS as readonly string[]).includes(value);
}

/** The rate-shock grid published for the latest quarter (CONTRACT 16). */
export const SHOCK_BP = [100, 200, 300, 400] as const;
export const DURATION_YEARS = [2, 3, 4, 5, 6] as const;

/** Horizon of every published score: four quarters, the 12-month failure probability. */
export const HORIZON_QUARTERS = 4;
