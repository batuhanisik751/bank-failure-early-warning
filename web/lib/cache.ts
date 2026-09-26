import { unstable_cache } from "next/cache";

/** The one cache tag every query carries; POST /api/revalidate expires it. */
export const DATA_TAG = "data";
/** Seconds before a cached query result is refreshed on its own (CONTRACT 18). */
export const DATA_TTL_SECONDS = 3600;

/**
 * Wrap a server-only query so its result is shared across requests for an hour and can be
 * expired on demand. Results must be JSON-serialisable (plain objects, strings for dates),
 * which the schema guarantees. `name` keeps two queries with the same argument list apart.
 */
export function cached<Args extends unknown[], Result>(
  name: string,
  fn: (...args: Args) => Promise<Result>,
): (...args: Args) => Promise<Result> {
  return unstable_cache(fn, [name], { tags: [DATA_TAG], revalidate: DATA_TTL_SECONDS });
}
