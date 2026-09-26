import "server-only";
import { drizzle, type NodePgDatabase } from "drizzle-orm/node-postgres";
import { Pool } from "pg";
import * as schema from "./schema";

export type Database = NodePgDatabase<typeof schema>;

const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1", "[::1]", "::1"]);

function isLocal(url: string): boolean {
  try {
    return LOCAL_HOSTS.has(new URL(url).hostname);
  } catch {
    return false;
  }
}

function makePool(): Pool {
  const url = process.env.DATABASE_URL;
  if (!url) throw new Error("DATABASE_URL is not set");
  return new Pool({
    connectionString: url,
    // Anything that is not the local container talks TLS with full verification
    // (CONTRACT 18: Neon pooled URL, sslmode=verify-full), whatever the URL says.
    ssl: isLocal(url) ? undefined : { rejectUnauthorized: true },
    max: 5,
    idleTimeoutMillis: 30_000,
    connectionTimeoutMillis: 10_000,
  });
}

// One pool per process; survives hot reloads in development.
const globalRef = globalThis as unknown as { __bankcanaryDb?: Database };

/** The read-only database handle. Only lib/queries/*.ts should import this. */
export function getDb(): Database {
  if (!globalRef.__bankcanaryDb) {
    globalRef.__bankcanaryDb = drizzle(makePool(), { schema });
  }
  return globalRef.__bankcanaryDb;
}

export { schema };
